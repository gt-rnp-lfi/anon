#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import concurrent.futures
import hashlib
import ipaddress
import os
import re
import sqlite3
import threading

import pandas as pd
import spacy
from docx import Document
from email_validator import EmailNotValidError, validate_email
from spacy.language import Language
from spacy.tokens import Span


@Language.component("relabel_ip_entities")
def relabel_ip_entities(doc):
    new_ents = []
    for ent in doc.ents:
        label = ent.label_
        try:
            if "/" in ent.text:
                network = ipaddress.ip_network(ent.text, strict=False)
                label = (
                    "CIDR_V4"
                    if isinstance(network, ipaddress.IPv4Network)
                    else "CIDR_V6"
                )
            else:
                ip = ipaddress.ip_address(ent.text)
                label = "IPV4" if isinstance(ip, ipaddress.IPv4Address) else "IPV6"
        except ValueError:
            pass
        new_ents.append(Span(doc, ent.start, ent.end, label=label))
    doc.ents = new_ents
    return doc


@Language.component("merge_entities")
def merge_entities(doc):
    merged_ents = []
    buffer = []

    def flush_buffer():
        if buffer:
            start, end = buffer[0][0], buffer[-1][1]
            merged_ents.append(Span(doc, start, end, label="ENDERECO"))
            buffer.clear()

    for ent in doc.ents:
        if ent.label_ in {"LOC", "GPE", "ENDERECO"}:
            buffer.append((ent.start, ent.end))
        else:
            flush_buffer()
            merged_ents.append(ent)
    flush_buffer()
    doc.ents = merged_ents
    return doc


# Configuração do pipeline spaCy para processamento de endereços e demais entidades
nlp_spacy = spacy.load("pt_core_news_lg")
ruler = nlp_spacy.add_pipe("entity_ruler", before="ner")
ruler.add_patterns(
    [
        {
            "label": "EMAIL",
            "pattern": [{"TEXT": {"REGEX": r"^[\w\.-]+@[\w\.-]+\.\w+$"}}],
        },
        {"label": "IP", "pattern": [{"TEXT": {"REGEX": r"^\d{1,3}(\.\d{1,3}){3}$"}}]},
        {
            "label": "IP",
            "pattern": [
                {"TEXT": {"REGEX": r"^([0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}$"}}
            ],
        },
    ]
)
nlp_spacy.add_pipe("relabel_ip_entities", after="ner")
nlp_spacy.add_pipe("merge_entities", last=True)


def blue(text):
    """Colore o texto de azul - pra informações"""
    return f"\033[34m{text}\033[0m"


def red(text):
    """Colore o texto de vermelho - pra hits e erros"""
    return f"\033[31m{text}\033[0m"


def orange(text):
    """Colore o texto de laranja - pro resumo de hits"""
    return f"\033[33m{text}\033[0m"


class DataAnonymizer:
    """Classe responsável pela anonimização de dados sensíveis"""

    EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

    def __init__(self, db_path):
        """Inicializa o anonimizador e os contadores de hits"""
        self.db_path = db_path
        # armazenar conexões específicas de thread (sqlite aceita apenas uma por thread)
        self.local = threading.local()
        # contador para hits de cada tipo de dado anonimizado
        self.hit_counts = {"ipv4": 0, "ipv6": 0, "email": 0}
        # modelo de linguagem para processamento de linguagem natural (PLN)
        nlp_spacy = spacy.load("pt_core_news_lg")

    def get_connection(self):
        """Retorna uma conexão para a thread atual ou cria uma nova"""
        if not hasattr(self.local, "conn"):
            self.local.conn = sqlite3.connect(self.db_path)
            self.local.conn.execute("PRAGMA journal_mode=WAL;")
        return self.local.conn

    def close(self):
        """Fecha a conexão com o banco de dados desta thread, se existir"""
        if hasattr(self.local, "conn"):
            try:
                self.local.conn.close()
                del self.local.conn
            except Exception as e:
                print(red(f"Erro ao fechar conexão com o banco: {str(e)}"))

    def hash_value(self, value):
        """Cria um hash SHA-256 do valor original"""
        hash_object = hashlib.sha256(value.encode("utf-8"))
        return hash_object.hexdigest()

    def save_to_db(self, data_type, original, hash_value):
        """Salva a tupla (tipo, original, hash) no banco, ignorando duplicatas"""
        conn = self.get_connection()
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO anon_pairs (type, original, hash) VALUES (?, ?, ?);",
                (data_type, original, hash_value),
            )

    def _anonymize_ip_addresses(self, text, line_info="", column_info=""):
        """Anonimiza endereços IPv4 e IPv6 usando o módulo ipaddress"""

        words = text.split()
        anonymized_words = []

        for word in words:
            # Tenta limpar a palavra para lidar com IPs que podem fazer parte do texto
            # Remove pontuações comuns que podem estar no final de um IP
            cleaned_word = word.strip('.,;:()[]{}"\'"')

            try:
                # Tenta analisar como um endereço IP (funciona para IPv4 e IPv6)
                ip = ipaddress.ip_address(cleaned_word)

                # Se chegarmos aqui, é um endereço IP válido
                original_ip = cleaned_word
                ip_type = "ipv4" if ip.version == 4 else "ipv6"
                # Incrementar o contador de hits
                self.hit_counts[ip_type] += 1
                hash_value = self.hash_value(original_ip)
                self.save_to_db(ip_type, original_ip, hash_value)

                hit_msg = f"Hit ({ip_type}): {original_ip}{line_info}{column_info}"
                print(red(hit_msg))

                # Construir um slug com os primeiros 8 caracteres do hash
                replacement = f"{ip_type}-anon-{hash_value[:8]}"
                anonymized_word = word.replace(cleaned_word, replacement)
                anonymized_words.append(anonymized_word)
            except ValueError:
                # Não é um endereço IP válido
                anonymized_words.append(word)

        return " ".join(anonymized_words)

    def _anonymize_email(self, text, line_info="", column_info=""):
        """Anonimiza endereços de e-mail no texto"""
        hits = []

        def replace_email(match):
            original_email = match.group(0)
            try:
                valid = validate_email(original_email, check_deliverability=False)
                # Se chegarmos aqui, é um e-mail válido, incrementar o contador de hits
                self.hit_counts["email"] += 1
                normalized = valid.normalized
                hash_value = self.hash_value(normalized)
                self.save_to_db("email", normalized, hash_value)
                hit_msg = f"Hit (e-mail): {original_email}{line_info}{column_info}"
                hits.append(hit_msg)
                return f"email-anon-{hash_value[:8]}"
            except EmailNotValidError:
                return original_email  # Mantém e-mails inválidos

        result = self.EMAIL_PATTERN.sub(replace_email, text)

        # Exibir os hits encontrados
        for hit in hits:
            print(red(hit))

        return result

    def _anonymize_with_spacy(self, text):
        doc = nlp_spacy(text)
        for ent in doc.ents:
            print(red(f"Hit ({ent.label_}): {ent.text}"))
            text = text.replace(ent.text, f"{ent.label_}-anon")
        return text

    def anonymize_text(self, text, line_info="", column_info=""):
        """Anonimiza uma stream de texto"""
        text = self._anonymize_ip_addresses(text, line_info, column_info)
        text = self._anonymize_email(text, line_info, column_info)
        text = self._anonymize_with_spacy(text)

        return text

    def anonymize_cell(self, value, row_idx=None, column_name=None):
        """Anonimiza uma célula em dados tabulares"""
        if not isinstance(value, str):
            return value

        line_info = f" (linha {row_idx+1})" if row_idx is not None else ""
        column_info = f" (coluna '{column_name}')" if column_name is not None else ""
        return self.anonymize_text(value, line_info, column_info)

    def print_hit_summary(self):
        """Resumo final dos hits encontrados"""
        total = sum(self.hit_counts.values())
        if total == 0:
            print(orange("Nenhum hit encontrado"))
            return
        else:
            summary = (
                "\nResumo da anonimização:\n"
                f"IPv4 encontrados:   {self.hit_counts['ipv4']:3}\n"
                f"IPv6 encontrados:   {self.hit_counts['ipv6']:3}\n"
                f"E-mails encontrados:{self.hit_counts['email']:3}\n"
                f"Total de hits:      {total:3}\n"
            )
            print(orange(summary))


class FileProcessor:
    """Classe que processa arquivos e coordena a anonimização"""

    def __init__(self):
        """Configura diretórios, banco de dados e anonimizador"""
        # Definir os caminhos dos diretórios
        self.db_dir = os.path.join(os.getcwd(), "db")
        self.output_dir = os.path.join(os.getcwd(), "output")

        # Criar diretórios se não existirem
        self._ensure_directories_exist()

        # Configurar o banco de dados
        self.db_path = os.path.join(self.db_dir, "anon.db")
        self._setup_database()

        # Inicializar o anonimizador
        self.anonymizer = DataAnonymizer(self.db_path)

    def _ensure_directories_exist(self):
        """Garante que os diretórios necessários existam"""
        try:
            if not os.path.exists(self.db_dir):
                os.makedirs(self.db_dir)
                print(blue(f"Diretório criado: {self.db_dir}"))

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                print(blue(f"Diretório criado: {self.output_dir}"))
        except Exception as e:
            print(red(f"Erro ao criar diretórios: {str(e)}"))
            raise

    def _setup_database(self):
        """Configura o banco de dados SQLite inicialmente"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL;")  # Modo WAL para melhor performance
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS anon_pairs (
                    type TEXT NOT NULL,
                    original TEXT NOT NULL,
                    hash TEXT PRIMARY KEY
                );
                """
            )
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(red(f"Erro ao configurar o banco de dados: {str(e)}"))
            raise

    def close(self):
        """Fecha recursos e conexões"""
        if hasattr(self, "anonymizer"):
            try:
                self.anonymizer.close()
            except Exception as e:
                print(red(f"Erro ao fechar conexão com o banco: {str(e)}"))

    def process_file(self, file_path):
        """Processa um arquivo baseado em sua extensão"""
        try:
            file_path = os.path.abspath(file_path)
            if not os.path.isfile(file_path):
                print(red(f"Erro: Arquivo '{file_path}' não encontrado"))
                return

            ext = os.path.splitext(file_path)[1].lower()

            # Delegar para o método com base na extensão
            if ext in [".txt", ".docx"]:
                self._process_textual(file_path, ext)
            elif ext in [".xlsx", ".csv"]:
                self._process_tabular(file_path, ext)
            else:
                print(red(f"Erro: Formato '{ext}' não suportado"))
        except Exception as e:
            print(red(f"Erro ao processar arquivo: {str(e)}"))

    def _process_textual(self, file_path, ext):
        """Processa arquivos textuais (.txt, .docx)"""
        filename = os.path.basename(file_path)

        # Extrair o conteúdo do arquivo
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as file_obj:
                content = file_obj.read()
        elif ext == ".docx":
            doc = Document(file_path)
            content = "\n".join([p.text for p in doc.paragraphs])
        else:
            print(red(f"Erro: Formato '{ext}' não suportado para textos"))
            return

        print(blue(f"\nProcessando {filename}...\n"))

        # Anonimizar o conteúdo
        anonymized_content = self.anonymizer.anonymize_text(content)

        # Salvar a saída contendo o relpath
        relative_filepath = os.path.relpath(file_path).replace(".", "-")
        safe_filepath = (
            relative_filepath.replace(" ", "-")  # Espaços
            .replace("/", "-")  # Barras normais (Linux)
            .replace("\\", "-")  # Barras invertidas (Windows)
            .replace(".", "-")  # Pontos no geral
            .strip()  # Espaços no início/fim
        )
        output_path = os.path.join(self.output_dir, f"{safe_filepath}-anon.txt")
        with open(output_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(anonymized_content)

        # Avisar os resultados
        print(blue(f"\nArquivo original: {file_path}"))
        print(blue(f"Arquivo anonimizado: {output_path}\n"))
        self.anonymizer.print_hit_summary()

    def _process_tabular(self, file_path, ext):
        """Processa arquivos tabulares (.xlsx, .csv) com paralelismo"""
        filename = os.path.basename(file_path)

        print(blue(f"\nProcessando {filename}...\n"))

        # Ler o arquivo tabular
        if ext == ".xlsx":
            df = pd.read_excel(file_path)
        elif ext == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8")
        else:
            print(red(f"Erro: Formato '{ext}' não suportado para tabelas"))
            return

        # Processar cada linha em paralelo
        df_anon = self._process_dataframe_parallel(df)

        # Salvar a saída contendo o relpath
        relative_filepath = os.path.relpath(file_path).replace(".", "-")
        safe_filepath = (
            relative_filepath.replace(" ", "-")  # Espaços
            .replace("/", "-")  # Barras normais (Linux)
            .replace("\\", "-")  # Barras invertidas (Windows)
            .replace(".", "-")  # Pontos no geral
            .strip()  # Espaços no início/fim
        )
        output_path = os.path.join(self.output_dir, f"{safe_filepath}-anon.csv")
        df_anon.to_csv(output_path, index=False, encoding="utf-8")

        # Avisar os resultados
        print(blue(f"\nArquivo original: {file_path}"))
        print(blue(f"Arquivo anonimizado: {output_path}\n"))
        self.anonymizer.print_hit_summary()

    def _process_dataframe_parallel(self, df):
        """Processa um DataFrame com paralelismo por linha"""
        column_names = df.columns.tolist()

        # Processar cada linha
        def process_row(row_tuple):
            idx, row = row_tuple
            processed_row = pd.Series(index=row.index, dtype=str)

            # Processando cada célula com informação da coluna
            for col_name in column_names:
                processed_row[col_name] = self.anonymizer.anonymize_cell(
                    row[col_name], idx, col_name
                )

            return processed_row

        # "Paralelizar" com threads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # (índice, row_series) tuplas para cada linha
            row_tuples = list(enumerate(df.iterrows()))
            processed_rows = list(
                # rt[0] é o índice, rt[1][1] é a linha como Series
                executor.map(lambda rt: process_row((rt[0], rt[1][1])), row_tuples)
            )

        return pd.DataFrame(processed_rows, columns=df.columns)


def main():
    parser = argparse.ArgumentParser(
        description="Anon: Ferramenta de anonimização de dados sensíveis"
    )

    parser.add_argument("input_file", type=str, help="caminho do arquivo a processar")

    args = parser.parse_args()

    processor = FileProcessor()
    try:
        processor.process_file(args.input_file)
    finally:
        processor.close()


if __name__ == "__main__":
    main()
