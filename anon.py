#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import os
import re
import sqlite3

from docx import Document
from email_validator import EmailNotValidError, validate_email


def blue(text):
    """Formata o texto em azul com ANSI escape sequences"""
    return f"\033[34m{text}\033[0m"


def red(text):
    """Formata o texto em vermelho com ANSI escape sequences"""
    return f"\033[31m{text}\033[0m"


class FileProcessor:
    """Classe para processar arquivos de texto e tabulares, anonimizando dados sensíveis"""

    def __init__(self):
        self.db_dir = os.path.join(os.getcwd(), "db")
        self.db_path = os.path.join(self.db_dir, "anon.db")
        self.setup_database()

        # Criar diretório de saída, se não existir
        self.output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def setup_database(self):
        """Configura o banco de dados SQLite, criando a pasta e o arquivo se não existirem"""
        os.makedirs(self.db_dir, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")  # Write-Ahead Logging
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS anon_pairs (
                type TEXT NOT NULL,
                original TEXT NOT NULL,
                hash TEXT PRIMARY KEY
            );
            """
        )
        self.conn.commit()

    def save_to_db(self, data_type, hash_value, original):
        """Salva (tipo, hash, original) no banco de dados, evitando duplicações"""
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO anon_pairs (type, original, hash) VALUES (?, ?, ?);",
                (data_type, original, hash_value),
            )

    def close(self):
        """Fecha a conexão com o banco de dados"""
        self.conn.close()

    def hash_value(self, value):
        """Gera um hash SHA-256 a partir do valor original"""
        hash_object = hashlib.sha256(value.encode("utf-8"))
        return hash_object.hexdigest()

    def anonymize_text(self, text):
        """Anonimiza IPs e e-mails em um texto, registrando no banco de dados"""
        ip_pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
        email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

        def replace_ip(match):
            original_ip = match.group(0)
            hash_value = self.hash_value(original_ip)
            self.save_to_db("ip", hash_value, original_ip)
            print(red(f"Hit: {original_ip}"))
            return f"ip-anon-{hash_value[:8]}"

        def replace_email(match):
            original_email = match.group(0)
            try:
                valid = validate_email(original_email, check_deliverability=False)
                normalized = valid.normalized
                hash_value = self.hash_value(normalized)
                self.save_to_db("email", hash_value, normalized)
                print(red(f"Hit: {original_email}"))
                return f"email-anon-{hash_value[:8]}"
            except EmailNotValidError:
                return original_email  # Mantém e-mails inválidos inalterados

        # Substituir IPs e e-mails no texto
        text = ip_pattern.sub(replace_ip, text)
        text = email_pattern.sub(replace_email, text)

        return text

    def process_textual(self, file_path, ext):
        """Processa arquivos textuais (.txt, .docx) e anonimiza seu conteúdo"""
        filename = os.path.basename(file_path)
        filename_no_ext = os.path.splitext(filename)[0]

        # Extrair o conteúdo do arquivo para `content`
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as file_obj:
                content = file_obj.read()
        elif ext == ".docx":
            doc = Document(file_path)
            content = "\n".join([p.text for p in doc.paragraphs])
        else:
            print(red(f"Erro: Formato '{ext}' não suportado para arquivos textuais"))
            return

        print(blue(f"\nProcessando {filename}...\n"))

        anonymized_content = self.anonymize_text(content)

        # Salvar a saída anonimizada em um arquivo .txt
        output_path = os.path.join(self.output_dir, f"{filename_no_ext}-anon.txt")
        with open(output_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(anonymized_content)

        print(blue(f"\nArquivo original mantido em: {file_path}"))
        print(blue(f"Arquivo anonimizado salvo em: {output_path}\n"))

    def process_file(self, file_path):
        """Processa um arquivo baseado em sua extensão"""
        file_path = os.path.abspath(file_path)
        if not os.path.isfile(file_path):
            print(red(f"Erro: Arquivo '{file_path}' não encontrado"))
            return

        ext = os.path.splitext(file_path)[1].lower()

        if ext in [".txt", ".docx"]:
            self.process_textual(file_path, ext)
        elif ext in [".xlsx", ".csv"]:
            self.process_tabular(file_path, ext)
        else:
            print(red(f"Erro: Formato '{ext}' não suportado"))


def main():
    parser = argparse.ArgumentParser(
        description="Anon: Ferramenta de anonimização de tickets de segurança"
    )

    parser.add_argument("input_file", type=str, help="caminho do arquivo de entrada")

    args = parser.parse_args()

    processor = FileProcessor()
    try:
        processor.process_file(args.input_file)
    finally:
        processor.close()


if __name__ == "__main__":
    main()
