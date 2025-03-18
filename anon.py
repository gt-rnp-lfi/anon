#!/usr/bin/env python3

import argparse
import concurrent.futures
import hashlib
import ipaddress
import logging
import os
import re
import sqlite3
import subprocess
from contextlib import contextmanager
from datetime import datetime as dt

import pandas as pd
import spacy
from docx import Document
from spacy.language import Language
from spacy.tokens import Span
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Globais
DB_DIR = os.path.join(os.getcwd(), "db")
DB_NAME = "entities.db"
DB_PATH = os.path.join(DB_DIR, DB_NAME)

MODEL_NAME = "Davlan/xlm-roberta-large-ner-hrl"
MODEL_DIR = os.path.join("models", MODEL_NAME)


def setup_logging(input_filename, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    base_filename = os.path.basename(input_filename)
    file_name, file_ext = os.path.splitext(base_filename)
    if file_ext.startswith("."):
        file_ext = file_ext[1:]
    timestamp = dt.now().strftime("%d-%m-%H-%M")
    log_filename = f"anon-{file_name}-{file_ext}-{timestamp}.log"
    log_file = os.path.join(log_dir, log_filename)

    logger = logging.getLogger("anonymizer")
    logger.setLevel(logging.INFO)
    # Clear any existing handlers
    logger.handlers = []

    # Add file handler only, with plain formatting
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


# Color-coding helper functions
def blue(text):
    return ("\033[34m{}\033[0m".format(text), logging.INFO)


def orange(text):
    return ("\033[33m{}\033[0m".format(text), logging.INFO)


def red(text):
    return ("\033[31m{}\033[0m".format(text), logging.WARNING)


class ColorLogger:
    def __init__(self, logger):
        self.logger = logger

    def log(self, color_tuple):
        colored_text, level = color_tuple
        # Remove ANSI color codes for file logging
        plain_text = re.sub(r"\033\[\d+m", "", colored_text)

        # Log plain text to file via logger
        if level == logging.INFO:
            self.logger.info(plain_text)
        elif level == logging.WARNING:
            self.logger.warning(plain_text)
        elif level == logging.ERROR:
            self.logger.error(plain_text)

        # Print colored text to the console
        print(colored_text)


# Custom spaCy components
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


def get_spacy_nlp():
    """
    Load the spaCy model
    Configure the pipeline with `entity_ruler`
    Add regex patterns for EMAIL, IPV4, IPV6, IPV4-CIDR, and IPV6-CIDR.
    """

    # Garantindo que o spaCy ok
    SPACY_MODEL = "pt_core_news_lg"
    SPACY_DIR = os.path.join("models", SPACY_MODEL)
    os.makedirs(SPACY_DIR, exist_ok=True)
    try:
        nlp = spacy.load(SPACY_DIR)
        print(f"[*] Modelo spaCy carregado de {SPACY_DIR}")
    except OSError:
        print(f"[*] Modelo {SPACY_MODEL} não encontrado. Baixando...")
        # Baixar modelo via subprocess para evitar erro de permissão
        subprocess.run(["spacy", "download", SPACY_MODEL], check=True)
        # Copiar modelo baixado para a pasta local
        subprocess.run(
            [
                "cp",
                "-r",
                os.path.expanduser(f"~/.cache/huggingface/hub/{SPACY_MODEL}"),
                SPACY_DIR,
            ]
        )
        # Carregar novamente
        nlp = spacy.load(SPACY_MODEL)
        print(f"[*] Modelo {SPACY_MODEL} baixado e carregado.")

    ruler = nlp.add_pipe("entity_ruler", before="parser")
    patterns = [
        # Email: simple pattern with word boundaries
        {
            "label": "EMAIL",
            "pattern": [{"TEXT": {"REGEX": r"\b[\w\.-]+@[\w\.-]+\.\w+\b"}}],
        },
        # IPV4-CIDR: IPv4 followed by a slash and one or two digits
        {
            "label": "IPV4-CIDR",
            "pattern": [{"TEXT": {"REGEX": r"\b(?:\d{1,3}\.){3}\d{1,3}/\d{1,2}\b"}}],
        },
        # IPV6-CIDR: IPv6 followed by a slash and one to three digits
        {
            "label": "IPV6-CIDR",
            "pattern": [
                {
                    "TEXT": {
                        "REGEX": r"\b(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}/\d{1,3}\b"
                    }
                }
            ],
        },
        # IPV4: a standard IPv4 address
        {
            "label": "IPV4",
            "pattern": [{"TEXT": {"REGEX": r"\b(?:\d{1,3}\.){3}\d{1,3}\b"}}],
        },
        # IPV6: a standard IPv6 address (simplified version)
        {
            "label": "IPV6",
            "pattern": [
                {"TEXT": {"REGEX": r"\b(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}\b"}}
            ],
        },
    ]
    ruler.add_patterns(patterns)
    nlp.add_pipe("relabel_ip_entities", after="parser")
    nlp.add_pipe("merge_entities", last=True)

    # Configurações dos modelos

    # Verificar se o modelo já foi baixado
    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        print(f"[*] Baixando o modelo {MODEL_NAME} para {MODEL_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

        # Salvar modelo localmente
        tokenizer.save_pretrained(MODEL_DIR)
        model.save_pretrained(MODEL_DIR)
        print(f"[*] Modelo salvo em {MODEL_DIR}")
    else:
        print(f"[*] Carregando o modelo {MODEL_NAME} de {MODEL_DIR}...")

    # Carregar modelo localmente
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    ner_pipeline = pipeline(
        "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
    )

    return nlp, ner_pipeline


class DataAnonymizer:
    def __init__(self, db_path, nlp, ner_pipeline, logger):
        self.db_path = db_path
        self.logger = logger
        self.nlp = nlp
        self.ner_pipeline = ner_pipeline
        self.hit_counts = {
            "IPV4": 0,
            "IPV6": 0,
            "EMAIL": 0,
            "PER": 0,  # PERSON
            "ORG": 0,  # ORGANIZATION
            "LOC": 0,  # LOCATION
            "MISC": 0,
        }

    @contextmanager
    def get_connection(self):
        """Manager for SQLite connection"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            yield conn
        finally:
            conn.close()

    def hash_value(self, value):
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def save_to_db(self, entity_type, original_name, full_hash):
        slug_name = f"[{entity_type}-{full_hash[:5]}]"

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO entities (entity_type, original_name, slug_name, full_hash, first_seen, last_seen)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(full_hash) DO UPDATE SET last_seen = CURRENT_TIMESTAMP;
                """,
                (entity_type, original_name, slug_name, full_hash),
            )
            conn.commit()

    def anonymize_text(self, text):
        CONFIDENCE_THRESHOLD = 0.99
        ALLOWED_ENTITIES = {"PER", "ORG", "EMAIL", "IPV4", "IPV6", "CIDR_V4", "CIDR_V6"}
        # Executar a pipeline do spaCy primeiro (regras personalizadas)
        doc = self.nlp(text)
        spacy_entities = [
            {
                "start": ent.start_char,
                "end": ent.end_char,
                "word": ent.text,
                "entity_group": ent.label_,
            }
            for ent in doc.ents
        ]

        # Executar o modelo transformers para detecção de NER
        transformers_entities = [
            ent
            for ent in self.ner_pipeline(text)
            if ent["score"] >= CONFIDENCE_THRESHOLD
        ]

        # Combinar as entidades das duas fontes
        all_entities = [
            ent
            for ent in (spacy_entities + transformers_entities)
            if ent["entity_group"] in ALLOWED_ENTITIES
        ]
        all_entities = sorted(all_entities, key=lambda x: x["start"])

        # Substituir entidades no texto original
        replacements = []
        last = 0
        for ent in all_entities:
            start, end, ent_text, label = (
                ent["start"],
                ent["end"],
                ent["word"],
                ent["entity_group"],
            )

            # Preservar texto antes da entidade
            replacements.append(text[last:start])

            # Criar hash da entidade
            hash_val = self.hash_value(ent_text)
            self.save_to_db(label, ent_text, hash_val)

            # Substituir entidade por slug
            replacement = f"[{label}-{hash_val[:5]}]"
            self.logger.log(red(f"Hit ({label}): {ent_text} -> {replacement}"))
            replacements.append(replacement)

            last = end

        # Adicionar o restante do texto
        replacements.append(text[last:])
        return "".join(replacements)

    def anonymize_cell(self, value, row_idx=None, column_name=None):
        """Anonymize a single cell from tabular data"""
        if not isinstance(value, str):
            return value
        return self.anonymize_text(value)

    def print_hit_summary(self):
        """Print a summary of all anonymization hits"""
        ip_total = self.hit_counts["IPV4"] + self.hit_counts["IPV6"]
        email_total = self.hit_counts["EMAIL"]
        spacy_total = (
            self.hit_counts["PER"]
            + self.hit_counts["ORG"]
            + self.hit_counts["LOC"]
            + self.hit_counts["MISC"]
        )
        grand_total = ip_total + email_total + spacy_total

        if grand_total == 0:
            self.logger.log(orange("Nenhum hit encontrado"))
            return

        summary = "\nResumo da anonimização\n"
        summary += "-----------------------\n"
        summary += f"IPv4 encontrados      | {self.hit_counts['IPV4']:3}\n"
        summary += f"IPv6 encontrados      | {self.hit_counts['IPV6']:3}\n"
        summary += f"E-mails encontrados   | {self.hit_counts['EMAIL']:3}\n"
        summary += f"Pessoas      (PER)    | {self.hit_counts['PER']:3}\n"
        summary += f"Organizações (ORG)    | {self.hit_counts['ORG']:3}\n"
        summary += f"Localizações (LOC)    | {self.hit_counts['LOC']:3}\n"
        summary += f"Outros       (MISC)   | {self.hit_counts['MISC']:3}\n"
        summary += f"Total de hits         | {grand_total:3}\n"
        self.logger.log(orange(summary))


class FileProcessor:
    """Coordinatinates Anonymization and Reporting"""

    def __init__(self, anonymizer, logger):
        self.logger = logger
        self.anonymizer = anonymizer
        self.db_dir = os.path.join(os.getcwd(), "db")
        self.output_dir = os.path.join(os.getcwd(), "output")
        self.logs_dir = os.path.join(os.getcwd(), "logs")
        self._ensure_directories_exist()

    def _ensure_directories_exist(self):
        directories = [self.db_dir, self.output_dir, self.logs_dir]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.log(blue(f"Diretório verificado/criado: {directory}"))

    def _setup_database(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_type   TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    slug_name     TEXT NOT NULL,
                    full_hash     TEXT NOT NULL UNIQUE,
                    first_seen    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_seen      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()

    def process_file(self, file_path):
        file_path = os.path.abspath(file_path)
        if not os.path.isfile(file_path):
            self.logger.log(red(f"Erro: Arquivo '{file_path}' não encontrado"))
            return
        ext = os.path.splitext(file_path)[1].lower()
        self._setup_database()

        if ext in [".txt", ".docx"]:
            self._process_textual(file_path, ext)
        elif ext in [".xlsx", ".csv"]:
            self._process_tabular(file_path, ext)
        else:
            self.logger.log(red(f"Erro: Formato '{ext}' não suportado"))

    def _process_textual(self, file_path, ext):
        filename = os.path.basename(file_path)
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                # Read the file preserving its line structure
                lines = f.readlines()
            content = "".join(lines)
        elif ext == ".docx":
            doc = Document(file_path)
            lines = [p.text for p in doc.paragraphs]
            content = "\n".join(lines)
        else:
            self.logger.log(red(f"Erro: Formato '{ext}' não suportado para textos"))
            return

        self.logger.log(blue(f"\nProcessando {filename}...\n"))
        anonymized_content = self.anonymizer.anonymize_text(content)

        relative_filepath = os.path.relpath(file_path).replace(".", "-")
        safe_filepath = (
            relative_filepath.replace(" ", "-")
            .replace("/", "-")
            .replace("\\", "-")
            .replace(".", "-")
            .strip()
        )
        output_path = os.path.join(self.output_dir, f"{safe_filepath}-anon.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(anonymized_content)

        self.logger.log(blue(f"\nArquivo original: {file_path}"))
        self.logger.log(blue(f"Arquivo anonimizado: {output_path}\n"))
        self.anonymizer.print_hit_summary()

    def _process_tabular(self, file_path, ext):
        filename = os.path.basename(file_path)
        self.logger.log(blue(f"\nProcessando {filename}...\n"))

        if ext == ".xlsx":
            df = pd.read_excel(file_path)
        elif ext == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8")
        else:
            self.logger.log(red(f"Erro: Formato '{ext}' não suportado para tabelas"))
            return

        df_anon = self._process_dataframe_parallel(df)
        relative_filepath = os.path.relpath(file_path).replace(".", "-")
        safe_filepath = (
            relative_filepath.replace(" ", "-")
            .replace("/", "-")
            .replace("\\", "-")
            .replace(".", "-")
            .strip()
        )
        output_path = os.path.join(self.output_dir, f"{safe_filepath}-anon.csv")
        df_anon.to_csv(output_path, index=False, encoding="utf-8")

        self.logger.log(blue(f"\nArquivo original: {file_path}"))
        self.logger.log(blue(f"Arquivo anonimizado: {output_path}\n"))
        self.anonymizer.print_hit_summary()

    def _process_dataframe_parallel(self, df):
        column_names = df.columns.tolist()

        def process_row(row_tuple):
            idx, row = row_tuple
            processed_row = {}
            for col in column_names:
                processed_row[col] = self.anonymizer.anonymize_cell(row[col], idx, col)
            return processed_row

        with concurrent.futures.ThreadPoolExecutor() as executor:
            row_tuples = list(enumerate(df.iterrows()))
            processed_rows = list(
                executor.map(lambda rt: process_row((rt[0], rt[1][1])), row_tuples)
            )

        return pd.DataFrame(processed_rows, columns=df.columns)


def main():
    parser = argparse.ArgumentParser(
        description="Anon: Ferramenta de anonimização de dados sensíveis"
    )
    parser.add_argument("input_file", type=str, help="caminho do arquivo a processar")
    args = parser.parse_args()

    logger_instance = setup_logging(args.input_file)
    log_instance = ColorLogger(logger_instance)
    nlp, ner_pipeline = get_spacy_nlp()

    anonymizer = DataAnonymizer(DB_PATH, nlp, ner_pipeline, log_instance)
    processor = FileProcessor(anonymizer, log_instance)
    try:
        processor.process_file(args.input_file)
    finally:
        pass


if __name__ == "__main__":
    main()
