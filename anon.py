#! /usr/bin/env python

import datetime
import hashlib
import os
import sqlite3
import sys
import time
import warnings

import numpy as np
import pandas as pd
import spacy
import torch
from docx import Document
from huggingface_hub import snapshot_download
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NerModelConfiguration, TransformersNlpEngine
from presidio_anonymizer import AnonymizerEngine, EngineResult, OperatorConfig
from presidio_anonymizer.operators import Operator, OperatorType
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Globals
ALLOW_LIST = ["TCP", "UDP", "HTTP", "HTTPS", "admin", "localhost"]
TRANSFORMER_MODEL = "Davlan/xlm-roberta-base-ner-hrl"
TRF_MODEL_PATH = os.path.join("models", TRANSFORMER_MODEL)

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Suppress warnings
warnings.filterwarnings("ignore")

# Entity mappings between the model's and Presidio's
ENTITY_MAPPING = dict(
    LOC="LOCATION",
    ORG="ORGANIZATION",
    PER="PERSON",
    EMAIL="EMAIL",
    PHONE="PHONE_NUMBER",
    PERSON="PERSON",
    LOCATION="LOCATION",
    GPE="LOCATION",
    ORGANIZATION="ORGANIZATION",
)


def initialize_db():
    db_dir = os.path.join(os.getcwd(), "db")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "entities.db")
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        # Little optimizations
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=10000;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT,
                original_name TEXT,
                slug_name TEXT,
                full_hash TEXT UNIQUE,
                first_seen TEXT,
                last_seen TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_full_hash ON entities(full_hash);")
        conn.commit()
    return db_path


# Should be global too
DB_PATH = initialize_db()


def save_entity(
    db_path: str, entity_type: str, original_name: str, slug_name: str, full_hash: str
) -> None:
    now = datetime.datetime.now().isoformat()
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        cur = conn.execute("SELECT id FROM entities WHERE full_hash=?", (full_hash,))
        row = cur.fetchone()
        if row:
            conn.execute("UPDATE entities SET last_seen=? WHERE id=?", (now, row[0]))
        else:
            conn.execute(
                "INSERT OR IGNORE INTO entities (entity_type, original_name, slug_name, full_hash, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?)",
                (entity_type, original_name.strip(), slug_name, full_hash, now, now),
            )
        conn.commit()


class ASNRecognizer(PatternRecognizer):
    PATTERNS = [Pattern("ASN", r"(?<=\W|^)AS[\s-_]*\d{1,10}(?=\W|$)", 0.5)]
    CONTEXT = []

    def __init__(self):
        super().__init__(supported_entity="AS_NUMBER", patterns=ASNRecognizer.PATTERNS)


class CustomSlugAnonymizer(Operator):
    # Strip before hashing to guarantee uniqueness
    def operate(self, text: str, params: dict = None) -> str:
        clean_text = " ".join(text.split()).strip()
        full_hash = hashlib.sha256(clean_text.encode()).hexdigest()
        slug = full_hash[:10]
        entity_type = params.get("entity_type", "UNKNOWN")
        save_entity(DB_PATH, entity_type, clean_text, slug, full_hash)
        return f"[{entity_type}_{slug}]"

    def validate(self, params: dict = None) -> None:
        pass

    def operator_name(self) -> str:
        return "custom_slug"

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize


def transformer_model_config():
    # Intantiate tokenizer and (transformer) model
    tokenizer = AutoTokenizer.from_pretrained(
        TRANSFORMER_MODEL, cache_dir=TRF_MODEL_PATH
    )
    model = AutoModelForTokenClassification.from_pretrained(
        TRANSFORMER_MODEL, cache_dir=TRF_MODEL_PATH
    )
    model.to("cuda:0" if torch.cuda.is_available() else "cpu")

    # Transformer model config
    trf_model_config = [
        {
            "lang_code": "pt",
            "model_name": {
                "spacy": "pt_core_news_lg",
                "transformers": TRANSFORMER_MODEL,  # Used only for NER
            },
        }
    ]

    # Transformer NER config
    ner_model_configuration = NerModelConfiguration(
        model_to_presidio_entity_mapping=ENTITY_MAPPING,
        alignment_mode="expand",  # "strict", "contract", "expand"
        aggregation_strategy="max",  # "simple", "first", "average", "max"
        labels_to_ignore=["O"],
    )

    return trf_model_config, ner_model_configuration


def get_presidio_engines(trf_model_config, ner_model_config):
    # Wrapper on spacy functionality
    transformers_nlp_engine = TransformersNlpEngine(
        models=trf_model_config, ner_model_configuration=ner_model_config
    )

    # Analyzer Engine config
    analyzer_engine = AnalyzerEngine(
        nlp_engine=transformers_nlp_engine,
        supported_languages=["pt", "en"],
        log_decision_process=False,
    )
    analyzer_engine.registry.add_recognizer(ASNRecognizer())

    # Anonymizer Engine config
    anonymizer_engine = AnonymizerEngine()
    anonymizer_engine.add_anonymizer(CustomSlugAnonymizer)

    return analyzer_engine, anonymizer_engine


def batch_process_text(texts, analyzer_engine, anonymizer_engine, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Convert to strings and handle NaN values
        batch = [str(text) if pd.notna(text) else "" for text in batch]

        # Remove MedicalLicense detection
        analyzer_engine.registry.remove_recognizer("MedicalLicenseRecognizer")

        # Run analysis in parallel on the batch,
        # while removing DATE_TIME detection
        analyzer_results = [
            analyzer_engine.analyze(
                text=text,
                language="pt",
                score_threshold=0.6,
                allow_list=ALLOW_LIST,
                entities=[
                    ent
                    for ent in analyzer_engine.get_supported_entities()
                    if ent != "DATE_TIME"
                ],
            )
            for text in batch
        ]

        # Run anonymization on the batch
        anonymized_texts = [
            anonymizer_engine.anonymize(
                text=batch[j],
                analyzer_results=analyzer_results[j],
                operators={
                    "DEFAULT": OperatorConfig("custom_slug"),
                    "AS_NUMBER": OperatorConfig("custom_slug"),
                },
            ).text
            for j in range(len(batch))
        ]

        results.extend(anonymized_texts)
    return results


def anonymize_dataframe(
    df: pd.DataFrame,
    analyzer_engine: AnalyzerEngine,
    anonymizer_engine: AnonymizerEngine,
) -> pd.DataFrame:
    # Flatten dataframe to a list of values
    all_values = df.values.flatten().tolist()

    # Process all values in batches
    anonymized_values = batch_process_text(
        all_values, analyzer_engine, anonymizer_engine
    )

    # Reshape back to dataframe structure
    anonymized_array = np.array(anonymized_values).reshape(df.shape)
    return pd.DataFrame(anonymized_array, columns=df.columns, index=df.index)


def read_file(file_path) -> str | pd.DataFrame:
    if len(sys.argv) != 2:
        print("[!] Uso: uv run anon.py <arquivo>")
        sys.exit(1)

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".docx":
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs]
        return "\n".join(paragraphs)
    elif ext == ".csv":
        return pd.read_csv(file_path, dtype=str)
    elif ext == ".xlsx":
        return pd.read_excel(file_path, dtype=str)
    else:
        raise ValueError("Formato não suportado")


def write_file(anonymizer_results: EngineResult | pd.DataFrame, file_path: str) -> None:
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    base_name, ext = os.path.splitext(os.path.basename(file_path))

    if isinstance(anonymizer_results, pd.DataFrame):
        output_file = os.path.join("output", f"anon_{base_name}_{ext[1:]}.csv")
        anonymizer_results.to_csv(output_file, index=False, encoding="utf-8")
    else:
        output_file = os.path.join("output", f"anon_{base_name}_{ext[1:]}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(anonymizer_results.text)

    print(f"Arquivo anonimizado salvo em: {output_file}")


def write_report(file_path: str, start_time: float, data: str | pd.DataFrame) -> None:
    # Ensure output directory exists
    os.makedirs("logs", exist_ok=True)
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    # Create the output file retaining original name and extension
    base_name, ext = os.path.splitext(os.path.basename(file_path))
    report_file = os.path.join(os.getcwd(), "logs", f"report_{base_name}_{ext[1:]}.txt")
    with open(report_file, "w", encoding="utf-8") as report:
        report.write(f"Arquivo processado: {file_path}\n")
        if isinstance(data, pd.DataFrame):
            report.write(f"Número de linhas processadas: {len(data)}\n")
        report.write(f"Tempo total gasto: {elapsed_time:.2f} segundos\n")
    print(f"Relatório salvo em: {report_file}")


def main() -> None:
    # For report-generating purposes
    start_time = time.time()

    file_path = sys.argv[1]
    data = read_file(file_path=file_path)

    trf_model_config, ner_model_config = transformer_model_config()

    analyzer_engine, anonymizer_engine = get_presidio_engines(
        trf_model_config, ner_model_config
    )

    # Analysis and Anonymization
    if isinstance(data, pd.DataFrame):
        anonymizer_results = anonymize_dataframe(
            data, analyzer_engine, anonymizer_engine
        )
    else:
        # Remove MedicalLicense detection
        analyzer_engine.registry.remove_recognizer("MedicalLicenseRecognizer")
        # Remove DATE_TIME detection
        entities = analyzer_engine.get_supported_entities()
        entities_without_date = [ent for ent in entities if ent != "DATE_TIME"]
        analyzer_results = analyzer_engine.analyze(
            text=data,
            language="pt",
            score_threshold=0.6,
            allow_list=ALLOW_LIST,
            entities=entities_without_date,
        )
        anonymizer_results = anonymizer_engine.anonymize(
            text=data,
            analyzer_results=analyzer_results,
            operators={
                "DEFAULT": OperatorConfig("custom_slug"),
                "AS_NUMBER": OperatorConfig("custom_slug"),  # ADICIONADO
            },
        )

    write_file(anonymizer_results, file_path)
    write_report(file_path, start_time, data)


if __name__ == "__main__":
    main()
