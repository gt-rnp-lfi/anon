#! /usr/bin/env python

import datetime
import hashlib
import ipaddress
import os
import sqlite3
import sys
import time
import warnings

import spacy
import torch
from docx import Document
from huggingface_hub import snapshot_download
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NerModelConfiguration, TransformersNlpEngine
from presidio_anonymizer import AnonymizerEngine, OperatorConfig
from presidio_anonymizer.operators import Operator, OperatorType
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Globals
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "Davlan/xlm-roberta-large-ner-hrl"
TRF_MODEL_PATH = os.path.join("models", MODEL)

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


def save_entity(
    entity_type: str, original_name: str, slug_name: str, full_hash: str
) -> None:
    db_dir = os.path.join(os.getcwd(), "db")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "entities.db")
    now = datetime.datetime.now().isoformat()
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
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
        cur = conn.execute(
            "SELECT id, first_seen FROM entities WHERE full_hash=?", (full_hash,)
        )
        row = cur.fetchone()
        if row:
            # Atualiza o timestamp last_seen
            conn.execute("UPDATE entities SET last_seen=? WHERE id=?", (now, row[0]))
        else:
            # Insere nova entidade
            conn.execute(
                "INSERT OR IGNORE INTO entities (entity_type, original_name, slug_name, full_hash, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?)",
                (entity_type, original_name, slug_name, full_hash, now, now),
            )
        conn.commit()


class CustomSlugAnonymizer(Operator):
    # Strip before hashing to guarantee uniqueness
    def operate(self, text: str, params: dict = None) -> str:
        clean_text = " ".join(text.split()).strip()
        full_hash = hashlib.sha256(clean_text.encode()).hexdigest()
        slug = full_hash[:10]
        entity_type = params.get("entity_type", "UNKNOWN")
        save_entity(entity_type, clean_text, slug, full_hash)
        return f"[{entity_type}_{slug}]"

    def validate(self, params: dict = None) -> None:
        pass

    def operator_name(self) -> str:
        return "custom_slug"

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize


def models_check() -> None:
    # Spacy
    spacy_model = "pt_core_news_lg"
    spacy_model_path = os.path.join(os.getcwd(), "models", spacy_model)
    try:
        nlp = spacy.load(spacy_model_path)
    except OSError:
        print(f"Downloading spaCy's `{spacy_model}`...")
        spacy.cli.download(spacy_model_path)
        nlp = spacy.load(spacy_model_path)
        nlp.to_disk(spacy_model_path)

    # Transformer
    if not os.path.exists(TRF_MODEL_PATH):
        print(f"Downloading transformer `{MODEL}`...")
        snapshot_download(
            repo_id=MODEL,
            local_dir=TRF_MODEL_PATH,
            repo_type="model",
            max_workers=10,
        )


def transformer_model_config():
    # Intantiate tokenizer and (transformer) model
    AutoTokenizer.from_pretrained(MODEL, cache_dir=TRF_MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, cache_dir=TRF_MODEL_PATH
    )
    # Send model to GPU, if possible
    model.to(DEVICE)

    # Transformer model config
    trf_model_config = [
        {
            "lang_code": "pt",
            "model_name": {
                "spacy": "pt_core_news_lg",
                "transformers": MODEL,  # Used only for NER
            },
        }
    ]

    # Transformer NER config
    ner_model_configuration = NerModelConfiguration(
        model_to_presidio_entity_mapping=ENTITY_MAPPING,
        alignment_mode="expand",  # "strict", "contract", "expand"
        aggregation_strategy="simple",  # "simple", "first", "average", "max"
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

    # Anonymizer Engine config
    anonymizer_engine = AnonymizerEngine()
    anonymizer_engine.add_anonymizer(CustomSlugAnonymizer)

    return analyzer_engine, anonymizer_engine


def main() -> None:
    start_time = time.time()

    file_path = sys.argv[1]
    text = read_file(file_path=file_path)

    models_check()

    trf_model_config, ner_model_config = transformer_model_config()
    analyzer_engine, anonymizer_engine = get_presidio_engines(
        trf_model_config, ner_model_config
    )

    # Analysis
    analyzer_results = analyzer_engine.analyze(
        text=text, language="pt", allow_list=["TCP", "UDP"]
    )
    # Anonymization
    anonymizer_results = anonymizer_engine.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators={"DEFAULT": OperatorConfig("custom_slug")},
    )

    write_file(anonymizer_results, file_path)

    elapsed_time = time.time() - start_time
    print(f"Tempo total gasto: {elapsed_time:.2f} segundos")


def read_file(file_path) -> str:
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
    else:
        raise ValueError("Formato não suportado")


def write_file(anonymizer_results, file_path):
    os.makedirs("output", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join("output", f"anon_{base_name}.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(anonymizer_results.text)

    print(f"Texto anonimizado salvo em: {output_file}")


if __name__ == "__main__":
    main()
