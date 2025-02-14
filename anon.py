# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import argparse
import os
import sys


def process_textual(file_path):
    """Processa arquivos de texto (.txt, .docx)."""

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as file_obj:
            content = file_obj.read()
    elif ext == ".docx":
        try:
            from docx import Document

            doc = Document(file_path)
            content = "\n".join([p.text for p in doc.paragraphs])
        except ImportError:
            print(
                "Erro: `python-docx` não instalado. Tente `uv pip install python-docx`."
            )
            return
    else:
        print(f"Erro: Formato `{ext}` não suportado para arquivos textuais.")
        return

    print("\n\nModo textual: Arquivo de texto\n\n")
    print(content)


def process_tabular(file_path):
    """Processa arquivos tabulares (.csv, .xlsx)."""
    import csv

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        with open(file_path, "r", encoding="utf-8") as file_obj:
            reader = csv.reader(file_obj)
            print("\n\nModo tabular: Arquivo CSV\n\n")
            header = next(reader, None)
            first_row = next(reader, None)
    elif ext == ".xlsx":
        try:
            from openpyxl import load_workbook

            wb = load_workbook(filename=file_path, read_only=True)
            sheet = wb.active
            rows = list(sheet.iter_rows(values_only=True))
            header, first_row = rows[0], rows[1] if len(rows) > 1 else None
        except ImportError:
            print("Erro: `openpyxl` não instalado. Tente `uv pip install openpyxl`.")
            return
    else:
        print(f"Erro: Formato `{ext}` não suportado para arquivos tabulares.")
        return

    print(header)
    print(first_row)


def process_file(file_path):
    """Redireciona para a função apropriada com base na extensão do arquivo."""
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        print(f"Erro: Arquivo `{file_path}` não encontrado.")
        sys.exit(1)

    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".txt", ".docx"]:
        process_textual(file_path)
    elif ext in [".csv", ".xlsx"]:
        process_tabular(file_path)
    else:
        print(f"Erro: Formato `{ext}` não suportado.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Anon: Ferramenta de anonimização de tickets de segurança."
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Caminho do arquivo de entrada",
    )

    args = parser.parse_args()

    process_file(args.input_file)


if __name__ == "__main__":
    main()
