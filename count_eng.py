#! /usr/bin/env python3

"""
Conta o número de trechos em inglês de arquivos CSV num caminho, passado como argumento.
"""

import re
import sys
from pathlib import Path

import pandas as pd

# Verificar se o usuário forneceu um caminho de pasta como argumento
if len(sys.argv) > 1:
    folder = Path(sys.argv[1])
    if not folder.exists() or not folder.is_dir():
        print(f"Erro: '{folder}' não é um diretório válido.")
        sys.exit(1)
else:
    print("[!] Uso: uv run count_eng.py <caminho_da_pasta>")
    sys.exit(1)

files = sorted(folder.glob("*.csv"))
if not files:
    print(f"Nenhum arquivo CSV encontrado em '{folder}'")
    sys.exit(1)

# Palavras comuns
english_pattern = re.compile(
    r"\b(the|and|for|from|attack|report|information|address|contact|system|service)\b",
    re.IGNORECASE,
)

# Pra armazenar os resultados
results = []

# Escanear todos os arquivos CSV
for file in files:
    df = pd.read_csv(file, encoding="utf-8", delimiter=",", on_bad_lines="skip")
    count = 0
    for col in df.columns:
        # Converter para string e tratar valores nulos
        series = df[col].astype(str).replace("nan", "")
        # Contar células que batem com o regex
        count += sum(bool(english_pattern.search(text)) for text in series)
    # Agregar num dict
    results.append({"Arquivo": file.name, "Trechos": count})

# Construir e printar o DataFrame dos resultados
summary_df = (
    pd.DataFrame(results)
    .sort_values(by="Trechos", ascending=True)
    .reset_index(drop=True)
)
print(summary_df)
print("\nTotal Geral de trechos em inglês:", summary_df["Trechos"].sum())
