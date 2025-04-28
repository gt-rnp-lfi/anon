#!/usr/bin/env python3

"""
Conta o número de tickets em um caminho, passado como argumento.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

# Warning irritante sobre estilo padrão
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# Caminho passado como argumento?
if len(sys.argv) < 2:
    print("Uso: uv run count_tickets.py <caminho_da_pasta>")
    sys.exit(1)

# Caminho a ser verificado existe?
base_path = Path(sys.argv[1])
if not base_path.exists() or not base_path.is_dir():
    print(f"Erro: '{base_path}' não é uma pasta válida.")
    sys.exit(1)

# Tipos de arquivos a serem considerados
extensoes = [".csv", ".xlsx", ".docx", ".txt"]

contagem = {}
soma_total = 0

for arquivo in base_path.iterdir():
    if arquivo.suffix.lower() not in extensoes:
        continue

    try:
        if arquivo.suffix == ".csv":
            df = pd.read_csv(arquivo)
            linhas = len(df)
        elif arquivo.suffix == ".xlsx":
            df = pd.read_excel(arquivo)
            linhas = len(df)
        elif arquivo.suffix == ".docx":
            linhas = 1  # cada docx é 1 ticket
        elif arquivo.suffix == ".txt":
            linhas = 1  # output anonimizado de docx é txt

        linhas = max(0, linhas)
        contagem[arquivo.name] = linhas
        soma_total += linhas

    except Exception as e:
        contagem[arquivo.name] = f"Erro: {e}"

largura_nome = max(len(nome) for nome in contagem)
for nome, linhas in contagem.items():
    print(f"{nome.ljust(largura_nome)} : {str(linhas).rjust(5)}")

print(f"\n{'Total de tickets'.ljust(largura_nome)}: {str(soma_total).rjust(5)}")
