#! /usr/bin/env python

"""
Printa métricas de agregação dos relatórios presentes na pasta `logs/`.
"""

import glob
import os
import re

import numpy as np


def parse_report(file_path):
    """
    Lê um relatório e extrai as métricas:
    - Número de linhas processadas
    - Tempo total gasto (em segundos)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    num_lines = 0
    time_spent = 0.0

    for line in content.splitlines():
        if line.startswith("Número de linhas processadas:"):
            match = re.search(r"\d+", line)
            if match:
                num_lines = int(match.group())
        elif line.startswith("Tempo total gasto:"):
            match = re.search(r"[\d.]+", line)
            if match:
                time_spent = float(match.group())

    return num_lines, time_spent


def aggregate_reports(log_folder):
    """
    Processa todos os relatórios na pasta 'logs/' e calcula estatísticas globais.
    """
    report_files = glob.glob(os.path.join(log_folder, "report_*.txt"))

    data = []

    for file in report_files:
        num_lines, time_spent = parse_report(file)
        data.append((num_lines, time_spent))

    # Caso não haja relatórios
    if not data:
        return None

    linhas, tempos = zip(*data)

    stats = {
        "total_arquivos": len(data),
        "total_linhas": sum(linhas),
        "média_linhas": np.mean(linhas),
        "desvio_linhas": np.std(linhas),
        "média_tempo": np.mean(tempos),
        "desvio_tempo": np.std(tempos),
        "tempo_total": sum(tempos),
        "correlação": np.corrcoef(linhas, tempos)[0, 1] if len(data) > 1 else None,
    }

    return stats


if __name__ == "__main__":
    log_folder = "logs"
    stats = aggregate_reports(log_folder)

    if stats:
        print("=== Estatísticas Gerais ===")
        print(f"Arquivos processados: {stats['total_arquivos']}")
        print(f"Total de linhas processadas: {stats['total_linhas']}")
        print(
            f"Média de linhas por arquivo: {stats['média_linhas']:.2f} ± {stats['desvio_linhas']:.2f}"
        )
        print(f"Tempo total gasto: {stats['tempo_total']:.2f} segundos")
        print(
            f"Média de tempo por arquivo: {stats['média_tempo']:.2f} ± {stats['desvio_tempo']:.2f} segundos"
        )
        if stats["correlação"] is not None:
            print(
                f"Correlação de Pearson entre linhas processadas e tempo gasto: {stats['correlação']:.3f}"
            )
