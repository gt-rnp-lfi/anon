#!/usr/bin/env python3

"""
Coleta métricas de agregação do script principal
ao longo de várias execuções, salvando os resultados em um CSV.
"""

import csv
import glob
import os
import re
import subprocess
import time

# Quantidade de runs
NUM_RUNS = 10
# Diretório onde o anon.py grava os relatórios
REPORT_DIR = "logs"
# CSV de saída
CSV_OUT = "metrics_runs.csv"
# Arquivos de teste
TEST_FILES = glob.glob("data/datasets-teste-base/*")
# Comando base para rodar o anon
CMD_BASE = ["uv", "run", "anon.py"]


def collect_run_metrics(run_id):
    """
    Executa uma run completa (conjunto de teste), aguarda os relatórios,
    extrai tempo e tickets de cada relatório e retorna um dict com:
      - número da run
      - total de tickets
      - tempo total (s)
      - tempo médio por arquivo (s)
      - tempo médio por ticket (s)
    """
    print(f"\n=== Iniciando run {run_id}/{NUM_RUNS} ===")

    per_file_times = []
    total_tickets = 0

    for idx, file in enumerate(TEST_FILES, start=1):
        print(f"[Run {run_id}] Processando arquivo {idx}/{len(TEST_FILES)}: {file}")
        # dispara o anon.py e deixa stdout/stderr no console
        try:
            subprocess.run(CMD_BASE + [file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Run {run_id}] ⚠️ Erro em {file}: {e}")

        # nome do relatório gerado
        base, ext = os.path.splitext(os.path.basename(file))
        report_file = os.path.join(REPORT_DIR, f"report_{base}_{ext[1:]}.txt")

        # espera até o relatório existir (timeout 10 min por arquivo)
        deadline = time.time() + 600
        while not os.path.exists(report_file) and time.time() < deadline:
            time.sleep(0.5)
        if not os.path.exists(report_file):
            print(f"[Run {run_id}] ⚠️ Relatório faltando: {report_file}")
            # fallback: .docx = 1 ticket, 0s; outros = 0 ticket, 0s
            tickets = 1 if ext.lower() == ".docx" else 0
            per_file_times.append(0.0)
            total_tickets += tickets
            continue

        # lê e extrai métricas
        content = open(report_file, encoding="utf-8").read()
        m_lines = re.search(r"Número de linhas processadas:\s*(\d+)", content)
        m_time = re.search(r"Tempo total gasto:\s*([\d.]+)", content)

        tickets = (
            int(m_lines.group(1)) if m_lines else (1 if ext.lower() == ".docx" else 0)
        )
        time_spent = float(m_time.group(1)) if m_time else 0.0

        per_file_times.append(time_spent)
        total_tickets += tickets

        print(f"[Run {run_id}] → tickets: {tickets}, tempo: {time_spent:.2f}s")

    # agrega métricas da run
    total_time = sum(per_file_times)
    avg_file = total_time / len(per_file_times) if per_file_times else 0.0
    avg_ticket = total_time / total_tickets if total_tickets else 0.0

    print(
        f"[Run {run_id}] ✔️ Concluída: total_time={total_time:.2f}s, "
        f"total_tickets={total_tickets}, avg_file={avg_file:.2f}s, "
        f"avg_ticket={avg_ticket:.2f}s"
    )

    return {
        "run": run_id,
        "total_tickets": total_tickets,
        "total_time_s": round(total_time, 2),
        "avg_time_per_file": round(avg_file, 2),
        "avg_time_per_ticket": round(avg_ticket, 2),
    }


def main():
    # Verifica se o CSV já existe
    first_time = not os.path.exists(CSV_OUT)

    # Abre em modo append
    with open(CSV_OUT, "a", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(
            csvf,
            fieldnames=[
                "Rodada",
                "Total de Tickets",
                "Tempo Total (s)",
                "Tempo Médio por Arquivo (s)",
                "Tempo Médio por Ticket (s)",
            ],
        )
        # Se for a primeira vez, escreve o cabeçalho
        if first_time:
            writer.writeheader()

        # Executa as runs sequencialmente
        for run_id in range(1, NUM_RUNS + 1):
            metrics = collect_run_metrics(run_id)
            writer.writerow(
                {
                    "Rodada": metrics["run"],
                    "Total de Tickets": metrics["total_tickets"],
                    "Tempo Total (s)": metrics["total_time_s"],
                    "Tempo Médio por Arquivo (s)": metrics["avg_time_per_file"],
                    "Tempo Médio por Ticket (s)": metrics["avg_time_per_ticket"],
                }
            )
            csvf.flush()
            print(f"[Main] Linha da run {run_id} adicionada ao CSV.")

    print(f"\n✅ Todas as {NUM_RUNS} runs concluídas. Métricas em: {CSV_OUT}")


if __name__ == "__main__":
    main()
