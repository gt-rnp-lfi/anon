#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys


def blue(text):
    """Formata o texto em azul com ANSI escape sequences."""
    return f"\033[34m{text}\033[0m"


class FileProcessor:
    """Classe responsável pelo processamento de diferentes tipos de arquivos."""

    def __init__(self):
        # Verifica se módulos opcionais estão disponíveis
        try:
            import docx

            self.docx_available = True
        except ImportError:
            self.docx_available = False

        try:
            import openpyxl

            self.openpyxl_available = True
        except ImportError:
            self.openpyxl_available = False

        # Verifica e configura a chave cryptopant
        self.key_path = os.path.join(os.getcwd(), "key.cryptopant")
        self.check_cryptopant_key()

        # Cria o diretório de saída se não existir
        self.output_dir = os.path.join(os.getcwd(), "output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(blue(f"Diretório de saída criado: {self.output_dir}"))
        else:
            print(blue(f"Usando diretório de saída existente: {self.output_dir}"))

    def check_cryptopant_key(self):
        """Verifica se a chave cryptopant existe e a cria se necessário usando o binário scramble_ips."""
        if not os.path.exists(self.key_path):
            print(blue("Chave cryptopant não encontrada. Criando nova chave..."))
            try:
                # Cria uma chave cryptopant usando o binário scramble_ips
                cmd = ["scramble_ips", "--newkey", self.key_path]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    print(blue(f"Erro ao criar chave cryptopant: {result.stderr}"))
                    sys.exit(1)

                print(blue(f"Chave cryptopant criada em: {self.key_path}"))
                # Permissões: apenas o usuário atual pode ler a chave
                os.chmod(self.key_path, 0o600)
            except Exception as e:
                print(blue(f"Erro ao criar chave cryptopant: {str(e)}"))
                sys.exit(1)
        else:
            print(blue(f"Usando chave cryptopant existente: {self.key_path}"))

    def process_file(self, file_path):
        """Processa um arquivo baseado em sua extensão."""
        file_path = os.path.abspath(file_path)
        if not os.path.isfile(file_path):
            print(blue(f"Erro: Arquivo '{file_path}' não encontrado."))
            sys.exit(1)

        ext = os.path.splitext(file_path)[1].lower()

        if ext in [".txt", ".docx"]:
            self.process_textual(file_path)
        elif ext in [".csv", ".xlsx"]:
            self.process_tabular(file_path)
        else:
            print(blue(f"Erro: Formato '{ext}' não suportado."))
            sys.exit(1)

    def process_textual(self, file_path):
        """Processa arquivos de texto (.txt, .docx) e anonimiza IPs usando cryptopant."""
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)
        filename_no_ext = os.path.splitext(filename)[0]

        # Extrair o conteúdo do arquivo
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as file_obj:
                content = file_obj.read()
        elif ext == ".docx":
            if not self.docx_available:
                print(
                    blue(
                        "Erro: 'python-docx' não instalado. Execute 'uv pip install python-docx'."
                    )
                )
                return
            from docx import Document

            doc = Document(file_path)
            content = "\n".join([p.text for p in doc.paragraphs])
        else:
            print(blue(f"Erro: Formato '{ext}' não suportado para arquivos textuais."))
            return

        print(blue(f"\nModo textual: Processando {filename}\n"))

        # Anonimização dos IPs usando cryptopant
        anonymized_content = self.anonymize_ips(content)

        # Mostrar o resultado da anonimização
        print(blue("Conteúdo após anonimização de IPs:"))
        print(anonymized_content)  # Conteúdo do arquivo não é colorido

        # Salvar o conteúdo anonimizado em txt
        output_path = os.path.join(self.output_dir, f"{filename_no_ext}-anon.txt")
        with open(output_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(anonymized_content)

        print(blue(f"Arquivo original mantido em: {file_path}"))
        print(blue(f"Arquivo anonimizado salvo em: {output_path}"))

    def anonymize_ips(self, content):
        """Anonimiza IPs no conteúdo usando o binário scramble_ips no modo texto via stdin."""
        try:
            # Executa o scramble_ips com a chave como argumento posicional e modo texto (-t)
            cmd = ["scramble_ips", "-t", self.key_path]

            # Usa Popen para ter controle sobre stdin/stdout
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Envia o conteúdo via stdin e captura stdout/stderr
            stdout, stderr = process.communicate(input=content)

            if process.returncode != 0:
                print(blue(f"Erro ao executar scramble_ips: {stderr}"))
                return content  # Retorna o conteúdo original em caso de erro

            return stdout

        except Exception as e:
            print(blue(f"Erro durante a anonimização de IPs: {str(e)}"))
            return content  # Retorna o conteúdo original em caso de erro

    def process_tabular(self, file_path):
        """Processa arquivos tabulares (.csv, .xlsx)."""
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        if ext == ".csv":
            import csv

            with open(file_path, "r", encoding="utf-8") as file_obj:
                reader = csv.reader(file_obj)
                header = next(reader, None)
                first_row = next(reader, None)
        elif ext == ".xlsx":
            if not self.openpyxl_available:
                print(
                    blue(
                        "Erro: 'openpyxl' não instalado. Execute 'uv pip install openpyxl'."
                    )
                )
                return
            from openpyxl import load_workbook

            wb = load_workbook(filename=file_path, read_only=True)
            sheet = wb.active
            rows = list(sheet.iter_rows(values_only=True))
            header = rows[0] if rows else None
            first_row = rows[1] if len(rows) > 1 else None
        else:
            print(blue(f"Erro: Formato '{ext}' não suportado para arquivos tabulares."))
            return

        print(blue(f"\nModo tabular: Processando {filename}\n"))
        print(blue("Cabeçalho:"), header)  # Header não é colorido
        print(blue("Primeira linha:"), first_row)  # Primeira linha não é colorida
        print(blue(f"Arquivo original mantido em: {file_path}"))


def main():
    """Função principal do programa."""
    parser = argparse.ArgumentParser(
        description="Anon: Ferramenta de anonimização de tickets de segurança."
    )

    # Permite usar tanto posicional quanto -i/--input
    parser.add_argument(
        "input_file",
        nargs="?",  # Torna opcional para permitir uso de -i
        type=str,
        help="Caminho do arquivo de entrada",
    )

    parser.add_argument(
        "-i",
        "--input",
        dest="input_arg",
        type=str,
        help="Caminho do arquivo de entrada (alternativa)",
    )

    args = parser.parse_args()

    # Prioriza o argumento posicional, depois o -i/--input
    file_path = args.input_file if args.input_file else args.input_arg

    if not file_path:
        parser.error(
            "É necessário fornecer um arquivo de entrada (posicional ou com -i/--input)"
        )

    processor = FileProcessor()
    processor.process_file(file_path)


if __name__ == "__main__":
    main()
