# AnonLFI: Framework de Anonimização de Incidentes para Cyber Threat Intelligence com LLMs

Ferramenta prática e inteligente para anonimizar tickets de incidentes. 🚀

---

_Resumo: Este trabalho aborda métodos de anonimização de dados presentes em incidentes de segurança, com o objetivo de alimentá-los em Large Language Models (LLMs). O objetivo é manter informações sensíveis não identificáveis, e, ao mesmo tempo, potencializar o uso de inteligência artificial (IA), permitindo a classificação e correlação pelo modelo de eventos, pessoas e ocorrências. São estabelecidos requisitos de anonimização para a utilização de incidentes reais em uma LLM, a bibliografia é revisitada a fim de avaliar os métodos e ferramentas existentes para o caso proposto, e finalmente é apresentada uma ferramenta que usa uma abordagem híbrida para solucionar o problema especificado._

---

## Pré-Requisitos

Ferramenta `uv` na versão `0.4.30`:

- **Windows:**

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.4.30/install.ps1 | iex"
  ```

- **Linux:**

  ```bash
  curl -LsSf https://astral.sh/uv/0.4.30/install.sh | sh
  ```

---

## Versões e Ambiente utilizado

TODO!()

---

## Instalação e Configuração

1. **Clone o repositório:**

   ```bash
   git clone git@github.com:gt-rnp-lfi/anon.git
   ```

2. **Entre no diretório do projeto:**

   ```bash
   cd ./anon
   ```

3. **Instale o Python 3.11:**

   ```bash
   uv python install 3.11
   ```

> ⚠️ **Nota: Primeira execução:** Na primeira execução, os modelos necessários (Spacy e Transformer) serão baixados automaticamente. O processo pode levar um momento, devido ao seu tamanho ⏳

---

## Uso

Execute o script passando um arquivo como argumento:

```bash
uv run anon.py <arquivo>
```

**Exemplo:**

```bash
uv run anon.py caminho/para/seu/arquivo.csv
uv run anon.py caminho/para/um/excel/dados.xlsx
```

---

## Formatos de Arquivos Suportados

A ferramenta aceita os seguintes formatos de arquivo:

- **Planilhas:** `.csv`, `.xlsx`
- **Documento Word:** `.docx`
- **Texto:** `.txt`

---

## Observações Adicionais

- **Logs e Relatórios:**  
  A saída (os arquivos anonimizados) serão salvos na pasta `output` - criada pela própria ferramenta -, e um relatório de execução será gerado na pasta `logs`, também gerada.

---
