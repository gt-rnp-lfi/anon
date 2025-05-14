# Anonimização de Incidentes de Segurança com Reidentificação Controlada

Ferramenta prática e inteligente para anonimizar tickets de incidentes de segurança, para ser usada localmente por CSIRTs. 

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

Ferramentas Principais:

| Ferramenta | Versão  |
|------------|---------|
| `uv`       | 0.4.30  |
| Python     | 3.11    |

Dependências do Projeto:

| Dependência                        | Versão       |
|------------------------------------|--------------|
| `email-validator`                  | >=2.2.0      |
| `openpyxl`                         | >=3.1.5      |
| `pandas`                           | >=2.2.3      |
| `pip`                              | >=25.0.1     |
| `presidio-analyzer[transformers]`  | >=2.2.357    |
| `presidio-anonymizer`              | >=2.2.357    |
| `protobuf`                         | >=6.30.1     |
| `python-docx`                      | >=1.1.2      |
| `sentencepiece`                    | >=0.2.0      |
| `transformers`                     | >=4.49.0     |

O desenvolvimento e testes iniciais foram feitos em uma máquina com Windows 10 22H2 sob o WSL2 com um Ubuntu 20.04. Memória RAM de 16GB, Processador AMD Ryzen 3 3300X com 4 cores e 8 núcleos, e Placa Gráfica NVIDIA GeForce GTX 1650.

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

---

## Uso

Execute o script passando um arquivo como argumento:

```bash
uv run anon.py <arquivo>
```

**Exemplos:**

```bash
uv run anon.py caminho/para/seu/arquivo.csv
uv run anon.py caminho/para/um/excel/dados.xlsx
```

> ⏳ **Nota: Primeira execução:** Na primeira execução, os modelos necessários (Spacy e Transformer) serão baixados automaticamente. O processo pode levar um momento, devido ao seu tamanho.

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

- **Tabela de Entidades:**
  
  Ao anonimizar arquivos, o *script* também cria um arquivo `db/entities.db`, uma base de dados SQLite3 contendo informações sobre as entidades encontradas na fase de análise.

---

## Exemplo de Execução

> ⚠️ O exemplo abaixo já consta com os modelos baixados.

```bash
  (anon) ➜  anon git:(main) uv run anon.py data/examples/example-short.csv
  Device set to use cpu
  Arquivo anonimizado salvo em: output/anon_example-short_csv.csv
  Relatório salvo em: logs/report_example-short_csv.txt
```
