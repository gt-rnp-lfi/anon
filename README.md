# Anonimização de Incidentes de Segurança com Reidentificação Controlada

Ferramenta prática e inteligente para anonimizar tickets de incidentes de segurança, para ser usada localmente por CSIRTs. 


---

_*Título*: Anonimização de Incidentes de Segurança com Reidentificação Controlada_

_*Resumo*: Este trabalho aborda métodos de anonimização de dados presentes em incidentes de segurança, com o objetivo de alimentá-los em Large Language Models (LLMs). O objetivo é manter informações sensíveis não identificáveis, e, ao mesmo tempo, potencializar o uso de inteligência artificial (IA), permitindo a classificação e correlação pelo modelo de eventos, pessoas e ocorrências. São estabelecidos requisitos de anonimização para a utilização de incidentes reais em uma LLM, a bibliografia é revisitada a fim de avaliar os métodos e ferramentas existentes para o caso proposto, e finalmente é apresentada uma ferramenta que usa uma abordagem híbrida para solucionar o problema especificado._

---

## Estrutura deste README

Esta documentação está organizada da seguinte maneira:

- **Estrutura do Repositório:** Arquivos e diretórios presentes no projeto, com suas funções.
- **Selos Considerados:** Selos pretendidos pelo artefato (Disponível, Funcional e Sustentável).
- **Informações básicas:** Pré-requisitos de software/hardware para execução da ferramenta.
- **Dependências:** Dependências de pacotes necessários para execução.
- **Preocupações com segurança:** Informe sobre possíveis preocupações de segurança. 
- **Instalação:** Passo a passo para baixar, instalar e executar a ferramenta.
- **Teste mínimo:** Exemplos de uso e vídeo demonstrativo da execução da ferramenta.
- **Ambiente de Testes:** Ambiente de hardware/software usado para desenvolvimento/testes.
- **Experimentos:** Sobre coleta de métricas e exemplos de execução dos scripts auxiliares.
- **LICENSE:** Informação sobre a licença do projeto.

---

## Estrutura do Repositório

```
.
├── dataset-teste-anonimizado  # Dados de teste já anonimizados
├── .gitignore                 # Arquivos ignorados pelo git
├── .python-version            # Versão do Python utilizada
├── anon.py                    # Script pricipal
├── count_eng.py               # Script utilitário, conta trechos em inglês
├── get_metrics.py             # Script utilitário, coleta métricas de execução
├── get_runs_metrics.py        # Script utilitário, gera métricas ao longo de várias execuções
├── get_ticket_count.py        # Script utilitário, conta o número de tickets em um diretório
├── LICENSE                    # Licença do projeto
├── pyproject.toml             # Arquivo de configuração, usado para definir dependências
├── README.md                  # Este arquivo
├── teste-exemplo-artigo.txt   # Trecho de incidente usado como exemplo no artigo
└── uv.lock                    # Cria o ambiente a partir do `pyproject.toml` 
```

Obs.: Após a primeira execução, são gerados 4 diretórios:

```
.
├── db       # Base de dados local, contendo as entidades detectadas
├── logs     # Relatórios com estatísticas básicas de execuções
├── models   # Modelos de Redes Neurais utilizadas
└── output   # Saída do script principal, arquivos anonimizados
```

---

## Selos Considerados

Os autores consideram os Selos Disponível, Funcional e Sustentável.

As requisições são baseadas nas informações providas neste repositório, contendo a documentação, título e resumo do trabalho - tornando o artefato disponível. Ademais, esta documentação busca explicitar ao máximo os passos necessários  para a execução do programa, além de todas dependências necessárias e com exemplos de uso, tornando o artefato funcional. Pelo cuidado tomado com documentação, legibilidade e modularidade do código, também é considerado sustentável.

---

## Informações básicas

Componentes necessários para a execução da ferramenta:

1. Ferramenta `uv` na versão `0.4.30`:

- **Windows:**

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.4.30/install.ps1 | iex"
  ```

- **Linux:**

  ```bash
  curl -LsSf https://astral.sh/uv/0.4.30/install.sh | sh
  ```

Requisitos de hardware:

2. No mínimo 5GB de espaço de armazenamento
    - Necessário devido ao tamanho das bibliotecas e modelos de redes neurais.

---

## Dependências

Dependências para a execução da ferramenta:

- Ferramentas Principais:

| Ferramenta | Versão  |
|------------|---------|
| `uv`       | 0.4.30  |
| Python     | 3.11    |

- Dependências do Projeto:

| Pacote                             | Versão       |
|------------------------------------|--------------|
| `email-validator`                  | >=2.2.0      |
| `fasttext`                         | >=0.9.3      |
| `openpyxl`                         | >=3.1.5      |
| `pandas`                           | >=2.2.3      |
| `pip`                              | >=25.0.1     |
| `presidio-analyzer[transformers]`  | >=2.2.357    |
| `presidio-anonymizer`              | >=2.2.357    |
| `protobuf`                         | >=6.30.1     |
| `python-docx`                      | >=1.1.2      |
| `sentencepiece`                    | >=0.2.0      |
| `transformers`                     | >=4.49.0     |

---

## Preocupações com segurança

Não são advertidas preocupações com segurança.

---

## Instalação

1. **Baixe (ou clone) o repositório:** https://github.com/gt-rnp-lfi/anon
   ```bash
   git clone git@github.com:gt-rnp-lfi/anon.git
   ```

2. **Entre no diretório do projeto:**

   ```bash
   cd ./anon
   ```

3. **Execute o script principal, passando um arquivo como argumento:**

    ```bash
    uv run anon.py <arquivo>
    ```

Obs.: A ferramenta aceita os seguintes formatos de arquivo:

- **Formato Texto:** `.txt`
- **Documento Word:** `.docx`
- **Formato IODEF**: `.xml`
- **Planilhas:** `.csv`, `.xlsx`

---

## Teste mínimo

**Exemplos de execução:**

```bash
uv run anon.py caminho/para/seu/arquivo.csv
uv run anon.py caminho/para/um/excel/dados.xlsx
```

> ⏳ **Nota: Primeira execução:** Na primeira execução, os modelos necessários (spaCy e Transformer) serão baixados automaticamente. O processo pode levar alguns minutos, devido aos seus tamanhos.

**Vídeo com exemplos de execução:**

> :warning: O player (no site externo) permite que trechos sejam copiados diretamente do vídeo.

<a href="https://asciinema.org/a/TC8KBxoPO5afHPqjIsSefNbCN" target="_blank"><img src="https://asciinema.org/a/TC8KBxoPO5afHPqjIsSefNbCN.svg" /></a>

---

## Ambiente de Testes

Visando garantir uma maior compatibilidade e resultados o mais próximos possíveis com o reportado, aconselha-se o uso de hardware em nível similar ao da máquina de desenvolvuimento da ferramenta:

### Ambiente de Teste 1:

**Hardware:** Desktop: Processador: AMD Ryzen 3 3300X; Memória RAM: 16GB DDR4 @ 2666Hz; Placa de Vídeo: NVIDIA GeForce 1650.

**Software:** Sistema Operacional: Windows 10 22H2.

**Software:** Sistema de Virtualização: WSL2, Ubuntu 20.04.

---

## Experimentos

### Coleta de métricas para 10 execuções

Para coletar métricas de performance ao longo de 10 *runs*, é possível usar de [um dos scripts auxiliares](./get_runs_metrics.py). Basta passar um diretório como único argumento de liha de comando:

```bash
uv run get_runs_metrics.py <diretório contendo um conjunto teste>
```

### Reinvindicação 1:

```bash
uv run anon.py dataset-teste-anonimizado/anon_incidents_xlsx.csv
```

### Reinvindicação 2:

```bash
uv run anon.py dataset-teste-anonimizado/anon_POP_-__-_RS_-__-_CERT-RS_\(Todo180_xlsx.csv
```

---

## LICENSE

Esta ferramenta está licenciada sob a [GPL-3.0](./LICENSE).
