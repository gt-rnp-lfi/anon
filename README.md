# anon
(Protótipo da) Ferramenta de Anonimização dos Tickets 

## Pré-Requisitos

Ferramenta `uv`. Para instalar:

* Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
* Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Uso

1. Clonar este repositório: `git clone git@github.com:gt-rnp-lfi/anon.git`
2. Entrar no diretório: `cd ./anon`
3. Instalar o python3.11: `uv python install 3.11`
4. Rodar o script principal: `uv run anon.py -h`

## Exemplos

```bash
λ uv run anon.py data/example.csv
λ uv run anon.py data/example.xlsx
λ uv run anon.py data/example.docx
```

