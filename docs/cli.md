# CLI

PocketGraphRAG provides two CLIs:

1. **Modern Typer CLI** (`pocketgraphrag-cli`) — atomic subcommands. Requires the `[cli]` extra.
2. **Legacy interactive CLI** (`python -m PocketGraphRAG.app`) — argparse REPL. Always available.

## Modern CLI

Install:

```bash
pip install -e ".[cli]"
```

### `init` — environment check

```bash
pocketgraphrag-cli init
```

Prints data path, index dir, embedding model, fusion strategy, and which LLM
providers are configured.

### `build` — build indexes

```bash
pocketgraphrag-cli build
pocketgraphrag-cli build --data my_triples.txt
```

Builds FAISS + entity + relation embedding indexes.

### `extract` — KG extraction

```bash
pocketgraphrag-cli extract -i document.txt -o triples.txt --min-confidence 0.6
```

Extracts triples from a text/markdown file using the LLM.

### `qa` — one-shot question

```bash
pocketgraphrag-cli qa "盗梦空间讲了什么？" --search-mode mix --top-k 5 --stream
```

### `serve` — launch a server

```bash
pocketgraphrag-cli serve web                       # Gradio UI on :7860
pocketgraphrag-cli serve api --port 8000           # FastAPI REST API
```

## Legacy Interactive CLI

```bash
python -m PocketGraphRAG.app --search-mode mix --multihop
```

An interactive REPL: type a question, `c` to clear history, `q` to quit.
