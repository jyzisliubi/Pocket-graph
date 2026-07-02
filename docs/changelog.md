# Changelog

See the root [`CHANGELOG.md`](https://github.com/JayZ/PocketGraphRAG/blob/main/CHANGELOG.md)
for the full version history.

## Current

### Added
- Async LLM entry points `acall_llm` / `acall_llm_stream`.
- Evaluation harness (`eval_harness.py`) with retrieval + generation metrics and optional RAGAS.
- Typer CLI with `init` / `build` / `extract` / `qa` / `serve` subcommands.
- MkDocs Material documentation site.
- Project governance files: LICENSE, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT,
  issue/PR templates, dependabot.
- CI now uploads coverage to Codecov; PyPI publish uses Trusted Publishing (OIDC).
