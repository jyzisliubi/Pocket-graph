# Contributing to PocketGraphRAG

Thanks for your interest in contributing! This guide covers the common contribution workflows.

## Code of Conduct

By participating you agree to uphold our [Code of Conduct](CODE_OF_CONDUCT.md). Please be kind and respectful.

## Ways to Contribute

- **Report bugs**: open an issue using the Bug Report template.
- **Suggest features**: open an issue using the Feature Request template.
- **Improve docs**: README, docstrings, or the MkDocs site under `docs/`.
- **Submit code**: fix a bug or implement a feature via Pull Request.

## Development Setup

```bash
# Clone
git clone https://github.com/JayZ/PocketGraphRAG.git
cd PocketGraphRAG

# Install in editable mode with dev extras (pulls all runtime + dev deps)
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Optional: all extras
pip install -e ".[all]"
```

## Workflow

1. Fork the repo and create a branch: `git checkout -b feat/my-feature`.
2. Make your changes. Keep diffs focused — one logical change per PR.
3. Run checks locally:

```bash
ruff check .
ruff format .
pytest PocketGraphRAG/tests/ -v
```

4. Commit using [Conventional Commits](https://www.conventionalcommits.org/) style:
   - `feat: add async streaming for DashScope`
   - `fix: dedup triples when confidence ties`
   - `docs: clarify search-mode table`
5. Push and open a Pull Request. Fill in the PR template.

## Code Style

- Formatting & linting: `ruff` (config in `ruff.toml`).
- Type hints are used throughout — keep new code annotated.
- Docstrings: Chinese is fine for user-facing modules; keep them concise.

## Testing

- All new features or bug fixes should include tests under `PocketGraphRAG/tests/`.
- Tests must pass on Python 3.8+ across Ubuntu / Windows / macOS (CI enforces this).
- Mock external LLM / network calls — do not hit real APIs in CI.

## Release Process

1. Maintainer bumps version in `pyproject.toml` and `PocketGraphRAG/__init__.py`.
2. Update `CHANGELOG.md`.
3. Tag `vX.Y.Z` — the `publish.yml` workflow builds and publishes to PyPI via Trusted Publishing.

## Reporting Security Issues

Do **not** open a public issue for security problems. See [SECURITY.md](SECURITY.md) for the private disclosure process.
