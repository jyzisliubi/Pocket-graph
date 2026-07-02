# Security Policy

## Supported Versions

PocketGraphRAG is currently in alpha. Security fixes are applied to the latest `main` branch only.

| Version | Supported          |
|---------|--------------------|
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, report them privately:

1. Go to the **Security** tab of the repository → **Report a vulnerability**, **or**
2. Email the maintainers directly (see the repository owner profile).

Please include:
- A clear description of the issue and its impact.
- Steps to reproduce (proof of concept, if possible).
- Affected version / commit.

You should receive an initial response within 72 hours. We will coordinate a fix and disclosure timeline with you.

## Scope

The following are considered security issues:
- Remote code execution via the REST API server (`api_server.py`) or Web UI (`webapp.py`).
- Path traversal through file upload / data import.
- Injection through untrusted triples or web-scraped content.
- Secret leakage (API keys) in logs or responses.

The following are **not** security issues — please open a normal issue instead:
- Self-hosted misuse (e.g. exposing `0.0.0.0` without auth).
- Hallucinations or incorrect answers from the LLM.
- Performance / availability issues.
