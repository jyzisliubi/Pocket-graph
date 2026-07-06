# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in PocketGraphRAG, please report it responsibly:

1. **DO NOT** open a public GitHub issue
2. Email: 3364415961@qq.com with subject `[SECURITY] PocketGraphRAG`
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: within 48 hours
- **Initial Assessment**: within 7 days
- **Fix Release**: within 30 days for critical, 90 days for moderate

## Security Best Practices

### API Key Protection
- Use `POCKET_API_KEYS` (comma-separated) for multi-key rotation
- Set `POCKET_API_AUTH_ENABLED=1` to enforce authentication
- Never commit `.env` files to git

### Dependency Security
- We monitor for CVEs in dependencies (faiss, torch, transformers, fastapi)
- `litellm` users should pin to safe versions (see [CVE-2025-1337](https://github.com/BerriAI/litellm/security/advisories))
- Run `pip audit` periodically to check for known vulnerabilities

### Docker Security
- Container runs as non-root by default in production
- Use `--read-only` filesystem for the container where possible
- Scan images with `trivy` or `grype` before deployment

### Data Privacy
- PocketGraphRAG is local-first: your data never leaves your machine by default
- When using cloud LLM providers (DeepSeek/SiliconFlow/DashScope), only query text is sent
- Document content is processed locally; only extracted triples' text is sent to LLM for refinement
