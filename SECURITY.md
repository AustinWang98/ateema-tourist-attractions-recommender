# Security Policy

## Reporting a vulnerability

Please do not disclose security issues in a public GitHub issue. Use GitHub's
private vulnerability reporting feature for this repository. Include affected
paths, reproduction steps, impact, and any suggested mitigation.

## Secrets and private data

- Keep `.env` files, API keys, credentials, and service-account files out of Git.
- Use the hosting platform's secret manager for runtime values.
- Keep authorized analytics exports under the ignored `data/private/` path.
- Rotate a credential immediately if it appears in a commit, log, artifact, or
  pull request.

The maintainers do not guarantee security support for outdated dependencies or
unofficial deployments.
