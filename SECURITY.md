# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | Yes       |
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly.

**Do not open a public issue.**

Instead, email **chunkytortoise@proton.me** with:

1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

You will receive an acknowledgment within 48 hours. We aim to provide a resolution or mitigation plan within 7 days.

## Scope

This project processes user-uploaded CSV and Excel files. Security considerations include:

- **File upload validation**: Only CSV and Excel formats accepted
- **Data isolation**: Uploaded data is processed in-memory and not persisted server-side
- **No authentication**: The Streamlit demo is a public read-only tool; no user accounts or PII storage
- **Dependencies**: We monitor dependencies for known vulnerabilities via GitHub Dependabot

## Acknowledgments

We appreciate responsible disclosure and will credit reporters (with permission) in release notes.
