# AI Agent Operational Rules & Guidelines

## 1. Public Repository Security & Privacy Guardrails (CRITICAL)
- This repository is **PUBLIC** (`chottokun/logit-gate-rag`).
- **NEVER** commit API keys, tokens, credentials, internal IP addresses (`10.x.x.x`, `192.168.x.x`), or specific local user path names (e.g. `/home/username/`).
- Always use environment variables (`os.getenv(...)`) and relative paths.
- Ensure temporary files, `.env*`, and local artifacts are ignored in `.gitignore`.

## 2. Architecture & Role Division
- **Cloud (Google Jules)**:
  - Global codebase indexing, multi-component planning, and generating clean Pull Requests (PRs).
- **Local (RTX 3060 / agy CLI / Gemma 4)**:
  - Local GPU verification (CUDA, PyTorch, SentencePiece, Logit extraction).
  - Fast, zero-token-cost local patching for test failures or device-specific adjustments.

## 3. Documentation & Standards
- Documentation follows the Open Knowledge Format (OKF v0.2) under `docs/`.
- Maintain concise, clean modular Python code.
