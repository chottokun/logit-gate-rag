# logit-rerank-rag

High-precision, ultra-low-latency RAG filtering pipeline combining **Ruri** semantic search and **Qwen2.5** pointwise logit classification.

## Overview

Traditional generative LLM rerankers introduce substantial latency due to autoregressive decoding loops. `logit-rerank-rag` solves this by:
1. Extracting candidate passages using lightweight dense retrieval (**Ruri-v3** + pure SentencePiece).
2. Pointwise relevance filtering by directly evaluating unnormalized logits ("Yes" / "No") at the LM-Head without generating tokens.

## Architecture & Knowledge Base

Project specifications, architecture decisions, and workflow guides are documented in [docs/](docs/):
- [Architecture Index](docs/architecture/README.md)
- [Jules × agy CLI Hybrid Pipeline](docs/architecture/jules_local_hybrid_pipeline.md)
- [Local LLM Setup & Optimization](docs/infrastructure/local_llm_setup.md)
- [Specifications & Plans](plan/search_logit_sem.md)

## Development Workflow

This project utilizes a hybrid development model:
- **Cloud (Google Jules)**: Autonomous implementation planning and GitHub Pull Request generation.
- **Local (RTX 3060 / agy CLI / Gemma 4)**: On-device CUDA/PyTorch verification and zero-token-cost local patching via Antigravity SDK.
