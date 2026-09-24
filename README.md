# logit-gate-rag

[English](README.md) | [日本語](README.ja.md)

High-precision, ultra-low-latency RAG filtering pipeline combining **Ruri** semantic search and **Qwen2.5** pointwise logit classification.


## Overview

Traditional generative LLM rerankers introduce substantial latency due to autoregressive decoding loops. `logit-gate-rag` solves this by:
1. Extracting candidate passages using lightweight dense retrieval (**Ruri-v3** + pure SentencePiece).
2. Pointwise relevance filtering by directly evaluating unnormalized logits ("Yes" / "No") at the LM-Head without generating tokens.


## Architecture Diagram

```mermaid
flowchart TD
    UserQuery[User Query] --> Embed[Dense Retrieval<br>Ruri-v3-30m]
    Embed --> Retrieve[Retrieve Top-K Documents]
    Retrieve --> Route[Pointwise Logit Router<br>Qwen2.5]
    Route -- "Relevance > Threshold" --> Accept[Accept: Pass to Generation LLM]
    Route -- "Relevance <= Threshold" --> Reject[Reject: Fallback (Fast Reject)]
    
    subgraph Fast Reject Pipeline
        Reject --> ReturnMessage[Return Predefined Message<br>~52ms Latency]
    end
```


## Key Features
- **Ruri-v3-30m Dense Retrieval**: Extremely fast semantic search using a lightweight embedding model (37M parameters, 256-dim).
- **Qwen2.5 Logit Router**: Evaluates "Yes" / "No" logits without autoregressive token generation, dramatically cutting latency compared to traditional rerankers.
- **Cost & Latency Savings**: Effectively drops irrelevant or near-miss queries before calling the heavy generation LLM, saving both compute cost and time.


## Benchmark Highlights
Based on real hardware testing (RTX 3060, N=54 across 4 domains):
- **High Accuracy**: 96.3% accuracy in routing decisions (52/54 correct).
- **Strong Safety**: 94.4% near-miss rejection rate (17/18 hard negative queries correctly blocked).
- **Ultra-low Latency**: ~52ms average decision latency per query.
- **Compute Savings**: Prevented 17 unnecessary generation LLM calls, saving ~72 seconds of execution time in a single run.


## Quickstart Guide

### Installation
This project uses `uv` for fast dependency management.

```bash
# Clone the repository
git clone https://github.com/chottokun/logit-gate-rag.git
cd logit-gate-rag

# Install dependencies using uv
uv sync
```

### Running the Demo
You can test the filtering pipeline locally using the provided demo script.

```bash
# Set PYTHONPATH and run the demo via uv
PYTHONPATH=. uv run demo.py
```

### Running Benchmarks
To run the N=54 sufficiency evaluation benchmark suite:

```bash
# Run the benchmark suite
PYTHONPATH=. uv run benchmarks/run_benchmark.py
```

## Architecture & Knowledge Base

Project specifications, architecture decisions, and benchmark reports are documented in [docs/](docs/):
- [Architecture & Benchmarks](docs/architecture/README.md)
  - [N=54 Multi-Domain Benchmark Report](docs/architecture/benchmark_report_n54.en.md)
  - [RTX 3060 Real-Device Verification Report](docs/architecture/benchmark_report_rtx3060.en.md)
- [Domain & Model Specifications](docs/domain/README.md)
  - [Ruri-v3-30m Lightweight Dense Retrieval](docs/domain/ruri_v3_30m.en.md)
- [Specifications & Planning](plan/search_logit_sem.md)

## Development Workflow

- **Cloud (Google Jules)**: Autonomous implementation, benchmarking expansion, and Pull Request authoring.
- **Local (RTX 3060 GPU)**: High-speed on-device verification (CUDA, PyTorch, SentencePiece, logit extraction).

