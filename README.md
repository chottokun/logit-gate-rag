# logit-rerank-rag

High-precision, ultra-low-latency RAG filtering pipeline combining **Ruri** semantic search and **Qwen2.5** pointwise logit classification.

## Overview

Traditional generative LLM rerankers introduce substantial latency due to autoregressive decoding loops. `logit-rerank-rag` solves this by:
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
git clone https://github.com/chottokun/logit-rerank-rag.git
cd logit-rerank-rag

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

Project specifications, architecture decisions, and workflow guides are documented in [docs/](docs/):
- [Architecture Index](docs/architecture/README.md)
- [Jules × agy CLI Hybrid Pipeline](docs/architecture/jules_local_hybrid_pipeline.md)
- [Local LLM Setup & Optimization](docs/infrastructure/local_llm_setup.md)
- [Specifications & Plans](plan/search_logit_sem.md)

## Development Workflow

This project utilizes a hybrid development model:
- **Cloud (Google Jules)**: Autonomous implementation planning and GitHub Pull Request generation.
- **Local (RTX 3060 / agy CLI / Gemma 4)**: On-device CUDA/PyTorch verification and zero-token-cost local patching via Antigravity SDK.
