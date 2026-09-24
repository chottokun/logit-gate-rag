---
okf_version: "0.2"
type: "Model Specification"
title: "cl-nagoya/ruri-v3-30m Embedding Model Specifications and Optimization"
description: "Usage specifications for an extremely low-latency semantic search model with 37M parameters, 256 embedding dimensions, and SentencePiece Unigram dependency."
status: "stable"
generated:
  by: "reference_agent/gemini-3.8-flash"
  at: "2026-09-24T11:04:00+09:00"
tags:
  - "domain"
  - "embedding"
  - "ruri"
  - "ruri-v3-30m"
  - "sentencepiece"
sources:
  - id: "huggingface-ruri-30m"
    resource: "https://huggingface.co/cl-nagoya/ruri-v3-30m"
    title: "cl-nagoya/ruri-v3-30m Hugging Face Model Card"
  - id: "plan-search-logit-sem"
    resource: "/plan/search_logit_sem.md"
    title: "Ruri embedding search and Qwen2.5 option logit router design document"
---

[English](ruri_v3_30m.en.md) | [日本語](ruri_v3_30m.md)

# cl-nagoya/ruri-v3-30m Embedding Model Specifications and Optimization

## 1. Overview

[`cl-nagoya/ruri-v3-30m`](https://huggingface.co/cl-nagoya/ruri-v3-30m) is an extremely lightweight general-purpose Japanese text embedding model based on the ModernBERT-Ja architecture and employing SentencePiece Unigram (vocab size of 100k).

It is optimal as the **default recommended model** for the first stage "Bi-Encoder Pre-screening (Dense Retrieval)" in this project (`logit-gate-rag`).

---

## 2. Model Characteristics and Suitability for This System

| Item | Specification | Advantages in this Pipeline (RTX 3060 / Local) |
|---|---|---|
| **Parameters** | 37M (Approx. 37 million) | Extremely small memory footprint of under 150MB. |
| **Embedding Dimensions** | 256 dimensions | Fast dot product computation for vector comparison and cosine similarity. Greatly reduces index size. |
| **Context Length** | Up to 8,192 tokens | Can encode long Japanese document chunks directly without truncation. |
| **JMTEB Score** | 74.51 | Despite being only 37M parameters, it achieves search accuracy comparable to models several times larger. |
| **Tokenizer** | SentencePiece Unigram (100k vocab) | Eliminates dependency on heavy HuggingFace Transformers, operating ultra-lightly purely on `sentencepiece` + PyTorch / ONNX. |

---

## 3. Asymmetric Prefix Convention (Mandatory)

Due to the contrastive learning characteristics of the Ruri series, the following prefixes are required.

* **For Search Queries**: `検索クエリ: ` + user's input text
* **For Registered Documents (Passages)**: `検索文書: ` + passage text

```python
# Implementation Example
def format_query(text: str) -> str:
    return f"検索クエリ: {text}"

def format_document(text: str) -> str:
    return f"検索文書: {text}"
```

---

## 4. Scenarios in RTX 3060 / Local Inference

1. **Complete Elimination of VRAM Contention**:
   - Even when loaded simultaneously with the downstream Qwen2.5-1.5B (approx. 3-4GB VRAM), `ruri-v3-30m` consumes only a few hundred MB, fitting comfortably within the RTX 3060 (12GB) memory.
2. **Standalone CPU Operation / Edge Readiness**:
   - Due to having only 37M parameters, it can generate query vectors in tens of milliseconds or less even in CPU environments where a GPU is unavailable.
