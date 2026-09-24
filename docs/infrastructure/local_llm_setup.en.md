---
okf_version: "0.2"
type: "Configuration"
title: "Local LLM Setup and Token Optimization Guide (RTX 3060 / Gemma 4 / agy CLI)"
description: "Configuration for integrating Ollama + Gemma 4 with the Antigravity SDK / agy CLI on an RTX 3060 (12GB VRAM), and know-how for applying high-quality patches with zero token consumption"
status: "stable"
generated:
  by: "reference_agent/gemini-3.8-flash"
  at: "2026-09-24T10:41:00+09:00"
tags:
  - "infrastructure"
  - "local-llm"
  - "rtx3060"
  - "gemma-4"
  - "ollama"
  - "antigravity-sdk"
sources:
  - id: "google-antigravity-local-blog"
    resource: "https://developers.googleblog.com/introducing-support-for-local-ai-models-in-the-antigravity-sdk/"
    title: "Introducing support for local AI models in the Antigravity SDK"
---

[English](local_llm_setup.en.md) | [日本語](local_llm_setup.md)

# Local LLM Setup and Token Optimization Guide (RTX 3060 / Gemma 4 / agy CLI)

## 1. Overview

This guide summarizes the setup and operational know-how for integrating Google's open model "Gemma 4" and the Antigravity SDK / `agy` CLI on an NVIDIA GeForce RTX 3060 (12GB VRAM) environment, realizing a **"zero cloud API token consumption"** and **"fast, high-quality local test-and-patch cycle"**.

---

## 2. Infrastructure & Execution Environment Requirements

1. **GPU & VRAM**:
   - NVIDIA GeForce RTX 3060 (12GB GDDR6)
   - CUDA 12.x / Driver 535+
2. **Model Execution Backend (Ollama or vLLM)**:
   - OpenAI compatible endpoint (e.g., `http://127.0.0.1:11434/v1`)
   - Model: `gemma4:latest` (Quantized to Q4_K_M or Q8_0, allowing full offload into 12GB VRAM)
3. **Antigravity SDK / CLI**:
   - `google-antigravity` Python package
   - `~/.local/bin/agy` (Antigravity CLI)

---

## 3. Local Agent Integration Code with Antigravity SDK

Based on the local model support specifications described in the Google Blog, we configure an agent using Ollama / Gemma 4 as the backend via `LocalOpenAIAgentConfig`.

```python
import os
from google.antigravity import Agent, LocalOpenAIAgentConfig, CapabilitiesConfig
from google.antigravity.hooks import policy

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Configuration to connect to Ollama (Gemma 4) on RTX 3060
local_gemma_config = LocalOpenAIAgentConfig(
    base_url=os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434/v1"),
    api_key=os.getenv("OLLAMA_API_KEY", "dummy"),
    model="gemma4:latest",
    workspaces=[PROJECT_ROOT],
    capabilities=CapabilitiesConfig(  # Allow file system read/write and command execution
        allow_file_system_write=True,
        allow_terminal_execution=True,
    ),
    policies=[policy.allow_all()]
)
```

---

## 4. Know-How for Maximizing Token Savings and Patch Quality

When applying local LLMs to coding, the biggest challenge is that "the context length and parameter size of the model are smaller than those of giant cloud LLMs." By adhering to the following principles, you can ensure reliable operation without degrading quality.

### Know-How 1: Strict Separation of "Full Comprehension" and "Local Modification"
- **Full Comprehension**: Jules (Cloud) takes an overarching view of the completed design document and the entire repository to create a PR.
- **Local Modification**: Do not force the local LLM to read the "entire picture." Feed it **only the failed test name, traceback, and a single target file**.

### Know-How 2: Test-Driven Self-Repair Prompts (Error Log Injection)
Avoid abstract instructions like "clean up the code" to the local LLM. Feed it the raw test result output to force a deterministic fix.

```python
prompt = f"""
You are a debugging worker for Python implementations.
The following error occurred during local unit testing.

[Target File]
{target_file_path}

[pytest Error Log]
{pytest_error_snippet}

[Task]
Identify the cause of the error and apply a pinpoint fix only to the relevant section in {target_file_path}.
Verify that the test passes after the modification.
"""
```

### Know-How 3: Utilizing the agy CLI
When manually trying out quick patches from the terminal, configure the `agy` CLI for the local model or define a dedicated profile to invoke it from the terminal.
```bash
agy "pytest failed on tests/test_logit_router.py. Investigate the cause and apply a patch."
```

---

## 5. Related Documents

* [Jules × agy CLI Hybrid Development Pipeline](../architecture/jules_local_hybrid_pipeline.md)
* [Knowledge Index](../README.md)
