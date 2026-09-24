---
okf_version: "0.2"
type: "Configuration"
title: "ローカルLLM設定とトークン最適化ガイド (RTX 3060 / Gemma 4 / agy CLI)"
description: "RTX 3060 (VRAM 12GB) における Ollama + Gemma 4 の Antigravity SDK / agy CLI 連携構成と、トークン消費ゼロで高品質パッチを当てるためのノウハウ"
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

# ローカルLLM設定とトークン最適化ガイド (RTX 3060 / Gemma 4 / agy CLI)

## 1. 概要

NVIDIA GeForce RTX 3060（VRAM 12GB）環境上で、Google のオープンモデル「Gemma 4」および Antigravity SDK / `agy` CLI を連携させ、**「クラウドAPIトークン消費ゼロ」** かつ **「高速・高品質な手元テスト修正ループ」** を実現するための設定と運用ノウハウをまとめます。

---

## 2. インフラ・実行環境の要件

1. **GPU & VRAM**:
   - NVIDIA GeForce RTX 3060 (12GB GDDR6)
   - CUDA 12.x / Driver 535+
2. **モデル実行基盤 (Ollama または vLLM)**:
   - OpenAI 互換エンドポイント（例: `http://127.0.0.1:11434/v1`）
   - モデル: `gemma4:latest`（量子化 Q4_K_M または Q8_0 で 12GB VRAM 内に完全オフロード可能）
3. **Antigravity SDK / CLI**:
   - `google-antigravity` Python パッケージ
   - `~/.local/bin/agy` (Antigravity CLI)

---

## 3. Antigravity SDK によるローカルエージェント連携コード

Google Blog に記載されたローカルモデルのサポート仕様に基づき、`LocalOpenAIAgentConfig` を用いて Ollama / Gemma 4 をバックエンドとするエージェントを構成します。

```python
import os
from google.antigravity import Agent, LocalOpenAIAgentConfig, CapabilitiesConfig
from google.antigravity.hooks import policy

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# RTX 3060 上の Ollama (Gemma 4) に接続する設定
local_gemma_config = LocalOpenAIAgentConfig(
    base_url=os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434/v1"),
    api_key=os.getenv("OLLAMA_API_KEY", "dummy"),
    model="gemma4:latest",
    workspaces=[PROJECT_ROOT],
    capabilities=CapabilitiesConfig(  # ファイル読み書きやコマンド実行を許可
        allow_file_system_write=True,
        allow_terminal_execution=True,
    ),
    policies=[policy.allow_all()]
)
```

---

## 4. トークン節約とパッチ品質を最大化するノウハウ

ローカルLLMをコーディングに適用する際、最大の課題は「モデルのコンテキスト長やパラメータ規模がクラウドの巨大LLMに比べて小さいこと」です。以下の原則を徹底することで、品質を落とさず確実に動作させます。

### ノウハウ 1: 「全容理解」と「局所修正」の厳格な分離
- **全容理解**: Jules（クラウド）が完了した設計書とリポジトリ全体を俯瞰してPRを作成。
- **局所修正**: ローカルLLMには「全容」を読ませず、**失敗したテスト名、トレースバック、対象ファイル1枚** のみを与える。

### ノウハウ 2: テスト駆動による自己修復プロンプト（エラーログ注入）
ローカルLLMに対して「コードを綺麗にして」といった抽象的な指示は避け、テスト結果の出力をそのまま与えて決定論的に修正させます。

```python
prompt = f"""
あなたは Python 実装のデバッグワーカーです。
手元の単体テスト実行で以下のエラーが発生しました。

【対象ファイル】
{target_file_path}

【pytest のエラーログ】
{pytest_error_snippet}

【タスク】
エラー原因を特定し、{target_file_path} の該当箇所のみをピンポイントで修正してください。
修正後、テストがパスすることを確認してください。
"""
```

### ノウハウ 3: agy CLI の活用
手動でターミナルから素早くパッチを試す場合、`agy` CLI をローカルモデル向けに設定するか、専用のプロファイルを定義してターミナルから呼び出します。
```bash
agy "pytest で tests/test_logit_router.py がコケた。原因を調査してパッチを適用して"
```

---

## 5. 関連ドキュメント

* [Jules × agy CLI ハイブリッド開発パイプライン](../architecture/jules_local_hybrid_pipeline.md)
* [ナレッジインデックス](../README.md)
