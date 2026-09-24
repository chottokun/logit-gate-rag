---
type: "Model Specification"
title: "cl-nagoya/ruri-v3-30m 埋め込みモデル仕様と最適化"
description: "パラメータ数37M・埋め込み次元256・SentencePiece Unigram依存による極限低遅延セマンティック検索モデルの活用仕様"
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
    title: "Ruri埋め込み検索とQwen2.5選択肢ロジットルーター設計書"
---

# cl-nagoya/ruri-v3-30m 埋め込みモデル仕様と最適化

## 1. 概要

[`cl-nagoya/ruri-v3-30m`](https://huggingface.co/cl-nagoya/ruri-v3-30m) は、ModernBERT-Ja アーキテクチャを基盤とし、SentencePiece Unigram（語彙数10万）を採用した極めて軽量な日本語汎用テキスト埋め込みモデルです。

本プロジェクト（`logit-gate-rag`）における第1段階の「Bi-Encoder 事前スクリーニング（Dense Retrieval）」の**デフォルト推奨モデル**として最適です。

---

## 2. モデル特性と本システムへの適合性

| 項目 | スペック | 本パイプライン（RTX 3060 / ローカル）における利点 |
|---|---|---|
| **パラメータ数** | 37M (約3,700万) | メモリフットプリントが約 150MB 未満と極小。 |
| **埋め込み次元** | 256 次元 | ベクトル比較・コサイン類似度計算の内積演算が高速。インデックスサイズも大幅削減。 |
| **コンテキスト長** | 最大 8,192 トークン | 日本語の長い文書チャンクも切り詰めずにそのままエンコード可能。 |
| **JMTEB スコア** | 74.51 | わずか 37M パラメータでありながら、従来の数倍規模のモデルに匹敵する検索精度を達成。 |
| **トークナイザ** | SentencePiece Unigram (10万語彙) | 外部の重厚な HuggingFace Transformers 依存を排し、純粋な `sentencepiece` ＋ PyTorch / ONNX のみで超軽量に動作可能。 |

---

## 3. 非対称プレフィックス規約（必須遵守）

Ruri シリーズの対照学習の特性上、以下の接頭辞付与が必須です。

* **検索クエリ時**: `検索クエリ: ` + ユーザーの入力文
* **登録文書（パッセージ）時**: `検索文書: ` + パッセージ本文

```python
# 実装例
def format_query(text: str) -> str:
    return f"検索クエリ: {text}"

def format_document(text: str) -> str:
    return f"検索文書: {text}"
```

---

## 4. RTX 3060 / ローカル推論におけるシナリオ

1. **VRAM 競合の完全排除**:
   - 後段の Qwen2.5-1.5B (約 3〜4GB VRAM) と同時にロードしても、`ruri-v3-30m` はわずか数百MBしか消費しないため、RTX 3060 (12GB) のメモリに余裕で同居可能。
2. **CPU 単体動作 / エッジ対応**:
   - わずか 37M パラメータのため、GPU が使えない CPU 環境でも数十ミリ秒以下でクエリベクトルを生成可能。
