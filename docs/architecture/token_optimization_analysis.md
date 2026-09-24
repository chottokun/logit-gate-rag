---
type: "Architecture Decision"
title: "ローカルLLM活用によるクラウドトークン削減と品質両立の設計分析"
description: "agy_sample.pyのアーキテクチャを踏まえ、クラウドAPIトークン消費を最小化しながら実装品質を維持するためのタスク粒度・パイプライン分割の検討"
status: "stable"
generated:
  by: "reference_agent/gemini-3.8-flash"
  at: "2026-09-24T11:08:00+09:00"
tags:
  - "architecture"
  - "token-optimization"
  - "local-llm"
  - "gemma-4"
  - "jules"
  - "agy-sample"
sources:
  - id: "agy-sample"
    resource: "/agy_sample.py"
    title: "agy_sample.py - Cloud Architect & On-Device Builder"
  - id: "google-antigravity-local-blog"
    resource: "https://developers.googleblog.com/introducing-support-for-local-ai-models-in-the-antigravity-sdk/"
    title: "Introducing support for local AI models in the Antigravity SDK"
---

[English](token_optimization_analysis.en.md) | [日本語](token_optimization_analysis.md)

# ローカルLLM活用によるクラウドトークン削減と品質両立の設計分析

## 1. 背景と課題

[agy_sample.py](../../agy_sample.py) では以下のパターンが提示されています：
1. **Cloud Architect (Gemini)**: `plan/*.md` 全体を読み込み、ローカル向けの詳細タスクリストに分解。
2. **On-Device Builder (Gemma 4)**: 分解された各タスクを 1 件ずつ Antigravity SDK 経由でローカル推論し実装。

このアプローチは「巨大な文脈理解」をクラウドに任せ、「個別コード生成」をローカルLLMにオフロードしてトークンを削減する有効な手法です。
しかし、ローカルLLM（7B〜9B規模の Gemma 4）にゼロから大規模コードを書かせると、コンポーネント間のインターフェースの不整合やライブラリ型の不整合が多発し、かえって修正ループが増えるという品質トレードオフが存在します。

---

## 2. トークン消費パターンの比較検討

| 開発アプローチ | クラウドトークン消費 | ローカル推論 (RTX 3060) | 実装品質・整合性 | 最適な適用シーン |
|---|---|---|---|---|
| **A. フルクラウド (Jules のみ)** | **高** (数十万〜数百万トークン) | なし | **極めて高い** | 大規模リファクタリング、初期リポジトリ構築 |
| **B. agy_sample.py 方式 (Cloud計画＋Local全実装)** | **低〜中** (計画分解の数千〜数万トークン) | **極大** (全コードをローカル生成) | **中** (コンポーネント間インターフェースのズレが発生しやすい) | 定型的なスクリプト群、独立したユーティリティ関数の量産 |
| **C. ハイブリッド最適化方式 (本プロジェクト推奨)** | **極小** (Julesに初期PRを1回だけ生成させる) | **中** (テスト駆動の局所パッチ・微調整のみ) | **極めて高い** (全体の骨組みはJulesが担保し、細部は実機検証で解決) | GPU/CUDA依存、外部ライブラリ連携の多いML/RAGパイプライン |

---

## 3. トークン削減と品質維持を両立する具体策 (Best Practices)

### 3.1 「ゼロからの生成」ではなく「骨組み＋局所修復」にローカルLLMを充てる
- **課題**: ローカルLLM（Gemma 4）に「Qwen2.5 の Logit 抽出と Ruri 検索の直列パイプライン全体を書いて」と指示すると、テンソル形状や未インポート例外が多発しやすい。
- **解決策**:
  - クラス設計、関数シグネチャ、基本モックテストは **Jules（クラウド）が 1 回の PR で完璧に生成**。
  - ローカル LLM（Gemma 4）には、**「pytest が失敗したエラー箇所」や「特定の関数のパラメータ追加・型アサーション」** だけを指示する。
  - これにより、クラウド API トークンの消費を最小（PR 1 回分）に抑えつつ、ローカル LLM の弱点（大局的整合性の維持困難）を完全にカバーできます。

### 3.2 プロンプト注入コンテキストの「外科手術的スライシング」
[agy_sample.py](../../agy_sample.py) のように仕様書全体を渡すのではなく、ローカルLLMへ渡すプロンプトを極小化します：
```python
# 悪い例: plan/*.md 全体と関連全ファイルをプロンプトに入れる (VRAM圧迫 & ハルシネーション誘発)
prompt = f"{full_plan_text}\n{entire_repo_context}\nFix the bug."

# 良い例: 失敗したスタックトレース + 対象関数のスコープ (20-50行) のみ注入
prompt = f"""
[Target File]: {target_file} (L40-L65)
[Pytest Error]:
{test_failure_traceback}

[Instruction]:
Adjust the tensor slicing or prefix to resolve the error above. Keep other lines unchanged.
"""
```

### 3.3 コスト対効果（ROI）の結論
1. **クラウド（Jules）の利用**: 「アーキテクチャの骨格」と「テストの定義」に限定（1機能につき 1 PR）。
2. **ローカル（Gemma 4 / agy CLI）の利用**:
   - テスト失敗時の自己修復ループ（試行回数が何十回になっても**トークン費用 0 円**）。
   - コメント・ドキュメントの拡充。
   - コーディング規約やリンターエラーの解消。
