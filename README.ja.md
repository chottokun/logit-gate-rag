# logit-rerank-rag

[English](README.md) | [日本語](README.ja.md)

**Ruri** によるセマンティック密ベクトル検索と、**Qwen2.5** の Pointwise Logit 判定を組み合わせた、高精度かつ極限低遅延の RAG フィルタリングパイプラインです。

## 概要

従来の生成型 LLM によるリランカー（再順位付け）は、トークンの自己回帰デコード（Autoregressive Decoding）を伴うため、数十秒単位の大幅な遅延が発生します。
本プロジェクト `logit-rerank-rag` では、以下の 2 ステップによりこの課題を解決します：

1. **超軽量密ベクトル検索**: **Ruri-v3-30m**（37M パラメータ、埋め込み次元 256、純粋 SentencePiece Unigram）による高速な候補ドキュメント抽出。
2. **生成レス・Pointwise Logit ルーティング**: 言語モデルの LM-Head における未正規化ロジット（"Yes" / "No"）を直接評価。トークン生成を一切行わずに情報充足性を判定し、不要な後段 LLM 呼び出しを高速に遮断（Fast Reject）。

---

## アーキテクチャ構成図

```mermaid
flowchart TD
    UserQuery[ユーザーの質問] --> Embed[密ベクトル検索<br>Ruri-v3-30m]
    Embed --> Retrieve[Top-K 候補文書を抽出]
    Retrieve --> Route[Pointwise Logit ルーター<br>Qwen2.5-1.5B]
    Route -- "情報充足 (Yes > No)" --> Accept[通過: 回答生成 LLM へ転送]
    Route -- "ニアミス / 不足 (No >= Yes)" --> Reject[高速遮断: フォールバック (~52ms)]
    
    subgraph 高速拒否パイプライン (Fast Reject)
        Reject --> ReturnMessage[定型文返答 / 検索再実行<br>後段の重いLLM生成をスキップ]
    end
```

---

## 主な特長

- **Ruri-v3-30m 密ベクトル検索**:
  - パラメータ数わずか 37M、256 次元の超軽量モデル。
  - 約 10ms の極限低レイテンシで関連文書を高速スクリーニング。
- **Qwen2.5 Pointwise Logit ルーター**:
  - トークン生成（Decoding）を行わず、プロンプトの Pre-fill 単一フォワードパス（1ステップ）で末尾の `"Yes"` / `"No"` ロジットマージンを直接抽出。
  - 通常の LLM リランカーに比べ、圧倒的な低遅延（~50ms）を実現。
- **後段 LLM コスト & 時間の大幅削減**:
  - 「トピックは似ているが答えが含まれていない」というセマンティック検索特有の**ニアミス（Hard Negative）文書**を確実に検知・遮断。
  - 無駄な生成 LLM の API コストや GPU 負荷を事前にカットします。

---

## 実機ベンチマーク検証（N=54）

NVIDIA GeForce RTX 3060（12GB VRAM）実機環境において、実務 4 ドメイン（金融・人事労務・ITインフラ・製品マニュアル）の N=54 データセットを用いた実機評価を実施しました。

| 指標 | 実測値 | 備考 |
| :--- | :--- | :--- |
| **ルーティング正解率 (Accuracy)** | **96.30%** (52/54) | 正例文書・ニアミス・無関係を的確に分類 |
| **適合率 (Precision)** | **94.44%** (17/18) | 不要なクエリの通過を極小化 |
| **再現率 (Recall)** | **94.44%** (17/18) | 必要な正例文書を逃さず通過 |
| **ニアミス遮断率 (Near-miss Rejection)** | **94.44%** (17/18) | セマンティック検索では排除できないニアミスを確実に遮断 |
| **無関係文書遮断率 (Irrelevant Rejection)** | **100.00%** (18/18) | 無関係なクエリを完全遮断 |
| **平均判定レイテンシ** | **51.93 ms** | 単一フォワードパスによる極限低遅延 |
| **生成スキップによる節約効果** | **17 回スキップ（約 72 秒短縮）** | 1 回のテストで後段 LLM 呼び出しを大幅に削減 |

> 詳細は [docs/architecture/benchmark_report_n54.md](docs/architecture/benchmark_report_n54.md) をご覧ください。

---

## クイックスタートガイド

### 動作要件・インストール
パッケージマネージャーとして `uv` を使用します。

```bash
# リポジトリのクローン
git clone https://github.com/chottokun/logit-rerank-rag.git
cd logit-rerank-rag

# 依存パッケージの同期
uv sync
```

### デモの実行
提供されているデモスクリプトで、Ruri 検索と Qwen2.5 Logit ルーターの連携動作を確認できます。

```bash
uv run python demo.py
```

### ベンチマークの実行
N=54 の情報充足性評価ベンチマークスイートを実行します：

```bash
uv run python benchmarks/run_benchmark.py
```

### 単体テストの実行
```bash
uv run pytest
```

---

## ドキュメント & ナレッジベース

設計仕様や運用ノウハウは [docs/](docs/) に OKF (Open Knowledge Format) v0.2 形式で整理されています：

- **[ナレッジインデックス](docs/README.md)**: ドキュメント全体の目次
- **[アーキテクチャ設計](docs/architecture/README.md)**:
  - [Jules × agy CLI ハイブリッド開発パイプライン](docs/architecture/jules_local_hybrid_pipeline.md)
  - [N=54 大規模データセット実機検証レポート](docs/architecture/benchmark_report_n54.md)
  - [ローカルLLM活用によるトークン最適化分析](docs/architecture/token_optimization_analysis.md)
- **[ドメイン知識](docs/domain/README.md)**:
  - [cl-nagoya/ruri-v3-30m モデル仕様](docs/domain/ruri_v3_30m.md)
- **[インフラストラクチャ](docs/infrastructure/README.md)**:
  - [ローカルLLM設定ガイド (RTX 3060 / Gemma 4 / agy CLI)](docs/infrastructure/local_llm_setup.md)
- **[仕様・計画書](plan/search_logit_sem.md)**: 本プロジェクトの初期企画・設計書

---

## 開発体制・ワークフロー

本プロジェクトはクラウドとローカルを組み合わせたハイブリッド開発モデルを採用しています：
- **Cloud (Google Jules)**: リポジトリ全体のインデックス把握、自律型プランニング、GitHub Pull Request (PR) の生成。
- **Local (RTX 3060 / agy CLI / Gemma 4)**: 実機 GPU 上での CUDA / PyTorch 検証、ゼロトークンコストでのローカルパッチ適用。
