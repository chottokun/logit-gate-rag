---
okf_version: "0.2"
title: Project Knowledge Index / プロジェクトナレッジインデックス
---

# Project Knowledge Index / ナレッジインデックス

This directory serves as the root knowledge index for architecture decisions, domain knowledge, infrastructure configuration, and benchmark results under the Open Knowledge Format (OKF v0.2).

本ディレクトリは、Open Knowledge Format (OKF v0.2) に準拠した、アーキテクチャ設計・ドメイン知識・インフラ設定・ベンチマーク検証結果の最上位ナレッジインデックスです。

---

## Knowledge Domains / ナレッジ領域

- **[Architecture / アーキテクチャ](./architecture/README.md)**
  - Jules × agy CLI (Local GPU) ハイブリッド開発パイプライン / Hybrid workflow design
  - RAG Logit フィルタリングアーキテクチャ / RAG Logit filtering design
  - [N=54 実機ベンチマーク検証レポート](./architecture/benchmark_report_n54.md) / Real-device N=54 benchmark report
  - [RTX 3060 実機検証ベンチマーク結果レポート](./architecture/benchmark_report_rtx3060.md) / Real-device RTX 3060 benchmark report
  - [ローカルLLM活用によるトークン最適化分析](./architecture/token_optimization_analysis.md) / Token optimization analysis

- **[Domain / ドメインモデル](./domain/README.md)**
  - [cl-nagoya/ruri-v3-30m 埋め込みモデル仕様](./domain/ruri_v3_30m.md) / Ruri-v3-30m embedding model specifications

- **[Infrastructure / インフラストラクチャ](./infrastructure/README.md)**
  - [ローカルLLM設定とトークン最適化ガイド](./infrastructure/local_llm_setup.md) / Local LLM (RTX 3060 / Gemma 4 / Ollama) setup guide

- **[References / 参照情報](./references/README.md)**
  - 外部仕様書、Google 公式資料、検証コード / External specifications and reference code

- **[Update Log / 更新履歴](./log.md)**
  - プロジェクトの更新履歴とマイルストーン / Milestone & progress logs