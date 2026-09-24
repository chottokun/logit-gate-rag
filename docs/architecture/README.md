---
okf_version: "0.2"
type: "Architecture"
title: "Architecture Index / アーキテクチャインデックス"
---

# Architecture Index / アーキテクチャインデックス

This directory manages knowledge related to overall system design, development pipelines, technical decisions, and hardware benchmark evaluations.

本ディレクトリでは、システム全体の設計方針・開発パイプライン・技術選定・実機ベンチマーク評価に関するナレッジを管理します。

## Document List / ドキュメント一覧

* [Jules × agy CLI (Local GPU) ハイブリッド開発パイプライン](./jules_local_hybrid_pipeline.md) / [English](./jules_local_hybrid_pipeline.en.md)
  - クラウドJulesによるPR自動生成とRTX 3060/Gemma 4ローカルLLMによる低トークンコスト・高品質パッチサイクルの設計方針
  - *Cloud Jules PR generation & Local RTX 3060 zero-token patch cycle design*

* [大規模データセット (N=54) 実機ベンチマーク検証レポート](./benchmark_report_n54.md) / [English](./benchmark_report_n54.en.md)
  - 実務4ドメイン・N=54件における正解率96.3%・ニアミス遮断率94.4%の実証
  - *Real-device benchmark results (N=54) demonstrating 96.3% accuracy and 94.4% near-miss rejection*

* [Ruri-v3-30m × Qwen2.5 Logit ルーター実機検証ベンチマーク結果レポート](./benchmark_report_rtx3060.md) / [English](./benchmark_report_rtx3060.en.md)
  - RTX 3060実機によるニアミス遮断率100%とLLM呼び出し削減効果の実証
  - *Initial RTX 3060 verification demonstrating 100% near-miss rejection and downstream latency savings*

* [ローカルLLM活用によるクラウドトークン削減と品質両立の設計分析](./token_optimization_analysis.md) / [English](./token_optimization_analysis.en.md)
  - agy_sample.pyの知見を踏まえたトークン最小化とコード品質両立のタスク分割指針
  - *Analysis of cloud token savings and code quality optimization via local LLM*
