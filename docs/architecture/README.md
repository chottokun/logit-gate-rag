---
type: "Architecture"
title: "Architecture Index"
---

# Architecture Index

本ディレクトリでは、システム全体の設計方針・開発パイプライン・技術選定に関するナレッジを管理します。

## ドキュメント一覧

* [Jules × agy CLI (Local GPU) ハイブリッド開発パイプライン](./jules_local_hybrid_pipeline.md) - クラウドJulesによるPR自動生成とRTX 3060/Gemma 4ローカルLLMによる低トークンコスト・高品質パッチサイクルの設計方針
* [ローカルLLM活用によるクラウドトークン削減と品質両立の設計分析](./token_optimization_analysis.md) - agy_sample.pyの知見を踏まえたトークン最小化とコード品質両立のタスク分割指針
* [Ruri-v3-30m × Qwen2.5 Logit ルーター実機検証ベンチマーク結果レポート](./benchmark_report_rtx3060.md) - RTX 3060実機によるニアミス遮断率100%とLLM呼び出し削減効果の実証
