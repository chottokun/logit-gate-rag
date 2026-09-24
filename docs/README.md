---
okf_version: "0.2"
title: Project Knowledge Index
---

# Project Knowledge Index

## 概要

本リポジトリにおけるドキュメント・アーキテクチャ・開発ワークフロー・ドメインモデルの最上位ナレッジインデックスです。

## ナレッジ領域

* [Architecture](./architecture/README.md) - Jules × agy CLI (Local GPU) ハイブリッド開発パイプライン、RAG Logit フィルタリングアーキテクチャ
* [Domain](./domain/README.md) - Ruri 埋め込みモデル・Qwen2.5 Logit ルーター仕様
* [Infrastructure](./infrastructure/README.md) - RTX 3060 / Ollama / Gemma 4 / Jules CLI 環境設定
* [References](./references/README.md) - 外部仕様書、Google 公式ブログ資料、検証コード

* [Update Log](./log.md) - プロジェクトの更新履歴とマイルストーン
* [N=54 実機ベンチマーク検証レポート](./architecture/benchmark_report_n54.md) - 実務4ドメイン・N=54件における正解率96.3%の実証
* [RTX 3060 実機検証ベンチマーク結果レポート](./architecture/benchmark_report_rtx3060.md) - RTX 3060実機によるニアミス遮断率100%の実証