---
okf_version: "0.2"
type: "Documentation"
title: "Document"
description: "Documentation for Document"
status: "stable"
---

### 1. ユーザー開発リポジトリ（統合対象アーキテクチャ）

* **`chottokun/ruri_with_sentencepiece_lite`**
SentencePiece単体依存による軽量な日本語テキスト埋め込みモデル運用とセマンティック検索の実装基盤。
* **`chottokun/logit-router`**
自己回帰生成を行わず、LM-Headの特定選択肢トークンロジット（Logit）を抽出して判定・ルーティングを行う軽量推論基盤。
* **`chottokun/local-ai-grep`**
ローカル環境でのファイル走査、チャンキング、grep文字列検索およびセマンティック検索を統合するオーケストレーション層。

---

### 2. Ruri 埋め込みモデル・仕様関連

* **Ruri v3 モデルカード・仕様（cl-nagoya）**
ModernBERT-Jaをベースとした日本語汎用埋め込みモデルのアーキテクチャ、SentencePiece Unigram（語彙数10万）、モデル規模（30M〜310M）およびJMTEB性能評価。


* `cl-nagoya/ruri-v3-30m`

* `cl-nagoya/ruri-v3-310m`



* **Ruri 非対称プレフィックス規約**
検索タスクにおいてクエリ（`"検索クエリ: "`）と登録文書（`"検索文書: "`）に異なる接頭辞を付与する必須仕様とベクトル類似度計算。



---

### 3. Logit抽出・Pointwiseリランキング・数理モデル関連

* **Direct Pointwise Reranking / Non-Reasoning Reranking の優位性**
自己回帰による推論系列（CoT）を生成させず、回答トークンのロジットから直接確率スコアを算定する手法の有効性と、指示追従モデルにおける肯定的応答バイアス（Positivity Bias）の分析。


* **Re-rankers as Relevance Judges & Score Thresholding**
MonoT5やRankLLaMA等のリランカーにおいて、`"true"`/`"false"` または `"Yes"`/`"No"` トークンのロジット出力から事後確率を算出し、閾値判定を行う数理的手法。


* **Lychee-Rerank 実装アーキテクチャ**
Qwenモデル等の単一フォワードパスから末尾トークンロジット（`logits[:, -1, :]`）を取得し、二値選択肢トークン間でリランキングを行う実装パターン。



---

### 4. Qwen2.5・プロンプト・推論最適化関連

* **Qwen2.5 トークナイザと ChatML 仕様**
`<|im_start|>assistant\n` 直後における単一トークンID（`"Yes"`: 9693、`"No"`: 2154）のエンコード挙動と、先行スペースの影響。


* **バッチ推論とテンソル整列**
左パディング（Left-Padding）を用いた可変長シーケンスの一括フォワードパスと、末尾ロジット層（`outputs.logits[:, -1, :]`）の効率的スライシング。