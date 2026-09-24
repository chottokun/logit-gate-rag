# Knowledge Update Log

## 2026-09-24
* **Creation**: LLM-Wiki (OKF v0.2) ナレッジベースを初期化しました。
* **Creation**: Jules (Cloud) × agy CLI (RTX 3060 / Gemma 4) ハイブリッド開発パイプラインのアーキテクチャおよびノウハウ仕様を策定しました。
* **Creation**: cl-nagoya/ruri-v3-30m 埋め込みモデル仕様をドメインナレッジに追加しました。
* **Update**: Jules による Ruri 埋め込み検索および Qwen2.5 Logit ルーターの自動実装を完了・手元マージし、ローカル環境 (RTX 3060 / pytest) にて全件パスを確認しました。
* **Creation**: agy_sample.py のアプローチを比較分析し、ローカルLLMを用いたトークン削減と実装品質を両立させる設計指針を docs/architecture/ に文書化しました。
* **Update**: Jules (Session 13520021364216529782) による肯定的応答バイアス補正 (calibrate) とパイプラインのフォールバック戦略 (strict/message/top_1/callable) の実装・テストを完了・手元反映し、全5件の単体テストがパスしました。
* **Update**: Jules (Session 2960451210814784349) による情報充足性評価ベンチマークスイート（benchmarks/datasets/sufficiency_eval.json, benchmarks/run_benchmark.py, tests/test_benchmark.py）の実装・テストを完了・手元反映し、全8件のテストがパスしました。
* **Update**: RTX 3060 実機 GPU 上で実モデル (Ruri-v3-30m / Qwen2.5-1.5B) によるベンチマークを実行。ニアミス文書の 100% 遮断 (Logit Margin: -6.1) および回答生成 LLM スキップによる 8 秒短縮効果を実証し、レポートを docs/architecture/ に作成しました。
* **Update**: Jules (Session 15689317489505004077) によりデータセットを N=54（実務4ドメイン）に約10倍拡張し、RTX 3060 実機で検証。正解率96.3%（52/54件）、ニアミス遮断率94.4%（17/18件）、平均レイテンシ51.9ms、下流LLMスキップによる72秒短縮を実証し、レポートを docs/architecture/ に作成しました。
* **Update**: Standardized and polished all repository documentation (README.md, docs/) according to OKF v0.2 and AGENTS.md guidelines. Organized architecture diagrams and added clear quickstart guides and benchmark summaries.
* **Update**: Added full Japanese README (`README.ja.md`) with bilingual language toggles and enriched bilingual index structures across `docs/` (OKF v0.2).
* **Update**: Renamed project and repository from `logit-rerank-rag` to `logit-gate-rag` to accurately reflect its core mechanism as a high-speed sufficiency gate and router rather than a traditional reranker.


