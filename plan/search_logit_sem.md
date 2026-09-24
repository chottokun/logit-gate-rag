---
okf_version: "0.2"
type: "Documentation"
title: "Ruri埋め込み検索とQwen2.5選択肢ロジットルーターによる高精度・低遅延RAGフィルタリングの設計と実装"
description: "Documentation for Ruri埋め込み検索とQwen2.5選択肢ロジットルーターによる高精度・低遅延RAGフィルタリングの設計と実装"
status: "stable"
---

# Ruri埋め込み検索とQwen2.5選択肢ロジットルーターによる高精度・低遅延RAGフィルタリングの設計と実装

## 1. はじめに

検索拡張生成（Retrieval-Augmented Generation: RAG）の運用において、ベクトル空間上の意味的類似度に基づく密ベクトル検索（Dense Retrieval）は、広範なコーパスから関連候補を高速に絞り込む基盤技術として広く定着している。

一方で、ベクター検索はクエリと文書の表層的・意味的親和性を捉えるにとどまり、文書内に「質問に回答するための十分な事実情報が含まれているか否か」という情報充足性（Information Sufficiency）を厳密に弁別する能力には限界が存在する。

この課題を克服するため、従来は大規模言語モデル（LLM）を用いた自己回帰的なリランカー（RankGPT や Listwise 方式など）が後段に配置されてきたが、トークン逐次生成に伴う長大な遅延と高い計算コストが実サービス適用の致命的な障壁となってきた。

本稿では、軽量かつ高精度な日本語テキスト埋め込みモデルを用いたセマンティック検索（ruri_with_sentencepiece_lite）と、言語モデルの自己回帰デコードループを完全にバイパスして特定選択肢トークンの未正規化対数尤度（Logit）を直接評価する軽量ルーター（logit-router × Qwen/Qwen2.5-1.5B-Instruct）を直列統合した、高精度・超低遅延フィルタリングパイプラインの設計と実装方式を提示する。

---

## 2. システムアーキテクチャと設計思想

情報検索システムにおける候補抽出と厳密判定の分離は、計算リソース消費と情報検索精度のトレードオフを均衡させる古典的かつ堅牢なアプローチである。

本アーキテクチャでは、Bi-Encoder 構成による高速な事前スクリーニングと、Pointwise アプローチを応用した超低遅延 Logit ルーターによる二値判定フィルタリングを連続的に実行する。

### 2.1 先行検索と二段階判定

セマンティック検索段階において、ユーザーから送信された自然言語クエリは埋め込みモデルによって即座に密ベクトルへと写像され、コーパス全体からコサイン類似度の上位 $K$ 件（一般に $K = 10 \sim 20$）の候補パッセージが抽出される。

続く Logit ルーティング段階では、抽出された各候補パッセージとクエリを二値分類用のプロンプトへと整形し、バッチ形式で推論モデルに入力する。

本設計が従来の生成型リランカーと根本的に異なる点は、モデルのテキスト生成機能（Generation Loop）を一切呼び出さない点にある。

プロンプト全体のフォワードパス（Pre-fill 処理）を 1 度のみ実行し、プロンプト直後に続く最初の 1 トークン目の出力ロジット層から、肯定選択肢（"Yes"）および否定選択肢（"No"）のロジット値を抽出する。

この 2 つのロジット値に対して局所ソフトマックス関数を適用することで、当該パッセージが質問への回答に十分な情報を含んでいる確率を解析的に算出する。

算定された確率値が所定の閾値 $\tau$ を下回るパッセージ、あるいはロジット差分が負となる無関係な文書は下流のコンテキストから完全に遮断され、高い情報密度を備えた適合文書のみが最終的な回答生成 LLM へと供給される。

### 2.2 比較表

| 比較項目 | 従来の生成型リランカー | 専用 Cross-Encoder | 選択肢 Logit ルーター（本構成） |
|---|---|---|---|
| 判定処理の基本形態 | 自己回帰的なトークン逐次生成（順位リスト出力） | 分類ヘッドまたは固定トークン出力 | 事前学習済み LM-Head の特定ロジット抽出 |
| フォワードパス回数 | 数十〜数百回（生成トークン数に比例） | 1 回（シーケンス全体のエンコード） | 1 回（Pre-fill 処理のみ） |
| KV キャッシュの管理 | 必須 | 不要（Encoder 専用モデル） | 不要（生成ループを完全排除） |
| 推論遅延目安（P95） | 500 ms 〜 2,500 ms | 80 ms 〜 250 ms | 15 ms 〜 50 ms（バッチ処理時） |
| モデル適応の柔軟性 | プロンプトで調整可能だが不安定 | 事後ファインチューニングが必須 | プロンプト変更のみで判定基準を柔軟に変更可能 |

---

## 3. Ruri軽量セマンティック検索の実装要件

第 1 段階の候補抽出を担う `ruri_with_sentencepiece_lite` は、名古屋大学の研究グループが開発した日本語汎用テキスト埋め込みモデル「Ruri（v3シリーズ）」を、最小限の依存関係（純粋な SentencePiece および標準テンソル演算環境）で実行することを企図した構成である。

### 3.1 Ruri モデルファミリの特性と選定

Ruri-v3 は、ModernBERT-Ja アーキテクチャをベースとして構築されており、最大 8,192 トークンの長いコンテキストウィンドウと、10 万トークンの大規模語彙辞書（SentencePiece Unigram）を備えている。

日本語ベンチマーク（JMTEB）において極めて高い検索性能を達成しており、エッジデバイスから大規模クラウドまで用途に応じたモデル規模の選択が可能である。

| モデル識別子 | パラメータ数 | 埋め込み次元 | 推奨用途 | JMTEB 平均スコア |
|---|---:|---:|---|---:|
| `cl-nagoya/ruri-v3-30m` | 37M | 256 | 極限の低遅延・CPU/エッジ推論環境 | 74.51 |
| `cl-nagoya/ruri-v3-70m` | 70M | 384 | リソース制約下の軽量 API サーバー | 75.48 |
| `cl-nagoya/ruri-v3-130m` | 132M | 512 | 汎用クラウド・標準検索パイプライン | 76.55 |
| `cl-nagoya/ruri-v3-310m` | 315M | 768 | 高精度を最優先するエンタープライズ検索 | 77.24 |

### 3.2 非対称プレフィックスの厳格な適用規約

Ruri シリーズを運用する上で最重要となる要件は、クエリと検索対象文書に対して異なる接頭辞を付与する非対称プレフィックスルールの遵守である。

対照学習（Contrastive Pre-training）の段階で、クエリ空間と文書空間の幾何学的関係がこれら接頭辞を介して最適化されているため、プレフィックスを欠落させるとベクトル空間内での方向整合性が損なわれ、検索精度が著しく低下する。

具体的には、検索クエリには `検索クエリ: ` を、コーパスに登録する文書には `検索文書: ` を付与することが規定されている。

両者のコサイン類似度 $\text{sim}(\mathbf{q}, \mathbf{d})$ は、埋め込みベクトルを L2 正規化した後の内積演算として以下のように定義される。

$$
\mathbf{e}_q = \frac{\text{Embed}(\text{"検索クエリ: "} + q)}{\Vert{}\text{Embed}(\text{"検索クエリ: "} + q)\Vert{}_2}, \quad \mathbf{e}_d = \frac{\text{Embed}(\text{"検索文書: "} + d)}{\Vert{}\text{Embed}(\text{"検索文書: "} + d)\Vert{}_2}
$$

$$
\text{sim}(q, d) = \mathbf{e}_q \cdot \mathbf{e}_d
$$

---

## 4. Qwen2.5による選択肢Logitフィルタリングの数理と設計

### 4.1 Logit Routerの数理モデル

自己回帰型言語モデルの語彙集合を $V$、末尾予測位置における未正規化スコア（ロジット）ベクトルを $\mathbf{z} \in \mathbb{R}^{|V|}$ とする。

クエリ $q$ と候補文書 $d$ を入力系列 $X(q, d)$ としてモデルに投入した際、LM-Head から得られる肯定トークン $w_{\text{pos}}$（例: Yes）および否定トークン $w_{\text{neg}}$（例: No）のロジットをそれぞれ $z_{\text{pos}}, z_{\text{neg}}$ とする。

この 2 つのトークンのみを標本空間とする局所ソフトマックス関数により、文書 $d$ がクエリ $q$ の回答に足る情報を含む事後確率 $P(\text{Relevant} \mid q, d)$ は以下のように解析的に記述される。

$$
P(\text{Relevant} \mid q, d) = \frac{\exp(z_{\text{pos}} / T)}{\exp(z_{\text{pos}} / T) + \exp(z_{\text{neg}} / T)} = \sigma\left(\frac{z_{\text{pos}} - z_{\text{neg}}}{T}\right)
$$

ここで $\sigma(x) = \frac{1}{1 + e^{-x}}$ はシグモイド関数であり、$T > 0$ は温度パラメータを表す。

この定式化から明らかなように、適合判定の序列および相対確率はロジットマージン $\Delta z = z_{\text{pos}} - z_{\text{neg}}$ に完全に支配される。

システムが要求する適合確率閾値を $\tau \in [0, 1]$ と設定した場合、フィルタリング判定基準は以下のように表される。

$$
\text{Decision}(q, d) =
\begin{cases}
\text{Pass}, & \text{if } P(\text{Relevant} \mid q, d) \ge \tau \iff \Delta z \ge T \ln\left(\frac{\tau}{1 - \tau}\right) \\
\text{Drop}, & \text{otherwise}
\end{cases}
$$

### 4.2 トークナイザ仕様とトークンIDの同定

Qwen/Qwen2.5-1.5B-Instruct のトークナイザは Byte-Pair Encoding（BPE）に基づいており、先行スペース（Leading Space）の有無や直前の改行トークンの存在によって異なるトークンIDが割り当てられる。

ChatML 形式に従ってアシスタント開始タグ `<|im_start|>assistant\n` でプロンプトを終端させた場合、直後に続く単語は先行スペースを含まない裸のトークンとして符号化される。

| 評価対象文字列 | トークンID | 入力コンテキスト条件 | 備考 |
|---|---:|---|---|
| `"Yes"` | 9693 | `<|im_start|>assistant\n` 直後 | 最適（標準プロンプトで使用） |
| `"No"` | 2154 | `<|im_start|>assistant\n` 直後 | 最適（標準プロンプトで使用） |
| `" Yes"` | 3838 | 先行スペースを含む場合（`"Answer: "` 直後等） | フォーマット管理が煩雑になるため非推奨 |
| `" No"` | 1243 | 先行スペースを含む場合（`"Answer: "` 直後等） | フォーマット管理が煩雑になるため非推奨 |
| `"はい"` | 20042 | 日本語直接生成を要求した場合 | 事前確率の偏向が大きくキャリブレーションが困難 |
| `"いいえ"` | 42578 | 日本語直接生成を要求した場合 | 複数サブワードへ分割されるリスクがあり非推奨 |

指示追従チューニング（Instruction Tuning）が施されたモデルにおいては、指示言語が日本語であっても出力選択肢として英語の `Yes` / `No` を指定する方が、語彙空間における事前確率の対称性が高く、安定したスコアリングが実現される。

### 4.3 プロンプト設計原則

プロンプトは、モデルが単なるトピックの類似性を評価するのではなく、「クエリに回答するための具体的な情報が文書内に明記されているか」という情報充足性を二値判定するように厳格に制約する。

プロンプトの構成は、システム指示部において判定基準を明確に宣言し、ユーザー部でクエリと文書を区切り文字で明示的に分離した上で、末尾をアシスタントの開始タグで打ち切る構造を採用する。

これにより、言語モデルは次のトークンとして `Yes` または `No` を生成する確率分布を直ちに計算する状態に置かれる。

---

## 5. 低遅延化と推論最適化のエンジニアリング

リアルタイム性が求められるプロダクション環境において、後段フィルタリングの許容レイテンシは概ね数十ミリ秒以内に収める必要がある。

本設計では以下の技術的アプローチにより、推論遅延の極小化を図る。

### 5.1 デコードプロセスの完全排除

通常のテキスト生成パイプラインでは、最初のトークンを生成した後も反復的にフォワードパスを実行し、停止条件を満たすまで KV キャッシュの更新とメモリアロケーションを継続する。

対して本方式は、プロンプト全体の 1 回限りの Pre-fill フォワードパスのみで完結するため、自己回帰ループに伴うカーネル起動オーバーヘッド、CPU-GPU 間の同期遅延、および KV キャッシュのメモリ逼迫が原理的に発生しない。

### 5.2 左パディングと完全ベクトル化バッチ処理

検索によって得られた上位 $K$ 件のパッセージはそれぞれトークン長が異なる。

右パディング（Right-Padding）を用いた場合、バッチ内の各サンプルにおける最終トークンの位置が不揃いとなり、ロジット抽出時に動的なインデックス指定操作（gather 等）が必要となる。

これに対し、左パディング（Left-Padding）を適用して入力テンソルを構築すると、バッチ内の全サンプルにおいてプロンプト末尾（アシスタント開始タグ直後）が常にインデックス `-1` に完全に整列する。

これにより、`outputs.logits[:, -1, :]` という単純かつ極めて高速なメモリスライシングのみで、バッチ全体の次トークンロジットを一括取得することが可能になる。

### 5.3 ハードウェアアクセラレーションと共通プレフィックスの恩恵

PyTorch 2.0 以降に標準統合された Scaled Dot-Product Attention（SDPA）を活用することで、FlashAttention-2 バックエンドを通じた高速なアテンション計算が実行される。

さらに、バッチ内のすべてのサンプルは「システムプロンプト」および「ユーザーの検索クエリ」という共通のトークン列を先頭に共有している。

SGLang 等のエンジンや適切なアテンション最適化機構を組み合わせることで、プレフィックス部分のアテンション計算結果を候補文書間で再利用することが可能となり、計算複雑性を大幅に削減できる。

### 5.4 推論性能の目安

| 推論実行スタック | バッチサイズ（K） | P50 推論遅延（Qwen2.5-1.5B） | P95 推論遅延（Qwen2.5-1.5B） | 実装および運用上の特性 |
|---|---:|---:|---:|---|
| PyTorch Native（BF16 / SDPA） | 8 | 36 ms | 48 ms | 追加依存なし・最小構成で導入可能 |
| PyTorch + `torch.compile` | 8 | 16 ms | 22 ms | CUDA グラフ捕捉により最小オーバーヘッドを達成 |
| vLLM（Classify / Pooling API） | 8 | 21 ms | 29 ms | 連続バッチ処理に強いが専用エンドポイント設計が必要 |
| SGLang（RadixAttention） | 8 | 18 ms | 25 ms | クエリ共通プレフィックスの自動キャッシュに秀でる |

---

## 6. エンドツーエンドの統合実装

以下に示す実装は、ruri によるセマンティック検索と、Qwen/Qwen2.5-1.5B-Instruct の単一フォワードパスによる選択肢ロジット抽出・フィルタリングを直列結合した完全なパイプラインコードである。

```python
from typing import List, Dict, Any
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


class RuriSemanticRetriever:
    """
    Ruri埋め込みモデルを用いたセマンティック検索モジュール。
    公式仕様である非対称プレフィックスを厳格に適用して検索を実行する。
    """
    def __init__(self, model_name: str = "cl-nagoya/ruri-v3-30m", device: str = "cuda"):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.corpus_texts: List[str] = []
        self.corpus_embeddings: torch.Tensor = torch.empty(0)

    def index_documents(self, documents: List[str], batch_size: int = 64) -> None:
        """文書コーパスを登録し、'検索文書: ' 接頭辞を付与してベクトル化する"""
        self.corpus_texts = documents
        prefixed_docs = ["検索文書: " + doc for doc in documents]

        embeddings = self.model.encode(
            prefixed_docs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        self.corpus_embeddings = embeddings.to(self.device)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """クエリに '検索クエリ: ' 接頭辞を付与し、類似度上位Top-K件を取得する"""
        prefixed_query = "検索クエリ: " + query
        query_embedding = self.model.encode(
            [prefixed_query],
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).to(self.device)

        scores = torch.mm(query_embedding, self.corpus_embeddings.t()).squeeze(0)
        topk_scores, topk_indices = torch.topk(scores, k=min(top_k, len(self.corpus_texts)))

        results = []
        for score, idx in zip(topk_scores.tolist(), topk_indices.tolist()):
            results.append({
                "corpus_id": idx,
                "text": self.corpus_texts[idx],
                "retrieval_score": float(score),
            })
        return results


class QwenLogitFilter:
    """
    Qwen2.5-1.5B-Instructを用いた選択肢Logitルーター。
    自己回帰デコードループをバイパスし、単一フォワードパスで充足確率を算定する。
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
        ).to(device).eval()

        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("No")

        assert self.yes_token_id != self.tokenizer.unk_token_id, "Token 'Yes' not found."
        assert self.no_token_id != self.tokenizer.unk_token_id, "Token 'No' not found."

    def _build_prompt(self, query: str, document: str) -> str:
        """ChatML仕様に厳格に準拠した情報充足性判定プロンプトを構築"""
        return (
            "<|im_start|>system\n"
            "You are an information sufficiency validator. Determine whether the provided "
            "document contains explicit information to directly answer the query. "
            "Answer only with 'Yes' or 'No'.<|im_end|>\n"
            "<|im_start|>user\n"
            f"[Query]\n{query}\n\n"
            f"[Document]\n{document}\n\n"
            "Does the document contain sufficient information to answer the query? "
            "Answer 'Yes' or 'No':<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @torch.inference_mode()
    def filter_documents(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        threshold: float = 0.5,
        temperature: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        全候補パッセージを一括バッチ処理し、充足確率が閾値以上の文書のみを抽出・再ソートする
        """
        if not candidates:
            return []

        prompts = [self._build_prompt(query, c["text"]) for c in candidates]

        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoded)
        next_token_logits = outputs.logits[:, -1, :]

        yes_logits = next_token_logits[:, self.yes_token_id]
        no_logits = next_token_logits[:, self.no_token_id]

        stacked_logits = torch.stack([no_logits, yes_logits], dim=1) / temperature
        probs = F.softmax(stacked_logits, dim=-1)
        yes_probs = probs[:, 1].tolist()

        filtered_results = []
        for candidate, p_yes, z_yes, z_no in zip(
            candidates, yes_probs, yes_logits.tolist(), no_logits.tolist()
        ):
            if p_yes >= threshold:
                item = dict(candidate)
                item["sufficiency_prob"] = float(p_yes)
                item["logit_margin"] = float(z_yes - z_no)
                filtered_results.append(item)

        filtered_results.sort(key=lambda x: x["sufficiency_prob"], reverse=True)
        return filtered_results


class HighPrecisionRAGPipeline:
    """
    Ruri検索とQwen Logitルーターを直列統合したRAG検索エンジン
    """
    def __init__(self, retriever: RuriSemanticRetriever, router: QwenLogitFilter):
        self.retriever = retriever
        self.router = router

    def search_and_filter(
        self,
        query: str,
        initial_k: int = 15,
        relevance_threshold: float = 0.55,
    ) -> List[Dict[str, Any]]:
        retrieved_candidates = self.retriever.retrieve(query, top_k=initial_k)
        verified_candidates = self.router.filter_documents(
            query=query,
            candidates=retrieved_candidates,
            threshold=relevance_threshold,
        )
        return verified_candidates
```

---

## 7. キャリブレーションと運用上のトレードオフ

本システムを実運用環境へ展開する際には、指示追従済みモデル固有の確率出力バイアスの補正と、閾値設定に伴う適合率（Precision）・再現率（Recall）の制御が決定的な役割を果たす。

指示追従モデルは、アラインメント（RLHF/DPO 等）の過程で肯定的応答を好むバイアス（Positivity Bias）を獲得している傾向があり、情報が曖昧あるいは不十分なパッセージに対しても `Yes` のロジットを高めに出力しやすい。

この現象を放置すると偽陽性（False Positive）が増加し、無関係な文書がコンテキストへ混入して回答生成 LLM の幻覚を誘発する原因となる。

この対策として、クエリと無相関な空文書を入力した際の基準ロジット差分 $(z_{\text{pos}}^{(0)} - z_{\text{neg}}^{(0)})$ をベースラインとしてあらかじめ測定し、観測ロジット差分から減算するプライア・オッズ補正を導入することが有効である。

また、単に確率値 $0.5$ を境界とせず、正のマージン（例: $\Delta z \ge 1.0$）を要求するオフセット設計を採用することで、確信度の高い文書のみを選択的に通過させることが可能となる。

### 7.1 閾値の設計

- 高い閾値（$\tau \ge 0.70$）
  - 無関係なノイズパッセージは徹底的に排除される。
  - 下流の回答生成精度は向上する。
  - ただし、複合的な質問で情報が断片化している場合、個々のパッセージが閾値を下回り、すべて除外されるリスクがある。

- 低い閾値（$\tau \le 0.40$）
  - 関連情報を取りこぼす危険性は減る。
  - しかし不要なコンテキストが増え、プロンプトコストと生成遅延が膨らむ。

### 7.2 フォールバック戦略

すべての候補パッセージが閾値未満となって除外された場合（返却件数 $N=0$）に対するフォールバック戦略の事前策定も不可欠である。

この状態は「社内文書や指定コーパス内に回答が存在しない」ことを明確に示唆しているため、回答生成 LLM に推測で回答させずに即座に情報不在の定型応答を返却するか、あるいはクエリの自動書き換え（Query Expansion）をトリガーして外部検索や別コーパスへの再検索を走らせる分岐処理を実装することが、運用全体の信頼性維持に極めて重要である。

---

## 8. 総括と推奨事項

`ruri_with_sentencepiece_lite` による密ベクトル検索と、Qwen/Qwen2.5-1.5B-Instruct の LM-Head ロジットを直接抽出・評価するルーター機構の組み合わせは、計算リソースの消費を最小限に抑えつつ、RAG パイプラインの検索品質を飛躍的に向上させる。

自己回帰ループを完全に排除した単一 Pre-fill フォワードパス設計により、従来の生成型リランカーと比較して推論遅延を 10 分の 1 以下に圧縮することが可能である。

本方式の実装・展開にあたっては、次の点が強く推奨される。

- Ruri モデル固有の非対称接頭辞（`検索クエリ: ` および `検索文書: `）を前処理段階で確実に付与する。
- BPE トークナイザの特性を踏まえて先行スペースを含まない単一トークンID（`Yes` / `No`）のロジットを厳密に抽出する。
- 左パディングを適用してバッチ推論の計算効率を最大化する。

これらのエンジニアリング要件を適切に満たすことで、ミリ秒単位の超低遅延と高い情報適合性を両立した高度な RAG システム基盤が確立される。
