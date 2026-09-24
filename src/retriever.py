from typing import List, Dict, Any
import torch
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
