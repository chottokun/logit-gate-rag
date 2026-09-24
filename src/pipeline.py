from typing import List, Dict, Any
from .retriever import RuriSemanticRetriever
from .router import QwenLogitFilter


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
