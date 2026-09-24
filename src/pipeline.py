from typing import List, Dict, Any, Union, Callable
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
        fallback_strategy: Union[str, Callable] = "strict",
    ) -> List[Dict[str, Any]]:
        retrieved_candidates = self.retriever.retrieve(query, top_k=initial_k)
        verified_candidates = self.router.filter_documents(
            query=query,
            candidates=retrieved_candidates,
            threshold=relevance_threshold,
        )
        
        if not verified_candidates:
            if callable(fallback_strategy):
                return fallback_strategy(query, retrieved_candidates)
            elif fallback_strategy == "strict":
                return []
            elif fallback_strategy == "message":
                return [{"text": "I don't have enough information to answer this query.", "sufficiency_prob": 0.0}]
            elif fallback_strategy == "top_1" and retrieved_candidates:
                return [retrieved_candidates[0]]
            
        return verified_candidates
