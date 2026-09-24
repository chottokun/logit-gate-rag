import pytest
import torch
from unittest.mock import patch, MagicMock
from src.retriever import RuriSemanticRetriever
from src.router import QwenLogitFilter
from src.pipeline import HighPrecisionRAGPipeline


@pytest.fixture
def mock_retriever():
    with patch("src.retriever.SentenceTransformer") as MockSentenceTransformer:
        # Create a mock model instance
        mock_model = MockSentenceTransformer.return_value

        # When encode is called, return dummy embeddings
        def mock_encode(texts, **kwargs):
            return torch.rand(len(texts), 256)

        mock_model.encode.side_effect = mock_encode

        retriever = RuriSemanticRetriever(model_name="mock-model", device="cpu")
        return retriever


def test_retriever(mock_retriever):
    documents = ["This is doc 1", "This is doc 2", "This is doc 3"]
    mock_retriever.index_documents(documents)

    assert len(mock_retriever.corpus_texts) == 3
    assert mock_retriever.corpus_embeddings.shape == (3, 256)

    results = mock_retriever.retrieve("test query", top_k=2)
    assert len(results) == 2
    assert "corpus_id" in results[0]
    assert "text" in results[0]
    assert "retrieval_score" in results[0]


@pytest.fixture
def mock_router():
    with (
        patch("src.router.AutoTokenizer.from_pretrained") as MockTokenizer,
        patch("src.router.AutoModelForCausalLM.from_pretrained") as MockModel,
    ):

        mock_tokenizer_instance = MockTokenizer.return_value
        mock_tokenizer_instance.pad_token = "[PAD]"
        mock_tokenizer_instance.convert_tokens_to_ids.side_effect = lambda x: (
            1 if x == "Yes" else 2
        )
        mock_tokenizer_instance.unk_token_id = 0

        # When tokenizer is called as a function
        mock_tokenizer_instance.return_value = MagicMock(
            to=lambda x: MagicMock(spec=dict)
        )

        mock_model_instance = MockModel.return_value
        mock_model_instance.to.return_value.eval.return_value = mock_model_instance

        # Mock forward pass
        def mock_forward(**kwargs):
            # Batch size is determined by kwargs
            class Output:
                def __init__(self):
                    # Batch size 2, length 10, vocab 3
                    self.logits = torch.randn(2, 10, 3)
                    # Make "Yes" (index 1) highly probable for the first item, "No" (index 2) for the second
                    self.logits[0, -1, 1] = 10.0
                    self.logits[0, -1, 2] = -10.0

                    self.logits[1, -1, 1] = -10.0
                    self.logits[1, -1, 2] = 10.0

            return Output()

        mock_model_instance.side_effect = mock_forward

        router = QwenLogitFilter(model_name="mock-model", device="cpu")
        return router


def test_router(mock_router):
    candidates = [
        {"corpus_id": 0, "text": "Good info", "retrieval_score": 0.9},
        {"corpus_id": 1, "text": "Bad info", "retrieval_score": 0.8},
    ]

    # We set a threshold such that only the first candidate should pass (as mocked above)
    filtered = mock_router.filter_documents("query", candidates, threshold=0.5)

    assert len(filtered) == 1
    assert filtered[0]["text"] == "Good info"
    assert "sufficiency_prob" in filtered[0]
    assert "logit_margin" in filtered[0]


def test_router_calibrate(mock_router):
    # Setup mock to return specific logits for calibration (batch_size=1)
    def mock_forward_calibrate(**kwargs):
        class Output:
            def __init__(self):
                self.logits = torch.randn(1, 10, 3)
                self.logits[0, -1, 1] = 5.0  # Yes
                self.logits[0, -1, 2] = 2.0  # No

        return Output()

    mock_router.model.side_effect = mock_forward_calibrate

    margin = mock_router.calibrate()
    assert margin == 3.0
    assert mock_router.baseline_margin == 3.0


def test_pipeline(mock_retriever, mock_router):
    pipeline = HighPrecisionRAGPipeline(retriever=mock_retriever, router=mock_router)

    # Mocking retrieve again so we get exact predictable candidates for the router test
    mock_retriever.retrieve = MagicMock(
        return_value=[
            {"corpus_id": 0, "text": "Good info", "retrieval_score": 0.9},
            {"corpus_id": 1, "text": "Bad info", "retrieval_score": 0.8},
        ]
    )

    results = pipeline.search_and_filter("query")
    assert len(results) == 1
    assert results[0]["text"] == "Good info"


def test_pipeline_fallback(mock_retriever, mock_router):
    pipeline = HighPrecisionRAGPipeline(retriever=mock_retriever, router=mock_router)

    # Empty verified candidates (no candidates pass threshold)
    mock_router.filter_documents = MagicMock(return_value=[])
    retrieved = [{"corpus_id": 0, "text": "Fallback info", "retrieval_score": 0.5}]
    mock_retriever.retrieve = MagicMock(return_value=retrieved)

    # test strict
    res_strict = pipeline.search_and_filter("query", fallback_strategy="strict")
    assert res_strict == []

    # test message
    res_msg = pipeline.search_and_filter("query", fallback_strategy="message")
    assert len(res_msg) == 1
    assert res_msg[0]["text"] == "I don't have enough information to answer this query."
    assert res_msg[0]["sufficiency_prob"] == 0.0

    # test top_1
    res_top1 = pipeline.search_and_filter("query", fallback_strategy="top_1")
    assert len(res_top1) == 1
    assert res_top1[0]["text"] == "Fallback info"

    # test callable
    def custom_fallback(q, candidates):
        return [{"text": "Custom fallback", "q": q}]

    res_custom = pipeline.search_and_filter("query", fallback_strategy=custom_fallback)
    assert len(res_custom) == 1
    assert res_custom[0]["text"] == "Custom fallback"
    assert res_custom[0]["q"] == "query"
