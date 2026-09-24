import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from benchmarks.run_benchmark import load_dataset, calculate_metrics, run_benchmark


def test_load_dataset():
    data = [
        {"id": "1", "query": "Test", "document": "Doc", "label": 1, "type": "positive"}
    ]
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    loaded_data = load_dataset(temp_path)
    assert len(loaded_data) == 1
    assert loaded_data[0]["id"] == "1"

    os.remove(temp_path)


def test_calculate_metrics():
    # results = list of dict with true_label, pred_label, type
    results = [
        {"true_label": 1, "pred_label": 1, "type": "positive"},  # True Positive
        {"true_label": 1, "pred_label": 0, "type": "positive"},  # False Negative
        {"true_label": 0, "pred_label": 1, "type": "near_miss"},  # False Positive
        {"true_label": 0, "pred_label": 0, "type": "near_miss"},  # True Negative
        {"true_label": 0, "pred_label": 0, "type": "unanswerable"},  # True Negative
    ]

    metrics = calculate_metrics(results)
    assert metrics["accuracy"] == 3 / 5  # 3 correct out of 5
    assert metrics["precision"] == 1 / 2  # TP(1) / (TP(1) + FP(1))
    assert metrics["recall"] == 1 / 2  # TP(1) / (TP(1) + FN(1))
    assert (
        metrics["near_miss_rejection_rate"] == 1 / 2
    )  # 1 correctly rejected out of 2 near_misses


@patch("benchmarks.run_benchmark.HighPrecisionRAGPipeline")
@patch("benchmarks.run_benchmark.QwenLogitFilter")
@patch("benchmarks.run_benchmark.RuriSemanticRetriever")
def test_run_benchmark(mock_retriever_class, mock_router_class, mock_pipeline_class):
    mock_retriever = mock_retriever_class.return_value
    mock_router = mock_router_class.return_value

    # Mock data
    data = [
        {"id": "1", "query": "Q1", "document": "D1", "label": 1, "type": "positive"},
        {"id": "2", "query": "Q2", "document": "D2", "label": 0, "type": "near_miss"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    # Configure mock retriever to always find the document
    mock_retriever.retrieve.return_value = [
        {"text": "D1"}
    ]  # Hardcoded behavior doesn't matter much for testing the run_benchmark flow

    def mock_retrieve_func(query, top_k):
        return [{"text": "D1"}] if query == "Q1" else [{"text": "D2"}]

    mock_retriever.retrieve.side_effect = mock_retrieve_func

    # Configure router to accept positive and reject near_miss
    def mock_filter_documents(query, candidates, threshold):
        return candidates if query == "Q1" else []

    mock_router.filter_documents.side_effect = mock_filter_documents

    baseline_metrics, pipeline_metrics = run_benchmark(temp_path, use_gpu=False)

    # Baseline finds the doc correctly, but it has no way to filter near_miss in our simple simulation
    # Actually, in run_benchmark, if doc is D2 and it retrieves D2, pred_label is 1 for baseline.
    assert (
        baseline_metrics["accuracy"] == 0.5
    )  # Correct on positive, wrong on near_miss

    # Pipeline filters near_miss correctly
    assert pipeline_metrics["accuracy"] == 1.0  # Correct on both
    assert pipeline_metrics["near_miss_rejection_rate"] == 1.0

    os.remove(temp_path)
