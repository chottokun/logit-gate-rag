import json
import time
import os
import argparse
from typing import List, Dict, Any

from src.retriever import RuriSemanticRetriever
from src.router import QwenLogitFilter
from src.pipeline import HighPrecisionRAGPipeline


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    total = len(results)
    if total == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "near_miss_rejection_rate": 0.0,
        }

    true_positives = sum(
        1 for r in results if r["true_label"] == 1 and r["pred_label"] == 1
    )
    false_positives = sum(
        1 for r in results if r["true_label"] == 0 and r["pred_label"] == 1
    )
    false_negatives = sum(
        1 for r in results if r["true_label"] == 1 and r["pred_label"] == 0
    )
    true_negatives = sum(
        1 for r in results if r["true_label"] == 0 and r["pred_label"] == 0
    )

    near_miss_total = sum(1 for r in results if r.get("type") == "near_miss")
    near_miss_rejected = sum(
        1 for r in results if r.get("type") == "near_miss" and r["pred_label"] == 0
    )

    accuracy = (true_positives + true_negatives) / total
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    near_miss_rejection_rate = (
        near_miss_rejected / near_miss_total if near_miss_total > 0 else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "near_miss_rejection_rate": near_miss_rejection_rate,
    }


def run_benchmark(dataset_path: str, use_gpu: bool = True):
    dataset = load_dataset(dataset_path)

    device = "cuda" if use_gpu else "cpu"

    print(f"Loading models to {device}...")

    # In a real environment, we would use the actual models, but for the sake of the benchmark logic
    # working in environments without enough memory or GPUs, we mock them if we can't load them.
    # However, since we are expected to evaluate HighPrecisionRAGPipeline, let's load them or mock them.

    # We will just instantiate the models. If it fails due to memory, we'll try to handle it.
    try:
        retriever = RuriSemanticRetriever(device=device)
        router = QwenLogitFilter(device=device)
    except Exception as e:
        print(
            f"Warning: Could not load real models on {device} ({e}). Benchmark may fail or use mocks if modified."
        )
        raise e

    pipeline = HighPrecisionRAGPipeline(retriever, router)

    print("Evaluating Baseline (Dense Retrieval Only)...")

    # For baseline, we index all documents.
    # To evaluate accuracy per query, we can just see if the retrieved doc is the correct one.
    # But the dataset format is query, document, label.
    # Let's index all unique documents.
    all_documents = list(set([item["document"] for item in dataset]))
    retriever.index_documents(all_documents)

    # Baseline logic: retrieve top-1, if the top-1 doc matches the positive doc for the query, label=1.
    # But the dataset is pairs of query/document. The evaluation might be simpler:
    # Does the retriever rank the positive doc highest?
    # Or, we can just treat the router as a zero-shot classifier on the pairs.

    baseline_results = []
    baseline_start_time = time.time()
    for item in dataset:
        query = item["query"]
        target_doc = item["document"]
        true_label = item["label"]
        item_type = item["type"]

        # Dense Retrieval alone (Baseline)
        # Assuming we retrieve from the corpus, does it return the target doc?
        # Actually, dense retrieval just scores the similarity.
        # If we just score query and target_doc, we can use a threshold.
        # But let's just do retrieval on all docs and see if target_doc is in top-1.
        retrieved = retriever.retrieve(query, top_k=1)
        pred_label = 1 if retrieved and retrieved[0]["text"] == target_doc else 0

        # If it's a near_miss, dense retrieval often ranks it high. Let's see if it's retrieved.
        if item_type == "near_miss":
            # If target doc is the near miss one, and it gets retrieved, then pred_label is 1 (False Positive)
            pred_label = 1 if retrieved and retrieved[0]["text"] == target_doc else 0

        baseline_results.append(
            {
                "id": item.get("id"),
                "query": query,
                "target_doc": target_doc,
                "true_label": true_label,
                "pred_label": pred_label,
                "type": item_type,
            }
        )
    baseline_end_time = time.time()
    baseline_latency = (baseline_end_time - baseline_start_time) / len(dataset)

    print("Evaluating HighPrecisionRAGPipeline (Ruri + Qwen2.5 Logit Router)...")
    # For pipeline, we retrieve and then route.
    pipeline_results = []
    pipeline_start_time = time.time()
    for item in dataset:
        query = item["query"]
        target_doc = item["document"]
        true_label = item["label"]
        item_type = item["type"]

        # Instead of searching the whole corpus, let's use the router directly on the pair to evaluate its classification power
        # since the dataset provides pairs.
        candidates = [{"text": target_doc}]
        verified = router.filter_documents(query, candidates, threshold=0.5)

        pred_label = 1 if len(verified) > 0 else 0

        pipeline_results.append(
            {
                "id": item.get("id"),
                "query": query,
                "target_doc": target_doc,
                "true_label": true_label,
                "pred_label": pred_label,
                "type": item_type,
            }
        )
    pipeline_end_time = time.time()
    pipeline_latency = (pipeline_end_time - pipeline_start_time) / len(dataset)

    baseline_metrics = calculate_metrics(baseline_results)
    pipeline_metrics = calculate_metrics(pipeline_results)

    print("\n--- Benchmark Results ---")
    print(f"Total Samples: {len(dataset)}")

    print("\nBaseline (Dense Retrieval Only):")
    print(f"  Accuracy:  {baseline_metrics['accuracy']:.4f}")
    print(f"  Precision: {baseline_metrics['precision']:.4f}")
    print(f"  Recall:    {baseline_metrics['recall']:.4f}")
    print(
        f"  Near-Miss Rejection Rate: {baseline_metrics['near_miss_rejection_rate']:.4f}"
    )
    print(f"  Avg Latency: {baseline_latency * 1000:.2f} ms / query")

    print("\nHighPrecisionRAGPipeline (Logit Router):")
    print(f"  Accuracy:  {pipeline_metrics['accuracy']:.4f}")
    print(f"  Precision: {pipeline_metrics['precision']:.4f}")
    print(f"  Recall:    {pipeline_metrics['recall']:.4f}")
    print(
        f"  Near-Miss Rejection Rate: {pipeline_metrics['near_miss_rejection_rate']:.4f}"
    )
    print(f"  Avg Latency: {pipeline_latency * 1000:.2f} ms / query")

    # Cost and latency savings simulation
    llm_generation_cost_per_query = 0.01  # Mock cost
    llm_generation_latency = 2.0  # seconds

    rejected_by_pipeline = sum(1 for r in pipeline_results if r["pred_label"] == 0)
    cost_saved = rejected_by_pipeline * llm_generation_cost_per_query
    latency_saved = rejected_by_pipeline * llm_generation_latency

    print("\nSimulation of Savings (Bypassing Downstream LLM):")
    print(
        f"  Queries rejected (unanswerable/near-miss): {rejected_by_pipeline}/{len(dataset)}"
    )
    print(f"  Estimated Cost Saved: ${cost_saved:.4f}")
    print(f"  Estimated Latency Saved: {latency_saved:.2f} seconds total")

    return baseline_metrics, pipeline_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG pipeline benchmark")
    parser.add_argument(
        "--dataset", type=str, default="benchmarks/datasets/sufficiency_eval.json"
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    args = parser.parse_args()

    run_benchmark(args.dataset, use_gpu=not args.cpu)
