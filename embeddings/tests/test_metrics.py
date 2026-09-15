"""Unit tests for retrieval evaluation metrics."""

import math
import pytest
from indic_edge_embeddings.eval.metrics import (
    compute_dcg_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_mrr_at_k,
    compute_retrieval_metrics,
)


def test_dcg_calculation():
    """Verify DCG calculation with known manual values."""
    scores = [1, 0, 1]
    expected = (1.0 / math.log2(2)) + (0.0 / math.log2(3)) + (1.0 / math.log2(4))
    assert math.isclose(compute_dcg_at_k(scores, k=3), expected, rel_tol=1e-6)


def test_ndcg_perfect_ranking():
    """Verify nDCG is 1.0 when ranked order matches ideal order."""
    ranked_docs = ["doc1", "doc2", "doc3"]
    qrels = {"doc1": 2, "doc2": 1, "doc3": 0}
    assert math.isclose(compute_ndcg_at_k(ranked_docs, qrels, k=3), 1.0, rel_tol=1e-6)


def test_ndcg_imperfect_ranking():
    """Verify nDCG is less than 1.0 when order is suboptimal."""
    ranked_docs = ["doc2", "doc1", "doc3"]
    qrels = {"doc1": 2, "doc2": 1, "doc3": 0}
    ndcg = compute_ndcg_at_k(ranked_docs, qrels, k=3)
    assert 0.0 < ndcg < 1.0


def test_recall_at_k():
    """Verify recall calculation against relevant document set."""
    ranked_docs = ["doc1", "doc3", "doc4"]
    qrels = {"doc1": 1, "doc2": 1}
    assert math.isclose(compute_recall_at_k(ranked_docs, qrels, k=3), 0.5, rel_tol=1e-6)
    assert math.isclose(compute_recall_at_k(ranked_docs, qrels, k=1), 0.5, rel_tol=1e-6)


def test_mrr_at_k():
    """Verify MRR calculation for first relevant hit."""
    ranked_docs = ["doc3", "doc1", "doc2"]
    qrels = {"doc1": 1, "doc2": 1}
    assert math.isclose(compute_mrr_at_k(ranked_docs, qrels, k=3), 0.5, rel_tol=1e-6)


def test_compute_retrieval_metrics():
    """Verify aggregate metrics across multiple queries."""
    results = {
        "q1": [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)],
        "q2": [("doc3", 0.9), ("doc4", 0.8)],
    }
    qrels = {
        "q1": {"doc1": 1, "doc2": 1},
        "q2": {"doc4": 1},
    }
    metrics = compute_retrieval_metrics(results, qrels, k_values=(1, 5, 10))
    assert "ndcg@10" in metrics
    assert "recall@5" in metrics
    assert "mrr@1" in metrics
    assert metrics["eval_query_count"] == 2.0
