"""Evaluation modules for retrieval benchmarking."""

from indic_edge_embeddings.eval.metrics import (
    compute_retrieval_metrics,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_mrr_at_k,
)

__all__ = [
    "compute_retrieval_metrics",
    "compute_ndcg_at_k",
    "compute_recall_at_k",
    "compute_mrr_at_k",
]
