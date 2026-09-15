"""Retrieval evaluation metrics calculation."""

import logging
import math
from typing import Dict, List, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


def compute_dcg_at_k(relevance_scores: Sequence[Union[int, float]], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank k.

    Args:
        relevance_scores: Binary or graded relevance labels in ranked order.
        k: Maximum rank depth to evaluate.

    Returns:
        DCG score at rank k.
    """
    dcg = 0.0
    for rank, score in enumerate(relevance_scores[:k], start=1):
        if score > 0:
            dcg += float(score) / math.log2(rank + 1)
    return dcg


def compute_ndcg_at_k(
    ranked_doc_ids: Sequence[str],
    relevant_doc_ids: Dict[str, Union[int, float]],
    k: int = 10,
) -> float:
    """Compute normalized Discounted Cumulative Gain at rank k for a single query.

    Args:
        ranked_doc_ids: Ordered list of retrieved document identifiers.
        relevant_doc_ids: Mapping of relevant document ID to its relevance score.
        k: Cutoff rank (default: 10).

    Returns:
        nDCG@k value between 0.0 and 1.0.
    """
    if not relevant_doc_ids:
        return 0.0

    actual_scores = [relevant_doc_ids.get(doc_id, 0.0) for doc_id in ranked_doc_ids[:k]]
    actual_dcg = compute_dcg_at_k(actual_scores, k)

    ideal_scores = sorted(relevant_doc_ids.values(), reverse=True)[:k]
    ideal_dcg = compute_dcg_at_k(ideal_scores, k)

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


def compute_recall_at_k(
    ranked_doc_ids: Sequence[str],
    relevant_doc_ids: Dict[str, Union[int, float]],
    k: int = 20,
) -> float:
    """Compute Recall at rank k for a single query.

    Args:
        ranked_doc_ids: Ordered list of retrieved document identifiers.
        relevant_doc_ids: Mapping of relevant document ID to its relevance score.
        k: Cutoff rank (default: 20).

    Returns:
        Fraction of relevant documents retrieved in the top k positions.
    """
    positive_relevant_ids = {
        doc_id for doc_id, score in relevant_doc_ids.items() if score > 0
    }
    if not positive_relevant_ids:
        return 0.0

    retrieved_relevant = sum(
        1 for doc_id in ranked_doc_ids[:k] if doc_id in positive_relevant_ids
    )
    return retrieved_relevant / len(positive_relevant_ids)


def compute_mrr_at_k(
    ranked_doc_ids: Sequence[str],
    relevant_doc_ids: Dict[str, Union[int, float]],
    k: int = 10,
) -> float:
    """Compute Mean Reciprocal Rank at cutoff k for a single query.

    Args:
        ranked_doc_ids: Ordered list of retrieved document identifiers.
        relevant_doc_ids: Mapping of relevant document ID to its relevance score.
        k: Cutoff rank (default: 10).

    Returns:
        Reciprocal rank of the first relevant document, or 0.0 if none in top k.
    """
    positive_relevant_ids = {
        doc_id for doc_id, score in relevant_doc_ids.items() if score > 0
    }
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in positive_relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_retrieval_metrics(
    results: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, Union[int, float]]],
    k_values: Sequence[int] = (1, 5, 10, 20),
) -> Dict[str, float]:
    """Compute aggregate retrieval metrics across all queries.

    Args:
        results: Mapping from query_id to list of (doc_id, score) tuples.
        qrels: Mapping from query_id to dict of (doc_id -> relevance).
        k_values: Tuple of k cutoffs to compute metrics for.

    Returns:
        Dictionary mapping metric names (e.g. 'ndcg@10', 'recall@20') to mean values.
    """
    if not results or not qrels:
        logger.warning("Empty results or qrels provided to compute_retrieval_metrics")
        return {}

    eval_query_ids = [qid for qid in qrels if qid in results]
    if not eval_query_ids:
        logger.warning("No overlapping queries between results and qrels")
        return {}

    metrics_acc: Dict[str, float] = {}
    for k in k_values:
        metrics_acc[f"ndcg@{k}"] = 0.0
        metrics_acc[f"recall@{k}"] = 0.0
        metrics_acc[f"mrr@{k}"] = 0.0

    for qid in eval_query_ids:
        ranked_docs = [doc_id for doc_id, _ in results[qid]]
        query_qrels = qrels[qid]

        for k in k_values:
            metrics_acc[f"ndcg@{k}"] += compute_ndcg_at_k(ranked_docs, query_qrels, k=k)
            metrics_acc[f"recall@{k}"] += compute_recall_at_k(
                ranked_docs, query_qrels, k=k
            )
            metrics_acc[f"mrr@{k}"] += compute_mrr_at_k(ranked_docs, query_qrels, k=k)

    num_queries = len(eval_query_ids)
    mean_metrics: Dict[str, float] = {
        metric: val / num_queries for metric, val in metrics_acc.items()
    }
    mean_metrics["eval_query_count"] = float(num_queries)
    return mean_metrics
