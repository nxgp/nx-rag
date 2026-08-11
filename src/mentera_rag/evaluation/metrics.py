"""
Standardized Evaluation Metrics.

Layer 1: Information Retrieval (IR) metrics via ranx (MRR@K, NDCG@K, Precision@K, Recall@K).
Layer 2: Generation Quality metrics (Faithfulness & Answer Relevance).
Layer 3: System Execution metrics (Stage Latency & Token Usage).
"""

import math
from typing import Any


def calculate_ir_metrics(
    retrieved_chunk_ids: list[str],
    ground_truth_chunk_ids: list[str],
    k: int = 10,
) -> dict[str, float]:
    """
    Calculate TREC-standard Information Retrieval metrics (MRR@K, NDCG@K, Precision@K, Recall@K).
    """
    if not retrieved_chunk_ids or not ground_truth_chunk_ids:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr_at_k": 0.0, "ndcg_at_k": 0.0}

    top_k_retrieved = retrieved_chunk_ids[:k]
    relevant_set = set(ground_truth_chunk_ids)

    # 1. Precision@K
    hits = [cid in relevant_set for cid in top_k_retrieved]
    num_hits = sum(hits)
    precision = num_hits / k

    # 2. Recall@K
    recall = num_hits / len(relevant_set)

    # 3. MRR@K (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            mrr = 1.0 / rank
            break

    # 4. NDCG@K (Normalized Discounted Cumulative Gain)
    dcg = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            dcg += 1.0 / math.log2(rank + 1)

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant_set), k) + 1))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return {
        f"precision_at_{k}": round(precision, 4),
        f"recall_at_{k}": round(recall, 4),
        f"mrr_at_{k}": round(mrr, 4),
        f"ndcg_at_{k}": round(ndcg, 4),
    }


def calculate_generation_metrics(
    query: str,
    answer: str,
    context: str,
) -> dict[str, float]:
    """
    Calculate RAG answer quality metrics (Faithfulness & Answer Relevance).
    """
    if not answer or not context:
        return {"faithfulness": 0.0, "answer_relevance": 0.0}

    # Lexical overlap heuristic for verification
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    query_words = set(query.lower().split())

    # Faithfulness: fraction of answer terms grounded in source context
    overlap_with_context = answer_words.intersection(context_words)
    faithfulness = len(overlap_with_context) / max(len(answer_words), 1)

    # Relevance: fraction of query terms addressed in answer
    overlap_with_query = query_words.intersection(answer_words)
    relevance = len(overlap_with_query) / max(len(query_words), 1)

    return {
        "faithfulness": round(min(faithfulness, 1.0), 4),
        "answer_relevance": round(min(relevance, 1.0), 4),
    }


def calculate_system_metrics(
    retrieval_ms: float,
    rerank_ms: float,
    generation_ms: float,
    total_tokens: int = 0,
) -> dict[str, Any]:
    """
    Calculate system performance latencies and token costs.
    """
    total_latency_ms = retrieval_ms + rerank_ms + generation_ms
    return {
        "retrieval_latency_ms": round(retrieval_ms, 2),
        "rerank_latency_ms": round(rerank_ms, 2),
        "generation_latency_ms": round(generation_ms, 2),
        "total_latency_ms": round(total_latency_ms, 2),
        "total_tokens": total_tokens,
    }
