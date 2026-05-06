from __future__ import annotations

import math


def recall_at_k(ranked_items: list[int], truth: set[int], k: int) -> float:
    if not truth:
        return 0.0
    hits = sum(1 for item in ranked_items[:k] if item in truth)
    return hits / len(truth)


def ndcg_at_k(ranked_items: list[int], truth: set[int], k: int) -> float:
    if not truth:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in truth:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate_ranking_metrics(user_rankings: dict[int, list[int]], truth: dict[int, set[int]], topk: list[int]) -> dict[str, float]:
    users = [u for u in truth if truth[u]]
    metrics: dict[str, float] = {}
    for k in topk:
        recalls = [recall_at_k(user_rankings.get(u, []), truth[u], k) for u in users]
        ndcgs = [ndcg_at_k(user_rankings.get(u, []), truth[u], k) for u in users]
        metrics[f"Recall@{k}"] = sum(recalls) / max(len(recalls), 1)
        metrics[f"NDCG@{k}"] = sum(ndcgs) / max(len(ndcgs), 1)
    return metrics

