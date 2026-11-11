"""Per-race ranking metrics. `pred` = driver_ids best->worst, `actual` = finishing order."""
import math

IDCG3 = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)


def race_metrics(pred: list, actual: list) -> dict:
    winner, podium = actual[0], actual[:3]
    rank = pred.index(winner) + 1 if winner in pred else len(pred) + 1
    rel = {d: 3 - i for i, d in enumerate(podium)}  # P1=3, P2=2, P3=1
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(pred[:3]))
    return {
        "winner_mrr": 1 / rank,                                   # winner metric
        "winner_top1": float(rank == 1),
        "podium_p@3": len(set(pred[:3]) & set(podium)) / 3,       # how many of the real top 3 we named
        "podium_set": float(set(pred[:3]) == set(podium)),        # all 3, any order
        "podium_exact": float(pred[:3] == podium),                # all 3, right order
        "ndcg@3": dcg / IDCG3,
    }


METRICS = ["winner_mrr", "winner_top1", "podium_p@3", "podium_set", "podium_exact", "ndcg@3"]
