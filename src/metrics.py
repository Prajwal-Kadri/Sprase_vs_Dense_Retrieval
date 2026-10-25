"""Metric functions. 

These four names are required since our tests import them.

gold: gold article titles for one question.
ranked: retrieved article titles, best first.
k: number of top titles to consider.

The following functions should be implemented with the given name.
Make sure output format is float.

1. recall_at
2. all_recall_at
3. mrr_at
4. near_miss_at

"""

from __future__ import annotations


def recall_at(gold: list[str], ranked: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(gold) & set(ranked[:k])) / len(set(gold))


def all_recall_at(gold: list[str], ranked: list[str], k: int) -> float:
    return 1.0 if set(gold) <= set(ranked[:k]) else 0.0


def mrr_at(gold: list[str], ranked: list[str], k: int = 10) -> float:
    gold_set = set(gold)
    for rank, title in enumerate(ranked[:k], start=1):
        if title in gold_set:
            return 1.0 / rank
    return 0.0


def near_miss_at(gold: list[str], ranked: list[str], k: int) -> float:
    return 1.0 if set(gold) & set(ranked[:k]) else 0.0