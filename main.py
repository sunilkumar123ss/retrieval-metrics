"""Rank-aware evaluation of externally produced retrieval results."""
import argparse
import json
from pathlib import Path
import math
import sys


def score_query(relevant, ranking, k):
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if not isinstance(relevant, dict) or not isinstance(ranking, list):
        raise ValueError("relevant must be a grade mapping and ranking a list")
    if any(not isinstance(x, str) or not x for x in list(relevant) + ranking):
        raise ValueError("document IDs must be nonempty strings")
    if len(set(ranking)) != len(ranking):
        raise ValueError("rankings cannot contain duplicate documents")
    if any(type(v) is not int or not 1 <= v <= 5 for v in relevant.values()):
        raise ValueError("relevance grades must be integers from 1 to 5")
    hits, precision_sum, reciprocal_rank, dcg = 0, 0.0, 0.0, 0.0
    for rank, document in enumerate(ranking[:k], 1):
        grade = relevant.get(document, 0)
        if grade:
            hits += 1
            precision_sum += hits / rank
            if not reciprocal_rank:
                reciprocal_rank = 1 / rank
        dcg += (2 ** grade - 1) / math.log2(rank + 1)
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    return {
        "precision": hits / k,
        "recall": hits / len(relevant) if relevant else 0.0,
        "reciprocal_rank": reciprocal_rank,
        "average_precision": precision_sum / min(len(relevant), k) if relevant else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
    }

