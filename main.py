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


def evaluate(queries, k):
    if not isinstance(queries, list) or not queries:
        raise ValueError("queries must be a nonempty list")
    scores = [score_query(q["relevant"], q["ranking"], k) for q in queries]
    return {"k": k, "queries": scores,
            "mean": {metric: sum(s[metric] for s in scores) / len(scores) for metric in scores[0]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    try:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = Path(__file__).resolve().parent / input_path
        with input_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        print(json.dumps(evaluate(data["queries"], args.k), indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Invalid retrieval evaluation: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
