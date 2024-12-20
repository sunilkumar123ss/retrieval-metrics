import math
import unittest
from main import evaluate, score_query


class RetrievalTests(unittest.TestCase):
    def test_rank_sensitive_metrics(self):
        result = score_query({"a": 3, "b": 1}, ["x", "a", "b"], 3)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertEqual(result["recall"], 1)
        self.assertEqual(result["reciprocal_rank"], 0.5)
        self.assertAlmostEqual(result["average_precision"], (0.5 + 2 / 3) / 2)
        self.assertAlmostEqual(result["ndcg"], (7 / math.log2(3) + 0.5) / (7 + 1 / math.log2(3)))

    def test_empty_judgments_and_short_ranking(self):
        self.assertEqual(score_query({}, [], 5)["ndcg"], 0)
        self.assertEqual(score_query({"a": 1}, ["a"], 5)["precision"], 0.2)

