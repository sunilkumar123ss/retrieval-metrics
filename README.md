# Retrieval metrics

Score synthetic or externally generated ranked document IDs using graded relevance judgments. No search engine, embeddings, or LLM is called.

~~~sh
python3 main.py --input example.json --k 3
python3 -m unittest -v
~~~

Reports precision@k, recall@k, reciprocal rank@k, average precision@k, and nDCG@k, then their query means. Positive integer grades 1–5 count as relevant; nDCG uses exponential gain. Precision always divides by k, even for short rankings. AP divides by min(relevant count, k). Queries without relevant documents score zero and remain in the aggregate. Duplicate ranked IDs are rejected.

Judgments are required ground truth, not automatically generated truth. Scores depend on judgment coverage; unjudged results are treated as nonrelevant.

## Scope

This is an educational implementation with synthetic fixtures. Run the tests and inspect the documented limits before adapting it.

## Repository checks

```sh
python3 verify.py
```

This runs the isolated test suite, synthetic example, and CLI help. The included GitHub Actions workflow repeats these checks when the repository is published. No hosted CI result is claimed by the local verification report.
