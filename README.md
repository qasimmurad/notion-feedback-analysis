# Notion Feedback Analysis

What are Notion users actually saying about the product, and how do they feel about it?

This project mines Notion's public iOS App Store + Google Play reviews, clusters
them into themes using sentence embeddings, scores sentiment, and surfaces the
results in an interactive dashboard — with an eye toward the kind of
"insight -> actionable recommendation" work a product/growth data science team does.

**[Live dashboard →](#)** *(link added after deploy)*

## Pipeline

1. **Collect** (`src/collect_reviews.py`) — pulls Notion's iOS App Store (via
   Apple's public RSS reviews feed) and Google Play reviews. No API key required.
2. **Process** (`src/process.py`) — filters to English reviews, embeds each one
   with `sentence-transformers` (all-MiniLM-L6-v2), clusters into themes with
   KMeans (k chosen via silhouette score), auto-labels each cluster with its top
   TF-IDF bigrams/trigrams, and scores sentiment with VADER.
3. **Dashboard** (`src/dashboard.py`) — Streamlit app: theme volume, sentiment by
   theme, star-rating-vs-sentiment sanity check, trends over time, sample quotes,
   and an auto-generated insights panel.

The repo ships with `data/processed.parquet` already generated, so the dashboard
runs immediately without needing to re-run the ML pipeline.

## Run the dashboard only

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/dashboard.py
```

## Reproduce the full pipeline (collect + process)

```bash
pip install -r requirements-dev.txt
python src/collect_reviews.py   # -> data/raw_posts.jsonl
python src/process.py           # -> data/processed.parquet
```

## Notes

- Everything runs locally, no API keys or credentials needed to reproduce.
- Embeddings/clustering run on-device — no LLM API calls involved.
