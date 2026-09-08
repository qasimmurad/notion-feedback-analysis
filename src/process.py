"""
Turn raw_posts.jsonl into a labeled, analysis-ready dataset:
embeddings -> clusters (themes) -> auto-generated theme labels -> sentiment.

Usage:
    python src/process.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from langdetect import DetectorFactory, detect
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DetectorFactory.seed = 42  # make langdetect deterministic

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IN_PATH = DATA_DIR / "raw_posts.jsonl"
OUT_PATH = DATA_DIR / "processed.parquet"

MIN_TEXT_LEN = 60  # drop one-liners like "Great app!" that carry no topic signal
K_RANGE = range(6, 15)
TOP_TERMS_PER_CLUSTER = 6

# Generic review filler that would otherwise dominate every cluster's label
# regardless of topic (star-rating sentiment words, not topic words).
LABEL_STOPWORDS = ENGLISH_STOP_WORDS.union({
    "app", "apps", "notion", "great", "good", "best", "love", "loved", "amazing",
    "awesome", "excellent", "perfect", "nice", "like", "just", "really", "use",
    "using", "used", "im", "ive", "dont", "doesnt", "didnt", "cant", "wont",
    "thing", "things", "lot", "ok", "okay", "app.", "app,", "5", "star", "stars",
})


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except Exception:
        return False


def load_records() -> pd.DataFrame:
    rows = [json.loads(line) for line in IN_PATH.open()]
    df = pd.DataFrame(rows)
    df["text"] = df.apply(
        lambda r: f"{r['title']}. {r['text']}" if r.get("title") else r["text"],
        axis=1,
    )
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() >= MIN_TEXT_LEN].drop_duplicates(subset="text")

    tqdm.pandas(desc="Language detection")
    df = df[df["text"].progress_apply(is_english)]

    df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s")
    return df.reset_index(drop=True)


def embed(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


def choose_k(embeddings: np.ndarray) -> int:
    best_k, best_score = K_RANGE[0], -1
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        print(f"  k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def label_clusters(df: pd.DataFrame) -> dict[int, str]:
    # Bigrams/trigrams only: single generic words ("great", "love") are useless
    # for a theme label, but phrases ("sync issues", "database templates") aren't.
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=list(LABEL_STOPWORDS),
        ngram_range=(2, 3),
        min_df=3,
    )
    tfidf = vectorizer.fit_transform(df["text"])
    terms = np.array(vectorizer.get_feature_names_out())

    labels = {}
    for cluster_id in sorted(df["cluster"].unique()):
        mask = (df["cluster"] == cluster_id).values
        mean_tfidf = tfidf[mask].mean(axis=0).A1
        top_idx = mean_tfidf.argsort()[::-1][:TOP_TERMS_PER_CLUSTER]
        labels[cluster_id] = ", ".join(terms[top_idx])
    return labels


def score_sentiment(texts: list[str]) -> list[float]:
    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(t)["compound"] for t in texts]


def main() -> None:
    print("Loading records...")
    df = load_records()
    print(f"  {len(df)} records after cleaning/dedup")

    print("Embedding...")
    embeddings = embed(df["text"].tolist())

    print("Choosing k via silhouette score...")
    k = choose_k(embeddings)
    print(f"  chosen k={k}")

    print("Clustering...")
    df["cluster"] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(embeddings)

    print("Labeling clusters via top TF-IDF terms...")
    cluster_labels = label_clusters(df)
    df["theme"] = df["cluster"].map(cluster_labels)

    print("Scoring sentiment (VADER)...")
    df["sentiment"] = score_sentiment(df["text"].tolist())

    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")

    print("\nTheme sizes:")
    print(df["theme"].value_counts())


if __name__ == "__main__":
    main()
