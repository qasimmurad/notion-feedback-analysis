"""
Streamlit dashboard: Notion user feedback themes, sentiment, and trends
mined from iOS App Store + Google Play reviews.

Usage:
    streamlit run src/dashboard.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed.parquet"

st.set_page_config(page_title="Notion Feedback Analysis", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


df = load_data()

st.title("What Notion users are actually saying")
st.caption(
    "Themes mined from Notion's iOS App Store + Google Play reviews using sentence "
    "embeddings + clustering, scored for sentiment with VADER."
)

# --- Sidebar filters ---
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date range",
    value=(df["created_utc"].min().date(), df["created_utc"].max().date()),
)
sources = st.sidebar.multiselect(
    "Store", options=sorted(df["source"].unique()), default=sorted(df["source"].unique())
)
themes = st.sidebar.multiselect(
    "Themes", options=sorted(df["theme"].unique()), default=sorted(df["theme"].unique())
)

mask = (
    (df["created_utc"].dt.date >= date_range[0])
    & (df["created_utc"].dt.date <= date_range[1])
    & (df["source"].isin(sources))
    & (df["theme"].isin(themes))
)
fdf = df[mask]

st.markdown(f"**{len(fdf):,} reviews** in current filter")

# --- Top-level metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg star rating", f"{fdf['score'].mean():.2f}")
col2.metric("Avg sentiment", f"{fdf['sentiment'].mean():.2f}")
col3.metric("Themes", fdf["theme"].nunique())
col4.metric("Most negative theme", fdf.groupby("theme")["sentiment"].mean().idxmin())

# --- Theme volume + sentiment ---
theme_summary = (
    fdf.groupby("theme")
    .agg(volume=("id", "count"), avg_sentiment=("sentiment", "mean"))
    .sort_values("volume", ascending=False)
    .reset_index()
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Theme volume")
    fig = px.bar(theme_summary, x="volume", y="theme", orientation="h")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Sentiment by theme")
    fig = px.bar(
        theme_summary.sort_values("avg_sentiment"),
        x="avg_sentiment",
        y="theme",
        orientation="h",
        color="avg_sentiment",
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    st.plotly_chart(fig, use_container_width=True)

# --- Rating vs. sentiment sanity check ---
st.subheader("Star rating vs. text sentiment")
st.caption(
    "Sanity-checks the sentiment model against the star rating users actually left — "
    "and surfaces mismatches (e.g. a 5-star review with harsh text, or vice versa)."
)
fig = px.box(fdf, x="score", y="sentiment", points="outliers")
st.plotly_chart(fig, use_container_width=True)

# --- Trend over time ---
st.subheader("Theme volume over time")
fdf["month"] = fdf["created_utc"].dt.to_period("M").dt.to_timestamp()
trend = fdf.groupby(["month", "theme"]).size().reset_index(name="count")
fig = px.line(trend, x="month", y="count", color="theme")
st.plotly_chart(fig, use_container_width=True)

# --- Insights panel ---
st.subheader("Insights & recommendations")
worst = theme_summary.sort_values("avg_sentiment").iloc[0]
best = theme_summary.sort_values("avg_sentiment", ascending=False).iloc[0]
biggest = theme_summary.sort_values("volume", ascending=False).iloc[0]

st.markdown(f"""
- **Most negative theme**: `{worst['theme']}` (avg sentiment {worst['avg_sentiment']:.2f}, {worst['volume']} mentions) — worth prioritizing for product investigation.
- **Most positive theme**: `{best['theme']}` (avg sentiment {best['avg_sentiment']:.2f}) — a strength to reinforce in marketing/positioning.
- **Highest-volume theme**: `{biggest['theme']}` ({biggest['volume']} mentions) — whatever this is about, it's top-of-mind for the community regardless of sentiment.
""")

# --- Sample quotes ---
st.subheader("Sample quotes by theme")
selected_theme = st.selectbox("Pick a theme to read raw quotes", options=theme_summary["theme"])
sample = fdf[fdf["theme"] == selected_theme].assign(_len=lambda d: d["text"].str.len())
sample = sample.sort_values("_len", ascending=False).head(10)
for _, row in sample.iterrows():
    st.markdown(f"> {row['text'][:400]}{'...' if len(row['text']) > 400 else ''}")
    st.caption(f"{row['source']} · {row['score']}★ · sentiment {row['sentiment']:.2f} · [source]({row['url']})")
    st.divider()
