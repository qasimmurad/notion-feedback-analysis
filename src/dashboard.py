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

st.set_page_config(
    page_title="Notion Feedback Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",  # keep the first view uncluttered; filters are opt-in
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


df = load_data()

st.title("What Notion users are actually saying")

with st.container(border=True):
    st.markdown(
        "**Why this exists:** app-store reviews are unstructured and easy to skim past "
        "one at a time. This turns ~2,800 of them into a small number of *quantified* "
        "themes — how much volume each one has, and how positive or negative it skews — "
        "so a product team can answer \"what should we fix first?\" with evidence instead "
        "of anecdote. Use the filters (sidebar, top-left ›) to slice by store or date."
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
    "Themes", options=sorted(df["theme_label"].unique()), default=sorted(df["theme_label"].unique())
)

mask = (
    (df["created_utc"].dt.date >= date_range[0])
    & (df["created_utc"].dt.date <= date_range[1])
    & (df["source"].isin(sources))
    & (df["theme_label"].isin(themes))
)
fdf = df[mask]

st.caption(f"{len(fdf):,} reviews in current filter")

# --- Top-level metrics ---
theme_summary = (
    fdf.groupby("theme_label")
    .agg(volume=("id", "count"), avg_sentiment=("sentiment", "mean"))
    .sort_values("volume", ascending=False)
    .reset_index()
)
worst = theme_summary.sort_values("avg_sentiment").iloc[0]
best = theme_summary.sort_values("avg_sentiment", ascending=False).iloc[0]
biggest = theme_summary.sort_values("volume", ascending=False).iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg star rating", f"{fdf['score'].mean():.2f}")
col2.metric("Avg sentiment", f"{fdf['sentiment'].mean():.2f}")
col3.metric("Themes", fdf["theme_label"].nunique())
col4.metric(
    "Most negative theme",
    worst["theme_label"],
    help=f"avg sentiment {worst['avg_sentiment']:.2f} · {worst['volume']} mentions",
)

st.divider()

# --- Theme volume + sentiment ---
st.subheader("Themes: volume vs. sentiment")
st.caption("What people talk about most (left) isn't always what they're happiest about (right).")

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(theme_summary, x="volume", y="theme_label", orientation="h", labels={"theme_label": ""})
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380, margin=dict(l=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.bar(
        theme_summary.sort_values("avg_sentiment"),
        x="avg_sentiment",
        y="theme_label",
        orientation="h",
        color="avg_sentiment",
        color_continuous_scale="RdYlGn",
        labels={"theme_label": ""},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380, margin=dict(l=10))
    st.plotly_chart(fig, use_container_width=True)

# --- Rating vs. sentiment sanity check ---
with st.expander("Sanity check: does text sentiment agree with the star rating?"):
    st.caption(
        "Cross-checks the sentiment model against the star rating users actually left, "
        "and surfaces mismatches — e.g. a sarcastic 1-star review the model reads as positive."
    )
    fig = px.box(fdf, x="score", y="sentiment", points="outliers")
    st.plotly_chart(fig, use_container_width=True)

# --- Trend over time ---
with st.expander("Theme volume over time"):
    fdf["month"] = fdf["created_utc"].dt.to_period("M").dt.to_timestamp()
    trend = fdf.groupby(["month", "theme_label"]).size().reset_index(name="count")
    fig = px.line(trend, x="month", y="count", color="theme_label", labels={"theme_label": "theme"})
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Insights panel ---
st.subheader("Insights & recommendations")
st.markdown(f"""
- 🔴 **Fix first:** *{worst['theme_label']}* is the most negative theme (avg sentiment {worst['avg_sentiment']:.2f}, {worst['volume']} mentions) — a concrete candidate for prioritized product investigation.
- 🟢 **Protect it:** *{best['theme_label']}* is the most positive theme (avg sentiment {best['avg_sentiment']:.2f}) — a strength worth reinforcing in marketing/positioning, not just fixing what's broken.
- 📊 **Top of mind:** *{biggest['theme_label']}* is the highest-volume theme ({biggest['volume']} mentions) regardless of sentiment — whatever it's about, it's what the community talks about most.
""")

st.divider()

# --- Accessibility & inclusion lens ---
st.subheader("Accessibility & inclusion lens")
st.caption(
    "A different cut on the same data: instead of clustering by topic, this scans review "
    "text for self-reported accessibility or neurodivergence context (e.g. \"ADHD\", "
    "\"dyslexia\", \"screen reader\") to see how well the product serves those users "
    "specifically — a lens a generic theme-clustering dashboard would miss entirely."
)

acc = fdf[fdf["mentions_accessibility"]]
acc_pct = len(acc) / len(fdf) * 100 if len(fdf) else 0

if len(acc) == 0:
    st.info("No accessibility-related mentions in the current filter.")
else:
    st.warning(
        f"⚠️ Small sample: only **{len(acc)} of {len(fdf):,} reviews** ({acc_pct:.1f}%) "
        "mention accessibility/neurodivergence context. Treat this as a directional signal "
        "to investigate further (e.g. a targeted user survey), not a statistically robust "
        "conclusion on its own.",
        icon="⚠️",
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Flagged reviews", f"{len(acc)}", f"{acc_pct:.1f}% of filtered total")
    m2.metric("Avg sentiment (flagged)", f"{acc['sentiment'].mean():.2f}", f"{acc['sentiment'].mean() - fdf['sentiment'].mean():+.2f} vs. overall")
    m3.metric("Avg stars (flagged)", f"{acc['score'].mean():.2f}", f"{acc['score'].mean() - fdf['score'].mean():+.2f} vs. overall")

    tag_counts = (
        acc.assign(tag=acc["accessibility_tags"].str.split(", "))
        .explode("tag")
        .groupby("tag")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="count")
    )
    cc1, cc2 = st.columns([1, 1])
    with cc1:
        fig = px.bar(tag_counts, x="count", y="tag", orientation="h", labels={"tag": ""})
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=280, margin=dict(l=10))
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        top_tag = tag_counts.iloc[0]["tag"]
        if top_tag == "ADHD" and acc["sentiment"].mean() > fdf["sentiment"].mean():
            st.markdown(
                f"**Takeaway:** ADHD is by far the most-mentioned accessibility context "
                f"({int(tag_counts.iloc[0]['count'])} mentions), and sentiment among those "
                f"reviewers runs *higher* than the overall average — several describe Notion "
                f"as \"life changing\" for managing ADHD specifically. That's a genuine, "
                f"organic strength worth understanding and protecting (e.g. in user research, "
                f"onboarding flows aimed at this use case), not just a compliance checkbox."
            )
        else:
            st.markdown(
                f"**Takeaway:** the most-mentioned category is **{top_tag}** "
                f"({int(tag_counts.iloc[0]['count'])} mentions). Sample size is too small "
                f"here to generalize confidently — worth validating with a larger, "
                f"purpose-built accessibility survey."
            )

    with st.expander("Read the flagged quotes"):
        for _, row in acc.sort_values("score").iterrows():
            st.markdown(f"> {row['text'][:400]}{'...' if len(row['text']) > 400 else ''}")
            st.caption(f"{row['accessibility_tags']} · {row['source']} · {row['score']}★ · sentiment {row['sentiment']:.2f}")
            st.divider()

st.divider()

# --- Sample quotes ---
st.subheader("Sample quotes by theme")
selected_theme = st.selectbox("Pick a theme to read raw quotes", options=theme_summary["theme_label"])
raw_keywords = fdf.loc[fdf["theme_label"] == selected_theme, "theme"].iloc[0]
st.caption(f"Top keywords for this cluster: {raw_keywords}")

sample = fdf[fdf["theme_label"] == selected_theme].assign(_len=lambda d: d["text"].str.len())
sample = sample.sort_values("_len", ascending=False).head(10)
for _, row in sample.iterrows():
    st.markdown(f"> {row['text'][:400]}{'...' if len(row['text']) > 400 else ''}")
    st.caption(f"{row['source']} · {row['score']}★ · sentiment {row['sentiment']:.2f} · [source]({row['url']})")
    st.divider()
