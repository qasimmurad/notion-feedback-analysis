"""
Pull Notion's iOS App Store + Google Play reviews into data/raw_posts.jsonl.
No API key required.

Usage:
    python src/collect_reviews.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from google_play_scraper import Sort, reviews as gplay_reviews
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_PATH = DATA_DIR / "raw_posts.jsonl"

IOS_APP_ID = 1232780281
ANDROID_APP_ID = "notion.id"

# Apple's public customer-reviews RSS feed caps out around page 10 per country/sort.
IOS_COUNTRIES = ["us", "gb", "ca", "au", "in"]
IOS_PAGES_PER_COUNTRY = 10
ANDROID_REVIEW_COUNT = 2000


def collect_ios() -> list[dict]:
    print("Collecting iOS App Store reviews via Apple's public RSS feed...")
    records = []
    seen_ids = set()

    for country in IOS_COUNTRIES:
        for page in tqdm(range(1, IOS_PAGES_PER_COUNTRY + 1), desc=f"ios:{country}"):
            url = (
                f"https://itunes.apple.com/{country}/rss/customerreviews/"
                f"page={page}/id={IOS_APP_ID}/sortby=mostrecent/json"
            )
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                break
            entries = resp.json().get("feed", {}).get("entry", [])
            if not entries:
                break

            for e in entries:
                if "im:rating" not in e:
                    continue  # first entry in the feed is app metadata, not a review
                review_id = e["id"]["label"]
                if review_id in seen_ids:
                    continue
                seen_ids.add(review_id)
                records.append({
                    "type": "review",
                    "id": f"ios-{review_id}",
                    "source": f"ios_app_store_{country}",
                    "title": e.get("title", {}).get("label", ""),
                    "text": e.get("content", {}).get("label", ""),
                    "score": int(e["im:rating"]["label"]),
                    "created_utc": datetime.fromisoformat(e["updated"]["label"]).timestamp(),
                    "url": "https://apps.apple.com/us/app/notion-notes-docs-tasks/id1232780281",
                })

    print(f"  {len(records)} iOS reviews")
    return records


def collect_android() -> list[dict]:
    print("Collecting Android Play Store reviews...")
    result, _ = gplay_reviews(
        ANDROID_APP_ID,
        lang="en",
        country="us",
        sort=Sort.NEWEST,
        count=ANDROID_REVIEW_COUNT,
    )

    records = []
    for r in result:
        records.append({
            "type": "review",
            "id": f"android-{r['reviewId']}",
            "source": "android_play_store",
            "title": "",
            "text": r.get("content", ""),
            "score": r.get("score"),
            "created_utc": r["at"].timestamp(),
            "url": "https://play.google.com/store/apps/details?id=notion.id",
        })
    print(f"  {len(records)} Android reviews")
    return records


def main() -> None:
    records = collect_ios() + collect_android()

    with OUT_PATH.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(records)} records to {OUT_PATH}")


if __name__ == "__main__":
    main()
