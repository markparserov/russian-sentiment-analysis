"""Per-source loaders that map every dataset to one unified schema.

Unified columns
---------------
text                : str   - the review / post / news text
label               : str   - target in {negative, neutral, positive}
source              : str   - dataset id (e.g. geo_reviews, rusentiment)
domain              : str   - {geo, grocery, clothing, social, news, blogs, twitter}
label_source_type   : str   - provenance of the label:
                                'rating'  -> binned from a numeric star rating
                                'human'   -> manual sentiment annotation
                                'distant' -> auto label (emoji / smiley)
original_rating     : float - the original star rating where applicable, else NaN
original_label      : str   - the raw label as it appeared in the source
category            : str   - rubric / product category where available, else NaN
name                : str   - place / product name where available, else NaN
price               : float - product price (perekrestok only), else NaN

Variant A (robustness "all")     = every row
Variant B (robustness "no-rating") = rows where label_source_type != 'rating'

Run as a script to load everything and dump an interim concatenated parquet:
    /home/zabolonkov/.conda/envs/topic-sentiment/bin/python src/harmonize.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "data" / "raw" / "hf"
SR = ROOT / "data" / "raw" / "searayeah"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}

UNIFIED_COLS = [
    "text", "label", "source", "domain", "label_source_type",
    "original_rating", "original_label", "category", "name", "price",
]

_WS = re.compile(r"\s+")


def clean_text(x) -> str:
    if not isinstance(x, str):
        return ""
    return _WS.sub(" ", x.replace("\n", " ").replace("\t", " ")).strip()


def rating_to_sentiment(r: float) -> str | float:
    """1-5 star rating -> sentiment. 1-2 neg, 3 neu, 4-5 pos. NaN/0 -> NaN."""
    if pd.isna(r) or r <= 0:
        return np.nan
    r = round(float(r))
    if r <= 2:
        return "negative"
    if r == 3:
        return "neutral"
    return "positive"


def linis_to_sentiment(s: float) -> str | float:
    """LINIS Crowd score in {-2,-1,0,1,2} -> sentiment."""
    if pd.isna(s):
        return np.nan
    s = round(float(s))
    if s in (-2, -1):
        return "negative"
    if s == 0:
        return "neutral"
    if s in (1, 2):
        return "positive"
    return np.nan


def _finalize(df: pd.DataFrame, source: str, domain: str, label_source_type: str) -> pd.DataFrame:
    df = df.copy()
    df["source"] = source
    df["domain"] = domain
    df["label_source_type"] = label_source_type
    for col in UNIFIED_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    df = df[df["label"].isin(LABELS)]
    return df[UNIFIED_COLS].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# HF sources                                                                   #
# --------------------------------------------------------------------------- #
def load_geo_reviews() -> pd.DataFrame:
    df = pd.read_parquet(HF / "geo_reviews_2023.parquet")
    out = pd.DataFrame({
        "text": df["text"],
        "label": df["rating"].map(rating_to_sentiment),
        "original_rating": df["rating"].astype("float"),
        "original_label": df["rating"].astype("string"),
        "category": df["rubrics"].astype("string"),
        "name": df["name_ru"].astype("string"),
        "price": np.nan,
    })
    return _finalize(out, "geo_reviews", "geo", "rating")


def load_perekrestok() -> pd.DataFrame:
    df = pd.read_parquet(HF / "perekrestok_reviews.parquet")
    out = pd.DataFrame({
        "text": df["review_text"],
        "label": df["rating"].map(rating_to_sentiment),
        "original_rating": df["rating"].astype("float"),
        "original_label": df["rating"].astype("string"),
        "category": df["product_category"].astype("string"),
        "name": df["product_name"].astype("string"),
        "price": pd.to_numeric(df["product_price"], errors="coerce"),
    })
    return _finalize(out, "perekrestok", "grocery", "rating")


def load_ru_reviews_hf() -> pd.DataFrame:
    df = pd.read_parquet(HF / "ru_reviews_classification.parquet")
    out = pd.DataFrame({
        "text": df["text"],
        "label": df["label_text"].astype("string"),
        "original_label": df["label_text"].astype("string"),
    })
    return _finalize(out, "ru_reviews_hf", "clothing", "rating")


# --------------------------------------------------------------------------- #
# searayeah (GitHub) sources                                                   #
# --------------------------------------------------------------------------- #
def load_rureviews_github() -> pd.DataFrame:
    df = pd.read_csv(SR / "RuReviews" / "women-clothing-accessories.3-class.balanced.csv", sep="\t")
    lbl = df["sentiment"].replace({"neautral": "neutral"}).astype("string")
    out = pd.DataFrame({"text": df["review"], "label": lbl, "original_label": df["sentiment"].astype("string")})
    return _finalize(out, "rureviews_github", "clothing", "rating")


def load_rusentiment() -> pd.DataFrame:
    files = ["rusentiment_random_posts.csv", "rusentiment_preselected_posts.csv", "rusentiment_test.csv"]
    parts = [pd.read_csv(SR / "RuSentiment" / f) for f in files if (SR / "RuSentiment" / f).exists()]
    df = pd.concat(parts, ignore_index=True)
    df = df[df["label"].isin(["positive", "negative", "neutral"])]  # drop skip / speech
    out = pd.DataFrame({"text": df["text"], "label": df["label"].astype("string"), "original_label": df["label"].astype("string")})
    return _finalize(out, "rusentiment", "social", "human")


def load_kaggle_news() -> pd.DataFrame:
    parts = []
    for f in ["train.json", "test.json"]:
        p = SR / "Kaggle-Russian-News" / f
        if p.exists():
            parts.append(pd.read_json(p))
    df = pd.concat(parts, ignore_index=True)
    out = pd.DataFrame({"text": df["text"], "label": df["sentiment"].astype("string"), "original_label": df["sentiment"].astype("string")})
    return _finalize(out, "kaggle_news", "news", "human")


def load_linis() -> pd.DataFrame:
    frames = []
    for rel in ["Linis-Crowd-2015/text_rating_final.xlsx", "Linis-Crowd-2016/doc_comment_summary.xlsx"]:
        p = SR / rel
        if not p.exists():
            continue
        raw = pd.read_excel(p, header=None, usecols=[0, 1], names=["text", "score"])
        raw["score"] = pd.to_numeric(raw["score"], errors="coerce")
        raw = raw[raw["score"].isin([-2, -1, 0, 1, 2])]
        raw["text"] = raw["text"].map(clean_text)
        raw = raw[raw["text"].str.len() > 0]
        frames.append(raw)
    allrows = pd.concat(frames, ignore_index=True)
    # crowd aggregation: one row per text via mean score -> map to sentiment
    agg = allrows.groupby("text", as_index=False)["score"].mean()
    agg["label"] = agg["score"].map(linis_to_sentiment)
    out = pd.DataFrame({"text": agg["text"], "label": agg["label"].astype("string"), "original_label": agg["score"].round().astype("Int64").astype("string")})
    return _finalize(out, "linis_crowd", "blogs", "human")


def load_rutweetcorp() -> pd.DataFrame:
    parts = []
    for f, lab in [("positive.csv", "positive"), ("negative.csv", "negative")]:
        p = SR / "RuTweetCorp" / f
        if not p.exists():
            continue
        df = pd.read_csv(p, sep=";", header=None, quotechar='"', on_bad_lines="skip", dtype=str)
        part = pd.DataFrame({"text": df[3], "label": lab})
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    out = pd.DataFrame({"text": df["text"], "label": df["label"].astype("string"), "original_label": df["label"].astype("string")})
    return _finalize(out, "rutweetcorp", "twitter", "distant")


LOADERS = [
    load_geo_reviews,
    load_perekrestok,
    load_ru_reviews_hf,
    load_rureviews_github,
    load_rusentiment,
    load_kaggle_news,
    load_linis,
    load_rutweetcorp,
]


def load_all() -> pd.DataFrame:
    frames = []
    for fn in LOADERS:
        df = fn()
        frames.append(df)
        print(f"{fn.__name__:24s} rows={len(df):>8d} | {df['label'].value_counts().to_dict()}")
    out = pd.concat(frames, ignore_index=True)
    out["label_id"] = out["label"].map(LABEL_TO_ID).astype("int8")
    return out


if __name__ == "__main__":
    master = load_all()
    print("\n===== concatenated (pre-dedup) =====")
    print("total rows:", len(master))
    print("\nby source:\n", master["source"].value_counts().to_string())
    print("\nby label_source_type:\n", master["label_source_type"].value_counts().to_string())
    print("\nby label:\n", master["label"].value_counts().to_string())
    print("\nby domain:\n", master["domain"].value_counts().to_string())
    out_path = INTERIM / "concat_raw.parquet"
    master.to_parquet(out_path, index=False)
    print(f"\nsaved -> {out_path}")
