"""Deduplicate, engineer features, and create stratified splits.

Pipeline
--------
1. load interim concat_raw.parquet
2. build a normalised dedup key and resolve duplicates / label conflicts using a
   provenance priority (human > rating > distant); keys whose top-priority tier
   disagrees on the label are dropped
3. drop too-short texts
4. engineer text features (lengths, punctuation, capitalisation, ...)
5. assign stratified train/val/test splits independently for:
      split_A  -> Variant A (all rows)
      split_B  -> Variant B (label_source_type != 'rating'); NaN for rating rows
6. save data/processed/master.parquet

    /home/zabolonkov/.conda/envs/topic-sentiment/bin/python src/build_master.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

PRIORITY = {"human": 2, "rating": 1, "distant": 0}
MIN_CHARS = 3
MIN_WORDS = 1
SEED = 42

_NORM = re.compile(r"[^0-9a-zа-яё]+")


def norm_key(s: str) -> str:
    return _NORM.sub(" ", s.lower()).strip()


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dedup_key"] = df["text"].map(norm_key)
    df = df[df["dedup_key"].str.len() > 0]
    df["priority"] = df["label_source_type"].map(PRIORITY).astype("int8")
    # keep, per key, only the highest-priority tier
    df["max_prio"] = df.groupby("dedup_key")["priority"].transform("max")
    top = df[df["priority"] == df["max_prio"]].copy()
    # drop keys whose top tier disagrees on the label
    top["nlab"] = top.groupby("dedup_key")["label"].transform("nunique")
    consistent = top[top["nlab"] == 1]
    deduped = consistent.drop_duplicates("dedup_key", keep="first")
    return deduped.drop(columns=["priority", "max_prio", "nlab"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t = df["text"]
    df["n_chars"] = t.str.len()
    df["n_words"] = t.str.split().map(len)
    df["n_exclaim"] = t.str.count("!")
    df["n_question"] = t.str.count(r"\?")
    df["n_upper"] = t.str.count(r"[A-ZА-ЯЁ]")
    df["upper_ratio"] = (df["n_upper"] / df["n_chars"]).clip(0, 1)
    df["n_digits"] = t.str.count(r"\d")
    df["digit_ratio"] = (df["n_digits"] / df["n_chars"]).clip(0, 1)
    df["mean_word_len"] = (df["n_chars"] / df["n_words"]).replace([np.inf, -np.inf], np.nan)
    df["n_punct"] = t.str.count(r"[^\w\s]")
    return df


def assign_split(df: pd.DataFrame, col: str, mask: pd.Series, seed: int = SEED) -> pd.DataFrame:
    df[col] = pd.Series(pd.NA, index=df.index, dtype="string")
    sub = df[mask]
    strat = sub["source"].astype(str) + "|" + sub["label"].astype(str)
    train_idx, temp_idx = train_test_split(sub.index, test_size=0.2, random_state=seed, stratify=strat)
    temp_strat = strat.loc[temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed, stratify=temp_strat)
    df.loc[train_idx, col] = "train"
    df.loc[val_idx, col] = "val"
    df.loc[test_idx, col] = "test"
    return df


def main() -> None:
    df = pd.read_parquet(INTERIM / "concat_raw.parquet")
    n0 = len(df)
    print(f"loaded concat_raw: {n0} rows")

    df = dedup(df)
    print(f"after dedup: {len(df)} rows  (removed {n0 - len(df)} = {100*(n0-len(df))/n0:.1f}%)")

    df = add_features(df)
    df = df[(df["n_chars"] >= MIN_CHARS) & (df["n_words"] >= MIN_WORDS)].reset_index(drop=True)
    print(f"after min-length filter: {len(df)} rows")

    df = assign_split(df, "split_A", mask=pd.Series(True, index=df.index))
    df = assign_split(df, "split_B", mask=(df["label_source_type"] != "rating"))

    out = PROCESSED / "master.parquet"
    df.to_parquet(out, index=False)

    print("\n===== MASTER =====")
    print("rows:", len(df))
    print("\nby source:\n", df["source"].value_counts().to_string())
    print("\nby label:\n", df["label"].value_counts().to_string())
    print("\nVariant A split_A:\n", df["split_A"].value_counts(dropna=False).to_string())
    b = df[df["label_source_type"] != "rating"]
    print(f"\nVariant B rows: {len(b)}")
    print("Variant B by label:\n", b["label"].value_counts().to_string())
    print("Variant B split_B:\n", df["split_B"].value_counts(dropna=False).to_string())
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
