"""Build the out-of-bag (OOB) test set from the hand-labelled social-media sample.

Ground truth is the manually-annotated `human_sentiment` column
(0 -> negative, 1 -> neutral, 2 -> positive). All other *sentiment columns
(sentiment, sentiment_lex, xlm_sentiment) are artefacts of another project and
are ignored.

    /home/zabolonkov/.conda/envs/topic-sentiment/bin/python src/prep_oob.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "combined_platform_proportional_sample.xlsx"
OUT = ROOT / "data" / "processed" / "oob_test.parquet"

LABELS = ["negative", "neutral", "positive"]
HS2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
_WS = re.compile(r"\s+")


def clean_text(x) -> str:
    return _WS.sub(" ", str(x).replace("\n", " ").replace("\t", " ")).strip()


def main() -> None:
    df = pd.read_excel(SRC)
    out = pd.DataFrame({
        "text": df["text"].map(clean_text),
        "text_raw": df["text"].astype(str),
        "label": df["human_sentiment"].map(HS2LABEL),
        "platform": df["platform"],
        "platform_type": df["platform_type"],
    })
    out = out[(out["text"].str.len() > 0) & out["label"].isin(LABELS)].reset_index(drop=True)
    out.to_parquet(OUT, index=False)

    print(f"saved -> {OUT}  rows={len(out)}")
    print("\nlabel distribution:\n", out["label"].value_counts().reindex(LABELS).to_string())
    print("\nplatform:\n", out["platform"].value_counts().to_string())
    print("\nchar-length: median=%d  p95=%d  max=%d" % (
        out["text"].str.len().median(),
        out["text"].str.len().quantile(0.95),
        out["text"].str.len().max(),
    ))


if __name__ == "__main__":
    main()
