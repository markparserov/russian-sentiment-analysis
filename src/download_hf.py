"""Download the publicly available Hugging Face sources for the project.

Saves each dataset as a parquet file under data/raw/hf/ and prints a short
schema summary (columns, dtypes, a sample row, and value counts of the most
likely label / rating columns) so we can design the label-harmonisation step.

Run with the topic-sentiment env:
    /home/zabolonkov/.conda/envs/topic-sentiment/bin/python src/download_hf.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import pandas as pd
from datasets import concatenate_datasets, load_dataset

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "hf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (hf_id, local_name)
SOURCES = [
    ("d0rj/geo-reviews-dataset-2023", "geo_reviews_2023"),
    ("ai-forever/ru-reviews-classification", "ru_reviews_classification"),
    ("lapki/perekrestok-reviews", "perekrestok_reviews"),
]

# columns we want to eyeball to design the label mapping
LABEL_HINTS = ["label", "labels", "rating", "mark", "sentiment", "score", "stars", "class"]


def summarize(name: str, df: pd.DataFrame) -> None:
    print(f"\n===== {name}: shape={df.shape} =====")
    print("columns:", list(df.columns))
    print("dtypes:\n", df.dtypes.to_string())
    with pd.option_context("display.max_colwidth", 120, "display.width", 160):
        print("sample:\n", df.head(3).to_string())
    for col in df.columns:
        if any(h in col.lower() for h in LABEL_HINTS):
            vc = df[col].value_counts(dropna=False).head(15)
            print(f"value_counts[{col}] (top 15):\n{vc.to_string()}")


def main() -> int:
    failures = []
    for hf_id, name in SOURCES:
        try:
            print(f"\n>>> loading {hf_id} ...", flush=True)
            ds = load_dataset(hf_id)
            # merge all splits into one frame; keep split name as a column
            frames = []
            for split, d in ds.items():
                f = d.to_pandas()
                f["__split"] = split
                frames.append(f)
            df = pd.concat(frames, ignore_index=True)
            out = OUT_DIR / f"{name}.parquet"
            df.to_parquet(out, index=False)
            summarize(name, df)
            print(f"saved -> {out}")
        except Exception:  # noqa: BLE001 - we want to continue with other sources
            print(f"!!! FAILED {hf_id}", flush=True)
            traceback.print_exc()
            failures.append(hf_id)
    if failures:
        print("\nFAILED SOURCES:", failures)
        return 1
    print("\nAll HF sources downloaded OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
