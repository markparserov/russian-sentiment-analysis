"""Inspect raw schemas of the searayeah GitHub files so we can write loaders.

Prints, per file: how it was read, shape, columns, a couple of rows, and
value-counts of any label-like column. Encoding/delimiter are sniffed.

    /home/zabolonkov/.conda/envs/topic-sentiment/bin/python src/inspect_searayeah.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "searayeah"


def raw_head(path: Path, n: int = 2) -> None:
    for enc in ("utf-8", "cp1251"):
        try:
            with open(path, encoding=enc) as fh:
                lines = [next(fh).rstrip("\n") for _ in range(n)]
            print(f"  [raw {enc}] first {n} lines:")
            for ln in lines:
                print("   ", repr(ln[:300]))
            return
        except (UnicodeDecodeError, StopIteration):
            continue
        except Exception as e:  # noqa: BLE001
            print(f"  raw read failed ({enc}): {e}")
            return


def show_df(tag: str, df: pd.DataFrame, label_cols=()) -> None:
    print(f"  [{tag}] shape={df.shape} cols={list(df.columns)}")
    with pd.option_context("display.max_colwidth", 90, "display.width", 160):
        print(df.head(2).to_string())
    for c in label_cols:
        if c in df.columns:
            print(f"  value_counts[{c}]:\n{df[c].value_counts(dropna=False).head(12).to_string()}")


def main() -> None:
    targets = [
        ("Kaggle-Russian-News/train.json", "json"),
        ("RuReviews/women-clothing-accessories.3-class.balanced.csv", "csv_tab"),
        ("RuSentiment/rusentiment_random_posts.csv", "csv"),
        ("Linis-Crowd-2015/text_rating_final.xlsx", "xlsx"),
        ("Linis-Crowd-2016/doc_comment_summary.xlsx", "xlsx"),
        ("RuTweetCorp/positive.csv", "csv_semicolon"),
    ]
    for rel, kind in targets:
        path = ROOT / rel
        print(f"\n===== {rel} ({kind}) exists={path.exists()} =====")
        if not path.exists():
            continue
        raw_head(path)
        try:
            if kind == "json":
                df = pd.read_json(path)
            elif kind == "csv_tab":
                df = pd.read_csv(path, sep="\t")
            elif kind == "csv_semicolon":
                df = pd.read_csv(path, sep=";", header=None, nrows=5000, on_bad_lines="skip")
            elif kind == "xlsx":
                df = pd.read_excel(path, header=None, nrows=5000)
            else:
                df = pd.read_csv(path, nrows=5000)
            show_df(kind, df, label_cols=[c for c in df.columns if str(c).lower() in {"label", "sentiment", 1, "1"}] or [df.columns[-1]])
        except Exception as e:  # noqa: BLE001
            print(f"  pandas read failed: {e}")


if __name__ == "__main__":
    main()
