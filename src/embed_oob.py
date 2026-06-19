"""Embed the OOB test texts with USER-bge-m3 (same recipe as src/embed.py).

L2-normalised float16, in oob_test.parquet row order, so the saved classical
heads (trained on the full-corpus embeddings) can be applied directly.

    CUDA_VISIBLE_DEVICES=11 \
    /home/zabolonkov/.conda/envs/topic-sentiment/bin/python src/embed_oob.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODEL = "deepvk/USER-bge-m3"
OUT = PROC / "oob_emb_f16.npy"
MAX_SEQ = 512


def main() -> None:
    df = pd.read_parquet(PROC / "oob_test.parquet")
    model = SentenceTransformer(MODEL)
    model.max_seq_length = MAX_SEQ
    emb = model.encode(df["text"].tolist(), batch_size=64, show_progress_bar=True)
    emb = normalize(np.asarray(emb, dtype=np.float32), norm="l2").astype(np.float16)
    np.save(OUT, emb)
    print(f"saved -> {OUT}  shape={emb.shape}")


if __name__ == "__main__":
    main()
