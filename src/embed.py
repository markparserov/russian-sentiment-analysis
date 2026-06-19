"""Compute USER-bge-m3 embeddings for the whole master table (multi-GPU).

Embeddings are L2-normalised, stored as float16 in master row order, so any
subset (Variant A / B, a split, a balanced sample) is just a row filter.

    /home/zabolonkov/.conda/envs/topic-sentiment/bin/python src/embed.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODEL = "deepvk/USER-bge-m3"
OUT = PROC / "embeddings_full_f16.npy"
META = PROC / "embeddings_meta.json"
MAX_SEQ = 512
BATCH = 64


def free_devices(min_free_ratio: float = 0.9) -> list[str]:
    devs = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        if free / total >= min_free_ratio:
            devs.append(f"cuda:{i}")
    return devs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="embed only first N rows (debug)")
    args = ap.parse_args()

    df = pd.read_parquet(PROC / "master.parquet", columns=["text"])
    docs = df["text"].tolist()
    if args.limit:
        docs = docs[: args.limit]
    print(f"docs to embed: {len(docs):,}", flush=True)

    model = SentenceTransformer(MODEL)
    model.max_seq_length = MAX_SEQ
    dim = model.get_sentence_embedding_dimension()
    print(f"model loaded: {MODEL} | dim={dim} | max_seq_length={MAX_SEQ}", flush=True)

    devices = free_devices()
    print(f"using {len(devices)} GPUs: {devices}", flush=True)

    pool = model.start_multi_process_pool(target_devices=devices)
    try:
        emb = model.encode(docs, pool=pool, batch_size=BATCH, show_progress_bar=True)
    finally:
        model.stop_multi_process_pool(pool)

    emb = normalize(np.asarray(emb, dtype=np.float32), norm="l2").astype(np.float16)
    np.save(OUT, emb)
    META.write_text(json.dumps({
        "model": MODEL, "n": int(emb.shape[0]), "dim": int(emb.shape[1]),
        "max_seq_length": MAX_SEQ, "normalized": "l2", "dtype": "float16",
        "order": "data/processed/master.parquet row order",
    }, indent=2))
    print(f"saved -> {OUT}  shape={emb.shape}  dtype={emb.dtype}", flush=True)


if __name__ == "__main__":
    main()
