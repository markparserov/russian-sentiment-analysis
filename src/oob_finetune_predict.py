"""Predict on the OOB set with the saved fine-tuned encoder models (A and B).

The fine-tuned encoder is a tuned hyper-parameter in src/finetune.py (USER-bge-m3 /
USER2-base / ru-en-RoSBERTa / rubert-mini-frida), so the model label is derived from
the actually selected encoder recorded in finetune_meta.json rather than hard-coded.

Loads models/finetune_A and models/finetune_B (created by src/finetune.py) and
writes data/processed/preds/oob_ours_finetune.parquet.

    CUDA_VISIBLE_DEVICES=0 \
    /home/zabolonkov/.conda/envs/topic-sentiment/bin/python src/oob_finetune_predict.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
PRED = PROC / "preds"
LABELS = ["negative", "neutral", "positive"]
MAXLEN = 256
BS = 64


def ft_short() -> str:
    """Short name of the fine-tuned encoder chosen by the search (from meta)."""
    meta = PRED / "finetune_meta.json"
    enc = "deepvk/USER-bge-m3"
    if meta.exists():
        enc = json.loads(meta.read_text()).get("best_config", {}).get("encoder", enc)
    return enc.split("/")[-1]


@torch.no_grad()
def predict(model_dir, texts):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to("cuda").eval().half()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    out = []
    for i in range(0, len(texts), BS):
        batch = tok(texts[i:i + BS], truncation=True, max_length=MAXLEN,
                    padding=True, return_tensors="pt").to("cuda")
        ids = model(**batch).logits.argmax(-1).cpu().numpy()
        out.append(ids)
    ids = np.concatenate(out)
    return np.array([id2label[int(i)] for i in ids])


def main() -> None:
    oob = pd.read_parquet(PROC / "oob_test.parquet")
    texts = oob["text"].tolist()
    y = oob["label"].to_numpy()
    short = ft_short()
    rows = []
    for v in ["A", "B"]:
        d = MODELS / f"finetune_{v}"
        if not d.exists():
            print(f"!! {d} not found, skipping")
            continue
        pred = predict(d, texts)
        f1m = f1_score(y, pred, average="macro", labels=LABELS)
        name = f"{short} fine-tuned ({v})"
        print(f"{name}  OOB macro-F1={f1m:.4f}")
        rows.append(pd.DataFrame({"model": name, "group": "ours",
                                  "y_true": y, "y_pred": pred}))
    pd.concat(rows, ignore_index=True).to_parquet(PRED / "oob_ours_finetune.parquet", index=False)
    print(f"saved -> {PRED / 'oob_ours_finetune.parquet'}")


if __name__ == "__main__":
    main()
