"""Evaluate 12 external pretrained sentiment models on the OOB set.

Three families:
  * cls    - sequence-classification heads. We read each model's OWN id2label and
             normalise the predicted label to {negative, neutral, positive}
             (handles 5-class and reordered schemes automatically).
  * nli    - NLI checkpoints used via the zero-shot-classification pipeline with
             Russian candidate labels.
  * geracl - deepvk GeRaCl zero-shot classifier (custom `geracl` package).

Writes data/processed/preds/oob_external.parquet  [model, group, y_true, y_pred].
Each model is wrapped in try/except and results are flushed incrementally.

    CUDA_VISIBLE_DEVICES=0 \
    /home/zabolonkov/.conda/envs/topic-sentiment/bin/python src/oob_external.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PRED = PROC / "preds"
OUT = PRED / "oob_external.parquet"
LABELS = ["negative", "neutral", "positive"]
MAXLEN = 256
BS = 64

# name, hf_id, kind
REGISTRY = [
    ("cardiff-xlmr",        "cardiffnlp/twitter-xlm-roberta-base-sentiment", "cls"),
    ("clapAI-roberta-large","clapAI/roberta-large-multilingual-sentiment",   "cls"),
    ("seara-rubert-base",   "seara/rubert-base-cased-russian-sentiment",     "cls"),
    ("seara-rubert-tiny2",  "seara/rubert-tiny2-russian-sentiment",          "cls"),
    ("blanchefort-rubert",  "blanchefort/rubert-base-cased-sentiment",       "cls"),
    ("cointegrated-tiny",   "cointegrated/rubert-tiny-sentiment-balanced",   "cls"),
    ("tabularisai-distil",  "tabularisai/multilingual-sentiment-analysis",   "cls"),
    ("clapAI-modernbert-lg","clapAI/modernBERT-large-multilingual-sentiment", "cls"),
    ("clapAI-modernbert-bs","clapAI/modernBERT-base-multilingual-sentiment",  "cls"),
    ("mdeberta-xnli-2mil7", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "nli"),
    ("mdeberta-mnli-xnli",  "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",       "nli"),
    ("geracl-user2",        "deepvk/GeRaCl-USER2-base",                      "geracl"),
]

ZS_CANDS = ["негативный", "нейтральный", "позитивный"]
ZS_TEMPLATE = "Тональность этого текста: {}."
RU2LAB = {"негативный": "negative", "нейтральный": "neutral", "позитивный": "positive"}


def to3(raw: str):
    """Normalise any model's raw label string to negative/neutral/positive."""
    r = str(raw).strip().lower()
    if "neg" in r or "негатив" in r:
        return "negative"
    if "pos" in r or "позитив" in r:
        return "positive"
    if "neu" in r or "нейтрал" in r:
        return "neutral"
    return None


@torch.no_grad()
def run_cls(hf_id, texts):
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id).to("cuda").eval().half()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    raw2lab = {i: to3(lab) for i, lab in id2label.items()}
    assert all(v is not None for v in raw2lab.values()), f"unmapped labels: {id2label}"
    out = []
    for i in range(0, len(texts), BS):
        b = tok(texts[i:i + BS], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
        ids = model(**b).logits.argmax(-1).cpu().numpy()
        out.append(ids)
    ids = np.concatenate(out)
    return np.array([raw2lab[int(i)] for i in ids])


def run_nli(hf_id, texts):
    clf = pipeline("zero-shot-classification", model=hf_id, tokenizer=hf_id,
                   device=0, torch_dtype=torch.float16)
    preds = []
    res = clf(list(texts), candidate_labels=ZS_CANDS, hypothesis_template=ZS_TEMPLATE,
              multi_label=False, batch_size=32)
    if isinstance(res, dict):
        res = [res]
    for r in res:
        preds.append(RU2LAB[r["labels"][0]])
    return np.array(preds)


def run_geracl(hf_id, texts):
    from geracl import GeraclHF, ZeroShotClassificationPipeline
    model = GeraclHF.from_pretrained(hf_id).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(hf_id)
    pipe = ZeroShotClassificationPipeline(model, tok, device="cuda")
    # pass a flat shared label list -> same_labels=True path (robust to partial batches)
    idxs = pipe(list(texts), ZS_CANDS, batch_size=32)
    return np.array([RU2LAB[ZS_CANDS[int(i)]] for i in idxs])


def main() -> None:
    oob = pd.read_parquet(PROC / "oob_test.parquet")
    texts = oob["text_raw"].tolist()
    y = oob["label"].to_numpy()
    rows = []
    for name, hf_id, kind in REGISTRY:
        try:
            if kind == "cls":
                pred = run_cls(hf_id, texts)
            elif kind == "nli":
                pred = run_nli(hf_id, texts)
            else:
                pred = run_geracl(hf_id, texts)
            f1m = f1_score(y, pred, average="macro", labels=LABELS)
            acc = accuracy_score(y, pred)
            group = "external" if kind == "cls" else "zero-shot"
            rows.append(pd.DataFrame({"model": name, "group": group, "y_true": y, "y_pred": pred}))
            print(f"[OK] {name:22s} ({kind:6s}) macro-F1={f1m:.4f} acc={acc:.4f}", flush=True)
            pd.concat(rows, ignore_index=True).to_parquet(OUT, index=False)
        except Exception as e:
            print(f"[FAIL] {name:22s} ({kind}): {type(e).__name__}: {str(e)[:200]}", flush=True)
        torch.cuda.empty_cache()
    print(f"\nsaved -> {OUT}  ({sum(len(r) for r in rows)} rows, {len(rows)} models OK)")


if __name__ == "__main__":
    main()
