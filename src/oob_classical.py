"""Retrain our classical models on Variant A, SAVE them, and predict on the OOB set.

Reference linear baselines (kept for comparison):
- TF-IDF + LinearSVM  -> models/tfidf_svc_A.joblib
- Embeddings + LogReg (tuned C, cuML on full A) -> exported as a portable CPU
  sklearn LogisticRegression in models/emb_logreg_A.joblib

GPU CatBoost heads (the upgraded classical models), trained on BOTH variants
A and B for comparability with the fine-tuned encoder (A)/(B):
- CatBoost(text+feats): raw text (native tokenizer) + hand-crafted sentiment
  features -> models/catboost_text_{A,B}.cbm
- CatBoost(emb): frozen USER-bge-m3 embeddings as numeric features
  -> models/catboost_emb_{A,B}.cbm

Both CatBoost heads get a small random hyper-parameter search (depth /
learning_rate / l2_leaf_reg) on an A-train subsample; the best config is reused
for the full A and B refits and persisted to preds/catboost_search.json.

Writes OOB predictions to data/processed/preds/oob_ours_classical.parquet.

    CUDA_VISIBLE_DEVICES=11 FT_N_TRIALS=8 \
    /home/zabolonkov/.conda/envs/topic-sentiment/bin/python src/oob_classical.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from catboost_models import fit_emb, fit_text, predict_emb, predict_text, tune_emb, tune_text

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"; MODELS.mkdir(parents=True, exist_ok=True)
PRED = PROC / "preds"; PRED.mkdir(parents=True, exist_ok=True)

LABELS = ["negative", "neutral", "positive"]
SEED = 42
MAX_TFIDF_TRAIN = 400_000   # same as notebook 03
C_EMB = 29.15               # tuned by Optuna in notebook 03
CB_TRIALS = int(os.environ.get("FT_N_TRIALS", 8))   # CatBoost search budget
CB_SEARCH_TRAIN = 120_000   # A-train subsample used only for the search
CB_SEARCH_VAL = 25_000      # A-val subsample used to score the search


def subsample(mask, n, seed=SEED):
    idx = np.where(mask)[0]
    if len(idx) > n:
        idx = np.random.default_rng(seed).choice(idx, n, replace=False)
    return np.sort(idx)


def main() -> None:
    df = pd.read_parquet(PROC / "master.parquet", columns=["text", "label", "split_A", "split_B"])
    oob = pd.read_parquet(PROC / "oob_test.parquet")
    oob_emb = np.load(PROC / "oob_emb_f16.npy").astype(np.float32)
    y_oob = oob["label"].to_numpy()
    trA = (df["split_A"] == "train").to_numpy()
    vaA = (df["split_A"] == "val").to_numpy()
    texts_oob = oob["text"].to_numpy()
    rows = []

    def mask(variant, part):
        return (df[f"split_{variant}"] == part).fillna(False).to_numpy(dtype=bool)

    # ---- 1. TF-IDF + LinearSVC ----
    tr = subsample(trA, MAX_TFIDF_TRAIN)
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=120_000, sublinear_tf=True),
        LinearSVC(C=1.0, class_weight="balanced"),
    ).fit(df["text"].to_numpy()[tr], df["label"].to_numpy()[tr])
    joblib.dump(pipe, MODELS / "tfidf_svc_A.joblib")
    pred = pipe.predict(oob["text"].to_numpy())
    rows.append(pd.DataFrame({"model": "TFIDF+LinSVC", "group": "ours", "y_true": y_oob, "y_pred": pred}))
    print(f"TFIDF+LinSVC  OOB macro-F1={f1_score(y_oob, pred, average='macro', labels=LABELS):.4f}")

    # ---- 2. Embeddings + LogReg (tuned C) ----
    emb = np.load(PROC / "embeddings_full_f16.npy")
    Xtr = emb[trA].astype(np.float32)
    ytr = df["label"].to_numpy()[trA]
    try:
        from cuml.linear_model import LogisticRegression as cuLR
        code = pd.Categorical(ytr, categories=LABELS).codes.astype("float32")
        m = cuLR(C=C_EMB, max_iter=1000, class_weight="balanced")
        m.fit(Xtr, code)
        W = np.asarray(getattr(m.coef_, "get", lambda: m.coef_)(), dtype=np.float64)
        b = np.asarray(getattr(m.intercept_, "get", lambda: m.intercept_)(), dtype=np.float64).ravel()
        p = np.asarray(getattr(m.predict(oob_emb), "get", lambda: m.predict(oob_emb))())
        pred = np.asarray(LABELS)[np.rint(p).astype(int)]
        backend = "cuML"
    except Exception as e:
        print("cuML unavailable, sklearn fallback:", type(e).__name__)
        m = LogisticRegression(C=C_EMB, class_weight="balanced", max_iter=1000, n_jobs=-1).fit(Xtr, ytr)
        W, b = m.coef_, m.intercept_
        pred = m.predict(oob_emb)
        backend = "sklearn"

    # export portable CPU sklearn head from learned weights
    coef = W if W.shape == (len(LABELS), Xtr.shape[1]) else W.T
    sk = LogisticRegression(C=C_EMB, class_weight="balanced")
    sk.classes_ = np.array(LABELS)
    sk.coef_ = coef
    sk.intercept_ = b
    sk.n_features_in_ = Xtr.shape[1]
    joblib.dump(sk, MODELS / "emb_logreg_A.joblib")
    agree = float((sk.predict(oob_emb) == pred).mean())
    print(f"Emb+LogReg backend={backend} | portable-head agreement on OOB={agree:.4f}")
    rows.append(pd.DataFrame({"model": "Emb+LogReg(tuned)", "group": "ours", "y_true": y_oob, "y_pred": pred}))
    print(f"Emb+LogReg(tuned)  OOB macro-F1={f1_score(y_oob, pred, average='macro', labels=LABELS):.4f}")

    texts = df["text"].to_numpy()
    y_all = df["label"].to_numpy()

    # ---- 3. Hyper-parameter search for both CatBoost heads (A-subsample) ----
    sa = subsample(trA, CB_SEARCH_TRAIN)
    sv = subsample(vaA, CB_SEARCH_VAL, seed=SEED + 1)
    t0 = time.time()
    best_text, trials_text = tune_text(texts[sa], y_all[sa], texts[sv], y_all[sv], n_trials=CB_TRIALS)
    best_emb, trials_emb = tune_emb(emb[sa], y_all[sa], emb[sv], y_all[sv], n_trials=CB_TRIALS)
    text_cfg = {k: best_text[k] for k in ("depth", "learning_rate", "l2_leaf_reg")}
    emb_cfg = {k: best_emb[k] for k in ("depth", "learning_rate", "l2_leaf_reg")}
    (PRED / "catboost_search.json").write_text(json.dumps(
        {"n_trials": CB_TRIALS, "search_train": int(len(sa)), "search_val": int(len(sv)),
         "text": {"best": best_text, "trials": trials_text},
         "emb": {"best": best_emb, "trials": trials_emb}}, ensure_ascii=False, indent=2))
    print(f"search ({CB_TRIALS} trials, {time.time()-t0:.0f}s):  text best={text_cfg} "
          f"(val={best_text['val_macro_f1']:.4f}) | emb best={emb_cfg} (val={best_emb['val_macro_f1']:.4f})")

    # ---- 4. GPU CatBoost trained on BOTH A and B (tuned config) ----
    for v in ("A", "B"):
        trv, vav = mask(v, "train"), mask(v, "val")

        t0 = time.time()
        cb_text = fit_text(texts[trv], y_all[trv], texts[vav], y_all[vav], **text_cfg)
        cb_text.save_model(str(MODELS / f"catboost_text_{v}.cbm"))
        pred = predict_text(cb_text, texts_oob)
        rows.append(pd.DataFrame({"model": f"CatBoost(text+feats) ({v})", "group": "ours",
                                  "y_true": y_oob, "y_pred": pred}))
        print(f"CatBoost(text+feats) ({v})  trees={cb_text.tree_count_}  OOB macro-F1="
              f"{f1_score(y_oob, pred, average='macro', labels=LABELS):.4f}  ({time.time()-t0:.0f}s)")

        t0 = time.time()
        cb_emb = fit_emb(emb[trv], y_all[trv], emb[vav], y_all[vav], **emb_cfg)
        cb_emb.save_model(str(MODELS / f"catboost_emb_{v}.cbm"))
        pred = predict_emb(cb_emb, oob_emb)
        rows.append(pd.DataFrame({"model": f"CatBoost(emb) ({v})", "group": "ours",
                                  "y_true": y_oob, "y_pred": pred}))
        print(f"CatBoost(emb) ({v})  trees={cb_emb.tree_count_}  OOB macro-F1="
              f"{f1_score(y_oob, pred, average='macro', labels=LABELS):.4f}  ({time.time()-t0:.0f}s)")

    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(PRED / "oob_ours_classical.parquet", index=False)
    print(f"\nsaved -> {PRED / 'oob_ours_classical.parquet'}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
