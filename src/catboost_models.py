"""GPU CatBoost classical heads that replace the linear baselines.

Two models, both trained on GPU with balanced class weights and **macro-F1**
early stopping, sharing one fit/predict interface so notebook 03 and the OOB
script run identical logic:

* ``cb_text`` - CatBoost over the **raw text** (its own in-model tokenizer builds
  bag-of-words + bigram dictionaries) **plus** the hand-crafted sentiment
  features from :mod:`text_features`. Replaces TF-IDF + LinearSVC.
* ``cb_emb``  - CatBoost over the frozen ``USER-bge-m3`` embeddings (1024 numeric
  dims, no extra features - that scored best out-of-domain). Replaces
  embeddings + LogReg.

CatBoost's native ``embedding_features`` mode hangs on this GPU build, so the
embeddings are fed as plain numeric columns, which trains fine and generalises
better OOB than the linear head.

    from catboost_models import fit_text, predict_text, fit_emb, predict_emb
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score

from text_features import compute_features

LABELS = ["negative", "neutral", "positive"]
SEED = 42

# Compact, "honest" search space tuned over the three knobs that matter most for
# GBDT capacity/regularisation. Kept small on purpose - CatBoost trains in
# seconds on GPU, so a handful of random draws is enough and stays well under a
# minute-scale budget per trial.
SEARCH_SPACE = {
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.08, 0.12],
    "l2_leaf_reg": [1.0, 3.0, 6.0, 9.0],
}
TUNE_KEYS = tuple(SEARCH_SPACE)


def cb_params(**overrides):
    """Standard GPU CatBoost config (macro-F1 early stopping, balanced weights)."""
    params = dict(
        iterations=1500,
        learning_rate=0.06,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        auto_class_weights="Balanced",
        task_type="GPU",
        devices="0",
        od_type="Iter",
        od_wait=80,
        random_seed=SEED,
        verbose=200,
    )
    params.update(overrides)
    return params


def sample_grid(n_trials, seed=SEED):
    """Reproducible random configs; the sensible default is always trial #0."""
    rng = np.random.default_rng(seed)
    grid = [dict(depth=6, learning_rate=0.06, l2_leaf_reg=3.0)]
    seen = {tuple(sorted(grid[0].items()))}
    guard = 0
    while len(grid) < n_trials and guard < 200:
        guard += 1
        cfg = {k: rng.choice(v).item() for k, v in SEARCH_SPACE.items()}
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            grid.append(cfg)
    return grid


def _macro(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", labels=LABELS))


# --------------------------------------------------------------------------- #
#  text + engineered features                                                 #
# --------------------------------------------------------------------------- #
def _text_frame(texts, feats=None) -> pd.DataFrame:
    f = (compute_features(texts) if feats is None else feats).reset_index(drop=True).copy()
    f["text"] = pd.Series(texts, dtype="string").fillna("").reset_index(drop=True)
    return f


def text_pool(texts, y=None, feats=None) -> Pool:
    return Pool(
        _text_frame(texts, feats),
        label=(None if y is None else list(y)),
        text_features=["text"],
    )


def fit_text(texts_tr, y_tr, texts_val, y_val, **overrides) -> CatBoostClassifier:
    model = CatBoostClassifier(**cb_params(**overrides))
    model.fit(text_pool(texts_tr, y_tr), eval_set=text_pool(texts_val, y_val))
    return model


def predict_text(model: CatBoostClassifier, texts) -> np.ndarray:
    return model.predict(text_pool(texts)).ravel()


# --------------------------------------------------------------------------- #
#  embeddings (numeric)                                                       #
# --------------------------------------------------------------------------- #
def fit_emb(emb_tr, y_tr, emb_val, y_val, **overrides) -> CatBoostClassifier:
    model = CatBoostClassifier(**cb_params(**overrides))
    model.fit(np.asarray(emb_tr, np.float32), list(y_tr),
              eval_set=(np.asarray(emb_val, np.float32), list(y_val)))
    return model


def predict_emb(model: CatBoostClassifier, emb) -> np.ndarray:
    return model.predict(np.asarray(emb, np.float32)).ravel()


# --------------------------------------------------------------------------- #
#  quick random hyper-parameter search (reused for the full A/B refits)       #
# --------------------------------------------------------------------------- #
def tune_text(texts_tr, y_tr, texts_val, y_val, n_trials=8, seed=SEED, iterations=1500):
    """Random search for CatBoost(text+feats); scored by macro-F1 on the val set.

    Pools are built once and reused across trials, so each extra config only
    pays for GPU training. Returns (best_cfg, trials)."""
    feats_tr, feats_val = compute_features(texts_tr), compute_features(texts_val)
    pool_tr = text_pool(texts_tr, y_tr, feats=feats_tr)
    pool_val = text_pool(texts_val, y_val, feats=feats_val)
    trials = []
    for cfg in sample_grid(n_trials, seed):
        m = CatBoostClassifier(**cb_params(iterations=iterations, verbose=0, **cfg))
        m.fit(pool_tr, eval_set=pool_val)
        trials.append({**cfg, "trees": int(m.tree_count_),
                       "val_macro_f1": _macro(y_val, m.predict(pool_val).ravel())})
    best = max(trials, key=lambda t: t["val_macro_f1"])
    return best, trials


def tune_emb(emb_tr, y_tr, emb_val, y_val, n_trials=8, seed=SEED, iterations=1500):
    """Random search for CatBoost(emb); scored by macro-F1 on the val set."""
    Xtr, Xva = np.asarray(emb_tr, np.float32), np.asarray(emb_val, np.float32)
    ytr, yva = list(y_tr), list(y_val)
    trials = []
    for cfg in sample_grid(n_trials, seed):
        m = CatBoostClassifier(**cb_params(iterations=iterations, verbose=0, **cfg))
        m.fit(Xtr, ytr, eval_set=(Xva, yva))
        trials.append({**cfg, "trees": int(m.tree_count_),
                       "val_macro_f1": _macro(y_val, m.predict(Xva).ravel())})
    best = max(trials, key=lambda t: t["val_macro_f1"])
    return best, trials
