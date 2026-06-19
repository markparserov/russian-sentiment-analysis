"""Fine-tune a sentence encoder with a 3-class sentiment head + hyperparameter search.

Two stages:
  1. SEARCH - random search over a grid of training hyperparameters *and the
     encoder itself* (USER-bge-m3 vs USER2-base). Each trial trains on the
     SEARCH_VARIANT train split (balanced subsample) and is scored by macro-F1
     on that variant's *validation* split. The best config is picked by val F1.
  2. FINAL  - the best config is retrained on Variant A and Variant B (larger
     balanced subsample), models are saved, and predictions on the A/B test
     sets are stored so the models notebook can compare against the baselines.

Search space and budget are configurable via env vars, e.g.:
    FT_N_TRIALS=16 FT_SEARCH_VARIANT=A FT_PER_CLASS_SEARCH=15000

Outputs (data/processed/preds/):
    finetune_search.json     every trial config + val macro-F1 + the best config
    finetune_preds.parquet   columns: train_on, test_on, y_true, y_pred
    finetune_meta.json       best config + final macro-F1 summary
Saved models: models/finetune_A, models/finetune_B

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    /home/zabolonkov/.conda/envs/topic-sentiment/bin/python src/finetune.py
"""

from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PRED = PROC / "preds"
PRED.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}
SEED = 42

# ----------------------------- search space --------------------------------
# The encoder is now one of the searched hyperparameters.
SEARCH_SPACE = {
    "encoder": ["deepvk/USER-bge-m3", "deepvk/USER2-base"],
    "lr": [1e-5, 2e-5, 3e-5, 5e-5],
    "epochs": [1, 2, 3],
    "batch_size": [16, 32],          # per device
    "max_len": [128, 256],
    "weight_decay": [0.01, 0.1],
    "warmup_ratio": [0.06, 0.1],
}
# random-search budget; if >= full grid size, the whole grid is evaluated
N_TRIALS = int(os.environ.get("FT_N_TRIALS", "40"))
SEARCH_VARIANT = os.environ.get("FT_SEARCH_VARIANT", "A")   # tune on this variant
PER_CLASS_SEARCH = int(os.environ.get("FT_PER_CLASS_SEARCH", "15000"))
PER_CLASS_FINAL = int(os.environ.get("FT_PER_CLASS_FINAL", "50000"))

_TOK_CACHE: dict[str, AutoTokenizer] = {}


def get_tok(encoder):
    if encoder not in _TOK_CACHE:
        _TOK_CACHE[encoder] = AutoTokenizer.from_pretrained(encoder)
    return _TOK_CACHE[encoder]


def tok_ds(tok, texts, max_len, labels=None):
    data = {"text": list(texts)}
    if labels is not None:
        data["labels"] = [L2I[l] for l in labels]
    ds = Dataset.from_dict(data)
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len),
                  batched=True, remove_columns=["text"])


def mask_of(df, variant, part):
    """NA-safe boolean mask; split_B is <NA> for rating rows."""
    return (df[f"split_{variant}"] == part).fillna(False).to_numpy(dtype=bool)


def balanced_idx(df, mask, per_class, seed=SEED):
    rng = np.random.default_rng(seed)
    out = []
    sub = df[mask]
    for l in LABELS:
        idx = sub.index[sub["label"] == l].to_numpy()
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        out.append(idx)
    out = np.concatenate(out)
    rng.shuffle(out)
    return out


def make_args(cfg, out_dir):
    return TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=128,
        fp16_full_eval=torch.cuda.is_available(),
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        fp16=torch.cuda.is_available(),
        logging_steps=200,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=4,
        seed=SEED,
    )


def train(df, variant, cfg, per_class, save_dir=None):
    tok = get_tok(cfg["encoder"])
    tr_idx = balanced_idx(df, mask_of(df, variant, "train"), per_class)
    train_ds = tok_ds(tok, df.loc[tr_idx, "text"].to_numpy(), cfg["max_len"],
                      df.loc[tr_idx, "label"].to_numpy())
    conf = AutoConfig.from_pretrained(
        cfg["encoder"], num_labels=3,
        id2label={i: l for l, i in L2I.items()}, label2id=L2I,
    )
    # ModernBERT (USER2-base) torch.compiles its MLP, which breaks under the
    # Trainer's multi-GPU DataParallel (FX tracing) -> force it off (default None
    # is auto-resolved to True at init, so set it explicitly).
    if hasattr(conf, "reference_compile"):
        conf.reference_compile = False
    model = AutoModelForSequenceClassification.from_pretrained(cfg["encoder"], config=conf)
    enc_tag = cfg["encoder"].split("/")[-1]
    args = make_args(cfg, str(PROC / f"_ft_{variant}_{enc_tag}"))
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      data_collator=DataCollatorWithPadding(tok))
    trainer.train()
    if save_dir is not None:
        trainer.save_model(str(save_dir))
        tok.save_pretrained(str(save_dir))
        print(f"[{variant}] saved fine-tuned model -> {save_dir}", flush=True)
    return trainer, tok


def evaluate(trainer, tok, df, variant, part, max_len):
    sub = df[mask_of(df, variant, part)]
    ds = tok_ds(tok, sub["text"].to_numpy(), max_len)
    pred_ids = np.asarray(trainer.predict(ds).predictions).argmax(-1)
    y_pred = np.asarray(LABELS)[pred_ids]
    y_true = sub["label"].to_numpy()
    return y_true, y_pred


def macro(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", labels=LABELS))


def sample_configs(space, n, seed=SEED):
    keys = list(space)
    grid = list(itertools.product(*(space[k] for k in keys)))
    rng = np.random.default_rng(seed)
    if n >= len(grid):
        chosen = grid
    else:
        chosen = [grid[i] for i in rng.choice(len(grid), n, replace=False)]
    return [dict(zip(keys, combo)) for combo in chosen]


def run_search(df):
    configs = sample_configs(SEARCH_SPACE, N_TRIALS)
    print(f"=== SEARCH: {len(configs)} trials on variant {SEARCH_VARIANT} "
          f"(grid size {np.prod([len(v) for v in SEARCH_SPACE.values()])}) ===", flush=True)
    trials = []
    for i, cfg in enumerate(configs, 1):
        t0 = time.time()
        try:
            trainer, tok = train(df, SEARCH_VARIANT, cfg, PER_CLASS_SEARCH)
            yt, yp = evaluate(trainer, tok, df, SEARCH_VARIANT, "val", cfg["max_len"])
            val_f1 = macro(yt, yp)
            status = "ok"
            del trainer
            torch.cuda.empty_cache()
        except Exception as e:  # a bad encoder/config must not abort the whole search
            val_f1, status = float("nan"), f"{type(e).__name__}: {str(e)[:120]}"
            torch.cuda.empty_cache()
        dt = round(time.time() - t0, 1)
        rec = {"trial": i, **cfg, "val_macro_f1": round(val_f1, 4) if val_f1 == val_f1 else None,
               "status": status, "secs": dt}
        trials.append(rec)
        print(f"[{i}/{len(configs)}] {cfg['encoder'].split('/')[-1]:13s} "
              f"lr={cfg['lr']:.0e} ep={cfg['epochs']} bs={cfg['batch_size']} "
              f"ml={cfg['max_len']} wd={cfg['weight_decay']} wu={cfg['warmup_ratio']} "
              f"-> val_f1={rec['val_macro_f1']} ({status}, {dt}s)", flush=True)
        _save_search(trials)   # incremental: survive a later crash

    ok = [t for t in trials if t["val_macro_f1"] is not None]
    best = max(ok, key=lambda t: t["val_macro_f1"])
    best_cfg = {k: best[k] for k in SEARCH_SPACE}
    print(f"\n=== BEST: val_macro_f1={best['val_macro_f1']} | {best_cfg} ===", flush=True)
    _save_search(trials, best)
    return best_cfg, trials


def _save_search(trials, best=None):
    payload = {"search_variant": SEARCH_VARIANT, "per_class_search": PER_CLASS_SEARCH,
               "space": SEARCH_SPACE, "trials": trials}
    if best is not None:
        payload["best"] = best
    (PRED / "finetune_search.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def final_train(df, best_cfg):
    plan = {"A": ["A", "B"], "B": ["B"]}   # train_variant -> test_variants
    rows, summary = [], {}

    def flush():
        pd.concat(rows, ignore_index=True).to_parquet(PRED / "finetune_preds.parquet", index=False)
        (PRED / "finetune_meta.json").write_text(json.dumps({
            "best_config": best_cfg, "per_class_final": PER_CLASS_FINAL,
            "search_variant": SEARCH_VARIANT, "macro_f1": summary,
        }, indent=2, ensure_ascii=False))

    print(f"\n=== FINAL training with best config: {best_cfg} ===", flush=True)
    for train_variant, test_variants in plan.items():
        save_dir = MODELS / f"finetune_{train_variant}"
        trainer, tok = train(df, train_variant, best_cfg, PER_CLASS_FINAL, save_dir=save_dir)
        for tv in test_variants:
            yt, yp = evaluate(trainer, tok, df, tv, "test", best_cfg["max_len"])
            f1m = macro(yt, yp)
            summary[f"{train_variant}->{tv}"] = round(f1m, 4)
            print(f"  {train_variant}->{tv}: macro-F1={f1m:.4f}", flush=True)
            rows.append(pd.DataFrame({"train_on": train_variant, "test_on": tv,
                                      "y_true": yt, "y_pred": yp}))
            flush()
        del trainer
        torch.cuda.empty_cache()
    print("saved ->", PRED / "finetune_preds.parquet", flush=True)
    print("summary:", summary, flush=True)


def main():
    df = pd.read_parquet(PROC / "master.parquet",
                         columns=["text", "label", "split_A", "split_B"])
    print(f"devices: {torch.cuda.device_count()} | rows: {len(df):,}", flush=True)
    best_cfg, _ = run_search(df)
    final_train(df, best_cfg)


if __name__ == "__main__":
    main()
