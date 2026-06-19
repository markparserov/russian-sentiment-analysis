"""Hand-crafted sentiment features for Russian social-media / review text.

Classical models (TF-IDF + LinearSVC, embeddings + LogReg) can cheaply exploit
structured, interpretable signals that a bag-of-words / frozen-embedding view
misses: punctuation intensity, emoticons, shouting, character elongation, a
compact polarity lexicon and negation. This module computes those signals in a
fully *vectorised* way (pandas str / regex) so it scales to the full 1.37M-row
master as well as the small OOB set, and exposes a single fixed column order so
the exact same matrix is produced at train and inference time.

Everything is derived from the (whitespace-cleaned) ``text`` column, which keeps
original case, punctuation, brackets and emoji - identical preprocessing for
master and OOB, so the features line up across domains.

    from text_features import compute_features, FEATURE_COLS
    feats = compute_features(df["text"]).to_numpy("float32")   # (n, len(FEATURE_COLS))
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --- compact Russian polarity lexicon (stems, matched case-insensitively) -----
# Deliberately small + high-precision; it is an *aggregate* signal for a linear
# model, so a little noise is acceptable.
_POS = [
    "хорош", "отличн", "прекрасн", "замечательн", "люблю", "любим", "нрав",
    "спасибо", "класс", "супер", "рекоменд", "восторг", "удобн", "быстр",
    "вкусн", "радост", "счастл", "шикар", "идеальн", "лучш", "чудесн",
    "приятн", "довол", "восхит", "огонь", "топ", "красот", "молодц",
]
_NEG = [
    "плох", "ужас", "отврат", "ненавиж", "разочаров", "обман", "хамств",
    "груб", "медленн", "кошмар", "отстой", "бесит", "раздраж", "проблем",
    "недостат", "невозможн", "худш", "зря", "сломал", "брак", "жалоб",
    "беспредел", "развод", "впустую", "отказыва", "недовол", "печаль",
]
# matched against a lower-cased copy of the text (cheaper than re.IGNORECASE)
_POS_RE = re.compile("|".join(_POS))
_NEG_RE = re.compile("|".join(_NEG))

# negation particles (word-boundary)
_NEG_PART_RE = re.compile(r"(?:\bне\b|\bнет\b|\bни\b|\bнельзя\b|\bбез\b)")

# emoji sets (single code points cover the common cases)
_EMO_POS = "😀😁😂🤣😊😍🥰😘👍❤💕💖🔥✨😻🙂😉👏🎉"
_EMO_NEG = "😞😔😢😭😡🤬👎💔😠😤🙁☹😩😫🤢🤮"

FEATURE_COLS = [
    "n_chars", "n_words", "mean_word_len",
    "n_exclaim", "excl_ratio", "n_question", "has_question",
    "n_punct", "punct_ratio", "upper_ratio", "caps_words",
    "n_elong",
    "n_close_paren", "n_open_paren", "paren_score",
    "has_multi_close", "has_multi_open",
    "n_emoji_pos", "n_emoji_neg",
    "n_pos_lex", "n_neg_lex", "lex_score",
    "n_negation", "has_negation",
]


def compute_features(texts) -> pd.DataFrame:
    """Vectorised feature matrix; returns a DataFrame with columns == FEATURE_COLS."""
    t = pd.Series(texts, dtype="string").fillna("")
    tl = t.str.lower()  # single lower-cased pass reused by the lexicon/negation counts
    n_chars = t.str.len().astype("float32")
    n_words = t.str.count(r"\S+").astype("float32")
    safe_chars = n_chars.where(n_chars > 0, 1.0)
    safe_words = n_words.where(n_words > 0, 1.0)

    n_exclaim = t.str.count("!").astype("float32")
    n_question = t.str.count(r"\?").astype("float32")
    n_upper = t.str.count(r"[A-ZА-ЯЁ]").astype("float32")
    n_punct = t.str.count(r"[^\w\s]").astype("float32")
    n_close = t.str.count(r"\)").astype("float32")
    n_open = t.str.count(r"\(").astype("float32")

    out = pd.DataFrame(index=t.index)
    out["n_chars"] = n_chars
    out["n_words"] = n_words
    out["mean_word_len"] = (n_chars / safe_words).astype("float32")
    out["n_exclaim"] = n_exclaim
    out["excl_ratio"] = (n_exclaim / (safe_words + 1.0)).astype("float32")
    out["n_question"] = n_question
    out["has_question"] = (n_question > 0).astype("float32")
    out["n_punct"] = n_punct
    out["punct_ratio"] = (n_punct / safe_chars).astype("float32")
    out["upper_ratio"] = (n_upper / safe_chars).clip(0, 1).astype("float32")
    # ALL-CAPS runs of length >= 2 = "shouting" tokens
    out["caps_words"] = t.str.count(r"[A-ZА-ЯЁ]{2,}").astype("float32")
    # char repeated >= 3 times: круутоо / ахаха-style elongation/emphasis
    out["n_elong"] = t.str.count(r"(.)\1{2,}").astype("float32")
    out["n_close_paren"] = n_close
    out["n_open_paren"] = n_open
    # RU bracket emoticons: ")" >> "(" is strongly positive, the reverse negative
    out["paren_score"] = (n_close - n_open).astype("float32")
    out["has_multi_close"] = t.str.contains(r"\){2,}", regex=True).astype("float32")
    out["has_multi_open"] = t.str.contains(r"\({2,}", regex=True).astype("float32")
    out["n_emoji_pos"] = t.str.count(f"[{_EMO_POS}]").astype("float32")
    out["n_emoji_neg"] = t.str.count(f"[{_EMO_NEG}]").astype("float32")
    n_pos = tl.str.count(_POS_RE).astype("float32")
    n_neg = tl.str.count(_NEG_RE).astype("float32")
    out["n_pos_lex"] = n_pos
    out["n_neg_lex"] = n_neg
    out["lex_score"] = (n_pos - n_neg).astype("float32")
    n_neg_part = tl.str.count(_NEG_PART_RE).astype("float32")
    out["n_negation"] = n_neg_part
    out["has_negation"] = (n_neg_part > 0).astype("float32")

    return out[FEATURE_COLS].astype("float32").reset_index(drop=True)
