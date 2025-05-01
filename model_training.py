"""Train LightGBM classifier (no per-symbol z-scaling)."""

from __future__ import annotations
import os
from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from logger_setup import get_logger

from config import MODEL_DIR, MODEL_PATH
from data_ingestion import load_price_data
from feature_engineering import compute_features

log = get_logger(__name__)

FEATURE_COLS = [
    "SMA_5", "SMA_20", "RSI_14", "BB_UPPER", "BB_LOWER",
    "Return_1", "Return_2", "Return_5",
    "ATR_14", "Mom_1", "TOD_sin", "TOD_cos", "VWAP_gap",
]
LOW_Q, HIGH_Q = 0.30, 0.70   # quantile labels


def prepare_dataset():
    X_parts, y_parts = [], []
    for tkr, df in load_price_data().items():
        feats = compute_features(df)
        fut_ret = df["Close"].pct_change().shift(-1).reindex(feats.index)
        lo, hi = fut_ret.quantile(LOW_Q), fut_ret.quantile(HIGH_Q)
        labels = fut_ret.apply(lambda r: 1 if r > hi else (-1 if r < lo else 0))
        mask = labels != 0
        if mask.any():
            X_parts.append(feats.loc[mask, FEATURE_COLS].values)
            y_parts.append(labels[mask].values)
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    log.info("Class distribution: %s", Counter(y))
    return X, y


def train():
    X, y = prepare_dataset()
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        is_unbalance=True,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
        force_col_wise=True,         # ← better for wide data
        reuse_hist=True,             # ← reduces memory use + faster
    )
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_va)
    log.info("Validation accuracy: %.3f", accuracy_score(y_va, y_pred))
    log.info("\n%s", classification_report(y_va, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    log.info("Model saved ➜ %s", MODEL_PATH)


if __name__ == "__main__":
    train()
