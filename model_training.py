"""Train Random-Forest with quantile labels & tqdm progress bar."""
from __future__ import annotations
import os
from collections import Counter

import joblib, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from config import MODEL_DIR, MODEL_PATH
from data_ingestion import load_price_data
from feature_engineering import compute_features
from lightgbm import LGBMClassifier

FEATURE_COLS = [
    "SMA_5","SMA_20","RSI_14","BB_UPPER","BB_LOWER",
    "Return_1","Return_2","Return_5",
    "ATR_14","Mom_1","TOD_sin","TOD_cos","VWAP_gap",
]
LOW_Q, HIGH_Q = 0.30, 0.70   # label quantiles

def prepare_dataset():
    price_data = load_price_data()
    X_parts, y_parts = [], []
    for tkr, df in tqdm(price_data.items(), desc="Feature generation", unit="sym"):
        feats = compute_features(df)
        fut_ret = df["Close"].pct_change().shift(-1).reindex(feats.index)
        lo, hi = fut_ret.quantile(LOW_Q), fut_ret.quantile(HIGH_Q)
        labels = fut_ret.apply(lambda r: 1 if r>hi else (-1 if r<lo else 0))
        m = labels != 0
        if m.any():
            X_parts.append(feats.loc[m, FEATURE_COLS].values)
            y_parts.append(labels[m].values)
    X, y = np.vstack(X_parts), np.concatenate(y_parts)
    print("Class distribution:", Counter(y))
    return X, y

def train():
    X, y = prepare_dataset()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary',
        is_unbalance=True,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    print("\nFitting Random Forest …"); clf.fit(Xtr, ytr)
    ypred = clf.predict(Xva)
    print("\nValidation accuracy:", accuracy_score(yva, ypred))
    print(classification_report(yva, ypred))
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved ➜ {MODEL_PATH}")

if __name__ == "__main__":
    train()
