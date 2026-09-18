"""Evaluate: classification metrics (out-of-sample) kept separate from trading metrics.

Usage: python src/models/evaluate.py [--csv data/processed/BNBUSDT_15m.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.data.preprocessing import clean
from src.features.indicators import FEATURE_COLS, add_all, make_target
from src.utils.config import ROOT, load_config


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/processed/BNBUSDT_15m.csv")
    p.add_argument("--test-frac", type=float, default=0.2)
    a = p.parse_args()

    df = add_all(clean(pd.read_csv(a.csv)), cfg).dropna().reset_index(drop=True)
    f = cfg.get("features", {})
    df["target"] = make_target(df, f.get("target_horizon", 3), f.get("target_threshold", 0.002))
    te = df.iloc[int(len(df) * (1 - a.test_frac)):]  # chronological tail = out-of-sample

    bundle = joblib.load(ROOT / cfg["model"]["dir"] / cfg["model"]["file"])
    proba = bundle["model"].predict_proba(te[FEATURE_COLS])[:, 1]
    pred = (proba >= 0.5).astype(int)
    y = te["target"]
    print(f"model={bundle['name']}  n={len(te)}  out-of-sample=True")
    print(f"accuracy={accuracy_score(y, pred):.4f} precision={precision_score(y, pred, zero_division=0):.4f} "
          f"recall={recall_score(y, pred, zero_division=0):.4f} f1={f1_score(y, pred, zero_division=0):.4f} "
          f"roc_auc={roc_auc_score(y, proba):.4f}")


if __name__ == "__main__":
    main()
