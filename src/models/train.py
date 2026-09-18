"""Train: chronological split → compare baseline/LogReg/RF/(XGBoost)/(LightGBM) → save best.

Never shuffles time series. Saves models/trained/model.pkl + metrics.json.
Usage: python src/models/train.py [--csv data/processed/BNBUSDT_15m.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score

from src.data.preprocessing import clean
from src.features.indicators import FEATURE_COLS, add_all, make_target
from src.utils.config import ROOT, load_config


def chrono_split(df: pd.DataFrame, train=0.6, val=0.2):
    n = len(df)
    i, j = int(n * train), int(n * (train + val))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]


def candidates():
    models = {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                n_jobs=-1, random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(n_estimators=400, max_depth=5,
                                          learning_rate=0.05, subsample=0.8,
                                          colsample_bytree=0.8, n_jobs=-1,
                                          eval_metric="logloss")
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(n_estimators=400, num_leaves=63,
                                            learning_rate=0.05, verbose=-1)
    except ImportError:
        pass
    return models


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/processed/BNBUSDT_15m.csv")
    a = p.parse_args()

    raw = pd.read_csv(a.csv)
    df = add_all(clean(raw), cfg).dropna().reset_index(drop=True)
    f = cfg.get("features", {})
    df["target"] = make_target(df, f.get("target_horizon", 3), f.get("target_threshold", 0.002))

    tr, va, te = chrono_split(df)
    X_tr, y_tr = tr[FEATURE_COLS], tr["target"]
    X_va, y_va = va[FEATURE_COLS], va["target"]
    X_te, y_te = te[FEATURE_COLS], te["target"]

    results, fitted = {}, {}
    for name, m in candidates().items():
        # Clone refit on the chronological train slice only (never touches
        # validation/test). Probability calibration (Platt/isotonic on the
        # validation slice) is roadmap Phase 3 — raw predict_proba for now.
        cal = clone(m)
        cal.fit(X_tr, y_tr)
        _val_proba = cal.predict_proba(X_va)[:, 1]  # validation slice kept for future calibration
        pred = cal.predict(X_te)
        proba = cal.predict_proba(X_te)[:, 1]
        results[name] = {
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
            "positives": int(pred.sum()),
            "mean_buy_prob": float(proba.mean()),
        }
        fitted[name] = cal
        print(f"{name:13s} precision={results[name]['precision']:.3f} f1={results[name]['f1']:.3f}")

    best = max(results, key=lambda k: (results[k]["f1"], results[k]["precision"]))
    out_dir = ROOT / cfg["model"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted[best], "features": FEATURE_COLS, "name": best},
                out_dir / cfg["model"]["file"])
    (out_dir / cfg["model"]["metrics_file"]).write_text(json.dumps(
        {"best": best, "results": results,
         "split": {"train": len(tr), "val": len(va), "test": len(te)}}, indent=2))
    print(f"saved best={best} -> {out_dir / cfg['model']['file']}")


if __name__ == "__main__":
    main()
