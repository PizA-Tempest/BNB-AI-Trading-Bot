"""Load trained model → BUY probability for latest bar(s)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import pandas as pd

from src.features.indicators import add_all
from src.utils.config import ROOT, load_config

_cache: dict = {}


def load_bundle():
    if "bundle" not in _cache:
        cfg = load_config()
        _cache["bundle"] = joblib.load(ROOT / cfg["model"]["dir"] / cfg["model"]["file"])
        _cache["cfg"] = cfg
    return _cache["bundle"], _cache["cfg"]


def predict_proba(df: pd.DataFrame) -> pd.Series:
    bundle, cfg = load_bundle()
    feat = add_all(df, cfg).dropna()
    proba = bundle["model"].predict_proba(feat[bundle["features"]])[:, 1]
    return pd.Series(proba, index=feat.index)
