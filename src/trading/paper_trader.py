"""Paper trader: real-time klines, simulated fills. Never touches real orders.

Usage: python src/trading/paper_trader.py [--once]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import pandas as pd

from src.data.collector import collect
from src.data.preprocessing import clean
from src.features.indicators import add_all
from src.models.predict import load_bundle
from src.risk.manager import RiskManager
from src.strategy.signal import HOLD, generate
from src.utils.config import ROOT, load_config


def latest_frame(cfg: dict, n_days: int = 30) -> pd.DataFrame:
    tmp = Path("data/raw/_paper_tmp.csv")
    try:
        collect(cfg["symbol"], cfg["interval"], n_days, tmp)
        df = clean(pd.read_csv(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    return df


def step(cfg: dict, risk: RiskManager) -> dict:
    bundle, _ = load_bundle()
    df = add_all(latest_frame(cfg), cfg).dropna()
    if df.empty:
        return {"signal": HOLD, "price": None}
    proba = float(bundle["model"].predict_proba(df[bundle["features"]].iloc[[-1]])[0][1])
    price = float(df["close"].iloc[-1])
    signal = generate(proba, 1 - proba, cfg["confidence_threshold"])
    ok, reason = risk.can_open()
    action = signal if (ok and signal != HOLD) else HOLD
    if action != HOLD:
        qty = risk.position_size(price)
        risk.on_fill()
        return {"signal": signal, "action": action, "price": price,
                "buy_prob": proba, "qty": qty, "balance": risk.balance}
    risk.on_bar()
    return {"signal": signal, "action": HOLD, "price": price, "buy_prob": proba,
            "reason": reason, "balance": risk.balance}


def main() -> None:
    cfg = load_config()
    if cfg.get("live_trading"):
        raise SystemExit("Refusing: LIVE_TRADING=true in a paper trader. Use live_trader.py explicitly.")
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--poll", type=int, default=int(cfg.get("trading", {}).get("poll_seconds", 60)))
    a = p.parse_args()
    risk = RiskManager.from_config(cfg)
    # Ensure a model exists; train on synthetic data as last resort for first run
    try:
        load_bundle()
    except FileNotFoundError:
        print("no trained model found — train first: python src/models/train.py")
        raise SystemExit(1)
    while True:
        print(step(cfg, risk))
        if a.once:
            break
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
