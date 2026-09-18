"""Vectorized backtest with fees + slippage. Reports model vs trading metrics separately.

Usage: python -m backtests.run [--csv data/processed/BNBUSDT_15m.csv]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.data.preprocessing import clean
from src.features.indicators import add_all
from src.risk.manager import RiskManager
from src.strategy.signal import HOLD
from src.utils.config import ROOT, load_config


def max_drawdown(equity: pd.Series) -> float:
    return float(((equity.cummax() - equity) / equity.cummax()).max() or 0.0)


def sharpe(returns: pd.Series) -> float:
    return float(returns.mean() / returns.std() * np.sqrt(365 * 96)) if returns.std() else 0.0


def run(df: pd.DataFrame, proba: np.ndarray, cfg: dict):
    thr = cfg["confidence_threshold"]
    fee, slip = cfg["backtest"]["fee_pct"], cfg["backtest"]["slippage_pct"]
    risk = RiskManager.from_config(cfg)
    bal, pos, entry = risk.balance, None, 0.0
    equity, trades = [], []
    signals = np.where(proba >= thr, "BUY", np.where((1 - proba) >= thr, "SELL", HOLD))

    for i, (ts, row, sig) in enumerate(zip(df["timestamp"], df.to_dict("records"), signals)):
        price = row["close"]
        risk.on_bar()
        if pos is not None:  # manage open position: stop / take / opposite signal
            sl, tp = risk.stop_take(entry, pos["side"])
            exit_px = None
            if pos["side"] == "BUY" and (price <= sl or price >= tp):
                exit_px = price
            elif pos["side"] == "SELL" and (price >= sl or price <= tp):
                exit_px = price
            elif (pos["side"] == "BUY" and sig == "SELL") or (pos["side"] == "SELL" and sig == "BUY"):
                exit_px = price
            if exit_px is not None:
                gross = (exit_px - entry) * pos["qty"] if pos["side"] == "BUY" else (entry - exit_px) * pos["qty"]
                net = gross - (entry * pos["qty"] + exit_px * pos["qty"]) * fee - exit_px * pos["qty"] * slip
                bal += net
                trades.append({"entry": entry, "exit": float(exit_px), "side": pos["side"], "pnl": float(net)})
                risk.on_close(bal)
                pos = None
        if pos is None and sig in ("BUY", "SELL"):
            ok, _ = risk.can_open()
            if ok:
                qty = risk.position_size(price)
                if qty > 0:
                    cost = price * qty
                    bal -= cost * (fee + slip)  # entry cost
                    entry = price
                    pos = {"side": sig, "qty": qty}
                    risk.on_fill()
        equity.append(bal + (0 if pos is None else
                             ((price - entry) * pos["qty"] if pos["side"] == "BUY" else (entry - price) * pos["qty"])))
    return pd.Series(equity, index=df["timestamp"]), trades, signals


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/processed/BNBUSDT_15m.csv")
    a = p.parse_args()

    df = add_all(clean(pd.read_csv(a.csv)), cfg).dropna().reset_index(drop=True)
    te = df.iloc[int(len(df) * 0.8):].reset_index(drop=True)  # out-of-sample tail
    bundle = joblib.load(ROOT / cfg["model"]["dir"] / cfg["model"]["file"])
    proba = bundle["model"].predict_proba(te[bundle["features"]])[:, 1]

    equity, trades, signals = run(te, proba, cfg)
    wins = [t for t in trades if t["pnl"] > 0]
    gross_w = sum(t["pnl"] for t in wins)
    gross_l = -sum(t["pnl"] for t in trades if t["pnl"] <= 0) or 1e-9
    rets = equity.pct_change().fillna(0)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": bundle["name"], "out_of_sample": cfg["backtest"]["out_of_sample"],
        "bars": len(te), "signals": int((signals != HOLD).sum()), "trades": len(trades),
        "win_rate": float(len(wins) / len(trades)) if trades else 0.0,
        "profit_factor": float(gross_w / gross_l),
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe(rets),
        "net_return": float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) else 0.0,
        "fees_included": True, "slippage_included": True,
    }
    out = ROOT / "backtests" / "results" / "latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("==================================\n       BNB BACKTEST REPORT\n==================================")
    for k, v in report.items():
        print(f"{k:18s}: {v}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
