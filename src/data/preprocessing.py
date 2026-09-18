"""Cleaning pipeline: dedupe, sort, missing-data handling."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

REQUIRED = ["timestamp", "open", "high", "low", "close", "volume"]

INTERVAL_FREQ = {"1m": "1min", "5m": "5min", "15m": "15min",
                 "1h": "1h", "4h": "4h", "1d": "1D"}


def clean(df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")
    df = df.set_index("timestamp").asfreq(freq)
    # Forward-fill small gaps (exchange hiccups), keep longer gaps visible
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].ffill(limit=3)
    df = df.dropna(subset=["close"])
    df["volume"] = df["volume"].fillna(0)
    for c in ["open", "high", "low"]:
        df[c] = df[c].fillna(df["close"])
    # Basic sanity: no negative prices
    df = df[(df["close"] > 0) & (df["volume"] >= 0)]
    return df.reset_index()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="data/raw/BNBUSDT_15m.csv")
    p.add_argument("--out", default="data/processed/BNBUSDT_15m.csv")
    p.add_argument("--interval", default="15m", choices=list(INTERVAL_FREQ))
    a = p.parse_args()
    df = clean(pd.read_csv(a.inp), INTERVAL_FREQ[a.interval])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"cleaned {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
