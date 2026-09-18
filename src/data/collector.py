"""Binance public klines collector (no API key needed for klines).

Downloads OHLCV candles and stores CSV in data/raw/.
Falls back to synthetic data when offline (--synthetic).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"
INTERVAL_MS = {
    "5m": 5 * 60_000, "15m": 15 * 60_000, "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000, "1d": 24 * 60 * 60_000,
}


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000) -> list:
    params = {"symbol": symbol, "interval": interval, "startTime": start_ms,
              "endTime": end_ms, "limit": limit}
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def collect(symbol: str, interval: str, days: int, out: Path) -> pd.DataFrame:
    step = INTERVAL_MS[interval]
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60_000
    rows: list = []
    cur = start_ms
    while cur < end_ms:
        chunk = fetch_klines(symbol, interval, cur, min(cur + 1000 * step, end_ms))
        if not chunk:
            break
        rows.extend(chunk)
        cur = chunk[-1][0] + step
        time.sleep(0.2)  # respect rate limits
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def synthetic(symbol: str, interval: str, days: int, out: Path, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    step_min = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}[interval]
    n = max(days * 24 * 60 // step_min, 500)
    rets = rng.normal(0.0002, 0.008, n)
    close = 650 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n,
                                   freq=f"{step_min}min"),
        "open": np.roll(close, 1), "high": close * 1.004,
        "low": close * 0.996, "close": close,
        "volume": rng.uniform(800, 2000, n),
    })
    df.loc[0, "open"] = df.loc[0, "close"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BNBUSDT")
    p.add_argument("--interval", default="15m", choices=list(INTERVAL_MS))
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--out", default="data/raw/BNBUSDT_15m.csv")
    p.add_argument("--synthetic", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    df = synthetic(a.symbol, a.interval, a.days, out) if a.synthetic else collect(a.symbol, a.interval, a.days, out)
    print(f"wrote {len(df)} candles -> {out}")


if __name__ == "__main__":
    main()
