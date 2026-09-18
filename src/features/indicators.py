"""Technical indicators in pure pandas (no pandas-ta dependency).

All functions take/return a DataFrame with OHLCV columns.
`add_all` is the single entry point used by training/backtests/live.
"""
from __future__ import annotations

import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bollinger(close: pd.Series, period: int = 20, std: float = 2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return mid + std * sd, mid, mid - std * sd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


FEATURE_COLS = [
    "ema20", "ema50", "ema200", "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "bb_width", "atr", "vol_change",
    "momentum", "roc", "volatility",
]


def add_all(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    f = (cfg or {}).get("features", {}) if isinstance(cfg, dict) else {}
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    close, vol = df["close"], df["volume"]

    df["ema20"] = ema(close, f.get("ema_fast", 20))
    df["ema50"] = ema(close, f.get("ema_mid", 50))
    df["ema200"] = ema(close, f.get("ema_slow", 200))
    df["rsi"] = rsi(close, f.get("rsi_period", 14))
    m, sig, hist = macd(close, f.get("macd_fast", 12), f.get("macd_slow", 26), f.get("macd_signal", 9))
    df["macd"], df["macd_signal"], df["macd_hist"] = m, sig, hist
    up, mid, lo = bollinger(close, f.get("bb_period", 20), f.get("bb_std", 2.0))
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = up, mid, lo
    df["bb_width"] = (up - lo) / mid
    df["atr"] = atr(df, f.get("atr_period", 14))
    df["vol_change"] = vol.pct_change().fillna(0)
    df["momentum"] = close - close.shift(f.get("momentum_period", 10))
    df["roc"] = close.pct_change(f.get("roc_period", 12)) * 100
    df["volatility"] = close.pct_change().rolling(f.get("volatility_window", 20)).std()
    return df


def make_target(df: pd.DataFrame, horizon: int = 3, threshold: float = 0.002) -> pd.Series:
    """1 if future return over `horizon` bars exceeds `threshold`, else 0."""
    future = df["close"].shift(-horizon)
    return (((future - df["close"]) / df["close"]) > threshold).astype(int)
