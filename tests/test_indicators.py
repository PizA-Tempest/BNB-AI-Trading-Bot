import pandas as pd

from src.features.indicators import add_all


def test_indicators_present_and_finite():
    n = 250
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "open": 650.0, "high": 655.0, "low": 648.0,
        "close": [650 + i * 0.1 for i in range(n)],
        "volume": 1200.0,
    })
    out = add_all(df).dropna()
    for c in ["rsi", "macd", "ema20", "ema50", "atr", "volatility"]:
        assert c in out.columns
        assert out[c].notna().all()
