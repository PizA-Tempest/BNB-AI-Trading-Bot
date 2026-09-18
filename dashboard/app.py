"""BNB real-time price dashboard (Streamlit).

Pure price tracker: live ticker + live chart from Binance public API.
No trading, no signals — just the price, updating on its own.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
import streamlit as st

from src.data.collector import INTERVAL_MS, fetch_klines

st.set_page_config(page_title="BNB Live Price", layout="wide")

SYMBOL = "BNBUSDT"
TICK_S = 2    # price ticker refresh
CHART_S = 10  # chart refresh
LOOKBACK = 300


def ticker_24h(symbol: str = SYMBOL) -> dict:
    r = requests.get("https://api.binance.com/api/v3/ticker/24hr",
                     params={"symbol": symbol}, timeout=5)
    r.raise_for_status()
    return r.json()


def recent_klines(symbol: str, interval: str, n: int = LOOKBACK) -> pd.DataFrame:
    step = INTERVAL_MS[interval]
    end_ms = int(time.time() * 1000)
    rows = fetch_klines(symbol, interval, end_ms - n * step, end_ms, limit=1000)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp")


st.title(f"{SYMBOL} · Live Price")
interval = st.sidebar.selectbox("Chart interval", ["1m", "5m", "15m", "1h"], index=0)
st.sidebar.caption("Source: Binance public API · no keys needed")


@st.fragment(run_every=TICK_S)
def price_ticker() -> None:
    try:
        t = ticker_24h()
        price, chg = float(t["lastPrice"]), float(t["priceChangePercent"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"${price:,.2f}", f"{chg:+.2f}% (24h)")
        c2.metric("24h High", f"${float(t['highPrice']):,.2f}")
        c3.metric("24h Low", f"${float(t['lowPrice']):,.2f}")
        c4.metric("24h Volume", f"{float(t['volume']):,.0f} BNB")
    except Exception as exc:
        st.error(f"🔴 price feed unavailable ({exc})")


@st.fragment(run_every=CHART_S)
def price_chart() -> None:
    try:
        df = recent_klines(SYMBOL, interval)
    except Exception as exc:
        st.error(f"🔴 chart feed unavailable ({exc})")
        return
    st.line_chart(df.set_index("timestamp")[["close"]], height=400)
    st.caption(f"Last {len(df)} × {interval} candles · "
               f"latest close ${df['close'].iloc[-1]:,.2f} at "
               f"{df['timestamp'].iloc[-1]:%H:%M:%S} UTC")
    with st.expander("Latest candles"):
        st.dataframe(df.tail(10).iloc[::-1], use_container_width=True)


price_ticker()
price_chart()
