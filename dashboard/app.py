"""BNB trading assistant (Streamlit).

Top: real-time price tracker (Binance public API, no keys).
Below: trading assistant — live model signal + suggested size/stop/take
to help you decide. Educational, not financial advice.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import requests
import streamlit as st

from src.data.collector import INTERVAL_MS, fetch_klines
from src.data.preprocessing import clean
from src.features.indicators import add_all
from src.risk.manager import RiskManager
from src.strategy.signal import generate
from src.utils.config import ROOT, load_config

st.set_page_config(page_title="BNB Live Price", layout="wide")

SYMBOL = "BNBUSDT"
TICK_S = 2    # price ticker refresh
CHART_S = 10  # chart refresh
ASSIST_S = 60  # assistant (model + indicators) refresh — heavier compute
LOOKBACK = 300
MODEL_BY_INTERVAL = {"1m": "model_1m.pkl"}
cfg = load_config()


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

st.divider()
st.header("Trading assistant")
st.caption("Experimental helper — not financial advice. The current model has a weak "
           "edge (out-of-sample AUC ≈ 0.60), so treat its signal as one input, not a decision.")


@st.cache_resource
def _bundle(model_file: str):
    return joblib.load(ROOT / cfg["model"]["dir"] / model_file)


@st.fragment(run_every=ASSIST_S)
def assistant() -> None:
    balance = st.number_input("Account balance (USDT)", min_value=10.0, value=1000.0, step=50.0)
    risk_pct = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1) / 100
    try:
        df = clean(recent_klines(SYMBOL, interval, n=500),
                   {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h"}[interval])
    except Exception as exc:
        st.error(f"🔴 assistant feed unavailable ({exc})")
        return
    model_file = MODEL_BY_INTERVAL.get(interval, cfg["model"]["file"])
    bundle = _bundle(model_file)
    feat = add_all(df, cfg).dropna().reset_index(drop=True)
    if feat.empty:
        st.warning("Not enough data to score yet.")
        return
    proba = bundle["model"].predict_proba(feat[bundle["features"]])[:, 1]
    last = feat.iloc[-1]
    buy_p, sig = float(proba[-1]), generate(float(proba[-1]), float(1 - proba[-1]),
                                            cfg["confidence_threshold"])
    price = float(last["close"])
    risk = RiskManager.from_config(cfg)
    risk.balance = risk.day_start_balance = risk.peak = balance
    risk.max_risk_per_trade = risk_pct
    qty = risk.position_size(price)
    if sig == "BUY":
        sl, tp = risk.stop_take(price, "BUY")
    elif sig == "SELL":
        sl, tp = risk.stop_take(price, "SELL")
    else:
        sl = tp = None

    c1, c2, c3 = st.columns(3)
    c1.metric("Signal", sig, f"BUY prob {buy_p * 100:.1f}% · gate {cfg['confidence_threshold']:.0%}")
    c2.metric("RSI / MACD", f"{last['rsi']:.1f} / {last['macd']:+.2f}",
              f"EMA20 ${last['ema20']:,.2f}")
    if sig == "HOLD":
        c3.metric("Suggestion", "Wait", "no high-confidence setup")
    else:
        c3.metric("Suggested size", f"{qty:.4f} BNB",
                  f"≈ ${qty * price:,.2f} · stop ${sl:,.2f} · take ${tp:,.2f}")
    st.caption(f"Model `{bundle['name']}` ({model_file}) · {interval} · "
               f"risk ${balance * risk_pct:,.2f}/trade · paper-trade before risking real funds.")


assistant()
