"""BNB AI Trading Bot dashboard (Streamlit).

Live mode (default): pulls recent BNB klines from Binance on every refresh,
runs them through indicators + the trained calibrated model, and plots the
prediction graph. Cached mode: reads data/processed CSV instead (offline).
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
from src.strategy.signal import HOLD, generate
from src.utils.config import ROOT, load_config

st.set_page_config(page_title="BNB AI Trading Bot", layout="wide")
cfg = load_config()
REFRESH_S = 60
TICK_S = 2  # live price ticker refresh (24hr ticker endpoint, weight ~2)
LOOKBACK = 500  # candles; ≈8h at 1m, ≈5d at 15m (covers EMA200 warmup)
MODEL_BY_INTERVAL = {"1m": "model_1m.pkl"}  # intervals not listed use the default model


@st.fragment(run_every=TICK_S)
def price_ticker() -> None:
    """Ticking live price — independent of the heavy 60s chart refresh."""
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/24hr",
                         params={"symbol": cfg["symbol"]}, timeout=5)
        r.raise_for_status()
        t = r.json()
        price, chg = float(t["lastPrice"]), float(t["priceChangePercent"])
        st.markdown(f"## {cfg['symbol']}  ${price:,.2f}  "
                    f"({chg:+.2f}% 24h)  🟢 LIVE")
    except Exception as exc:
        st.markdown(f"## {cfg['symbol']}  🔴 price feed unavailable ({exc})")


@st.cache_resource
def _bundle(model_file: str):
    return joblib.load(ROOT / cfg["model"]["dir"] / model_file)


def fetch_live(symbol: str, interval: str, n: int = LOOKBACK) -> pd.DataFrame:
    step = INTERVAL_MS[interval]
    end_ms = int(time.time() * 1000)
    rows = fetch_klines(symbol, interval, end_ms - n * step, end_ms, limit=1000)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def with_predictions(df: pd.DataFrame, bundle) -> pd.DataFrame:
    feat = add_all(df, cfg).dropna().reset_index(drop=True)
    proba = bundle["model"].predict_proba(feat[bundle["features"]])[:, 1]
    feat["buy_prob"] = proba
    feat["sell_prob"] = 1 - proba
    thr = cfg["confidence_threshold"]
    feat["signal"] = [generate(b, s, thr) for b, s in zip(feat["buy_prob"], feat["sell_prob"])]
    return feat


st.title("BNB AI Trading Bot")
price_ticker()  # ticks every 2s; charts below refresh every 60s
mode = st.sidebar.radio("Data source", ["Live (Binance)", "Cached CSV"], index=0)
interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "1h"],
                                index=["1m", "5m", "15m", "1h"].index(cfg["interval"]))
model_file = MODEL_BY_INTERVAL.get(interval, cfg["model"]["file"])
bundle = _bundle(model_file)
freq = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h"}[interval]
st.sidebar.caption(f"Symbol {cfg['symbol']} · {interval} · gate {cfg['confidence_threshold']:.0%}")
st.sidebar.caption(f"Auto-refresh every {REFRESH_S}s · model `{bundle['name']}` ({model_file})")


@st.fragment(run_every=REFRESH_S)
def live_view() -> None:
    t0 = time.time()
    try:
        if mode.startswith("Live"):
            df = clean(fetch_live(cfg["symbol"], interval), freq)
            source = f"LIVE · Binance {cfg['symbol']} {interval}"
        else:
            raise RuntimeError("cached mode selected")
    except Exception as exc:  # offline/API hiccup → fall back to cached file, say so
        src = ROOT / "data" / "processed" / f"{cfg['symbol']}_{interval}.csv"
        raw = ROOT / "data" / "raw" / f"{cfg['symbol']}_{interval}.csv"
        src = src if src.exists() else raw
        if not src.exists():
            st.error(f"No data: live fetch failed ({exc}) and no cached {interval} file. "
                     "Run `python src/data/collector.py` first.")
            st.stop()
        df = clean(pd.read_csv(src), freq).tail(LOOKBACK)
        source = f"CACHED · {src.name} (live failed: {exc})"

    feat = with_predictions(df, bundle)
    last = feat.iloc[-1]
    st.caption(f"{source} · updated {pd.Timestamp.now(tz='UTC'):%H:%M:%S} UTC "
               f"(took {time.time() - t0:.1f}s)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BNB Price", f"${last['close']:.2f}")
    c2.metric("Signal", last["signal"])
    c3.metric("BUY prob", f"{last['buy_prob'] * 100:.1f}%")
    c4.metric("RSI", f"{last['rsi']:.1f}")

    # --- Prediction graph: price with signal markers + BUY probability vs gate
    plot = feat.tail(200).copy()
    plot["buy_marker"] = plot["close"].where(plot["signal"] == "BUY")
    plot["sell_marker"] = plot["close"].where(plot["signal"] == "SELL")
    st.subheader("Price + signals")
    st.line_chart(plot.set_index("timestamp")[["close", "ema20", "buy_marker", "sell_marker"]])
    st.subheader("Predicted BUY probability vs gate")
    plot["gate"] = cfg["confidence_threshold"]
    st.line_chart(plot.set_index("timestamp")[["buy_prob", "gate"]])

    with st.expander("Latest bars"):
        st.dataframe(plot[["timestamp", "close", "rsi", "macd", "buy_prob", "signal"]]
                     .tail(10).iloc[::-1], use_container_width=True)


live_view()

rep_name = "latest_1m.json" if interval == "1m" else "latest.json"
rep = ROOT / "backtests" / "results" / rep_name
if rep.exists():
    with st.expander("Latest backtest (out-of-sample, fees + slippage included)"):
        st.json(rep.read_text())
