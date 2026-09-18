"""BNB AI Trading Bot dashboard (Streamlit)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.data.preprocessing import clean
from src.features.indicators import add_all
from src.strategy.signal import generate
from src.utils.config import ROOT, load_config

st.set_page_config(page_title="BNB AI Trading Bot", layout="wide")
cfg = load_config()
thr = cfg["confidence_threshold"]

st.title("BNB AI Trading Bot")

csv = ROOT / "data" / "processed" / f"{cfg['symbol']}_{cfg['interval']}.csv"
raw = ROOT / "data" / "raw" / f"{cfg['symbol']}_{cfg['interval']}.csv"
src = csv if csv.exists() else raw

if not src.exists():
    st.warning("No data yet. Run: `python src/data/collector.py --synthetic` then train the model.")
    st.stop()

df = add_all(clean(pd.read_csv(src)), cfg).dropna()
last = df.iloc[-1]

buy_prob = None
try:
    from src.models.predict import predict_proba
    buy_prob = float(predict_proba(clean(pd.read_csv(src))).iloc[-1])
except FileNotFoundError:
    st.info("No trained model yet — run `python src/models/train.py`.")

signal = generate(buy_prob, 1 - buy_prob, thr) if buy_prob is not None else "HOLD"

c1, c2, c3, c4 = st.columns(4)
c1.metric("BNB Price", f"${last['close']:.2f}")
c2.metric("Signal", signal)
c3.metric("Confidence", f"{(buy_prob * 100):.1f}%" if buy_prob is not None else "n/a")
c4.metric("Threshold", f"{thr:.0%}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("RSI", f"{last['rsi']:.1f}")
c2.metric("MACD", f"{last['macd']:+.2f}")
c3.metric("EMA 20", f"${last['ema20']:.2f}")
c4.metric("EMA 50", f"${last['ema50']:.2f}")

st.line_chart(df.set_index("timestamp")[["close", "ema20", "ema50"]].tail(300))

rep = ROOT / "backtests" / "results" / "latest.json"
if rep.exists():
    st.subheader("Latest backtest (out-of-sample, fees + slippage included)")
    st.json(rep.read_text())
