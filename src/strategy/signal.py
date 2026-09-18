"""BUY / SELL / HOLD gating on calibrated probabilities.

BUY  if buy_prob  >= threshold
SELL if sell_prob >= threshold
else HOLD. Threshold comes from config/env — never hardcode at call sites.
"""
from __future__ import annotations

BUY, SELL, HOLD = "BUY", "SELL", "HOLD"


def generate(buy_prob: float, sell_prob: float | None = None,
             threshold: float = 0.95) -> str:
    buy_prob = float(buy_prob)
    sell_prob = float(1.0 - buy_prob) if sell_prob is None else float(sell_prob)
    if buy_prob >= threshold:
        return BUY
    if sell_prob >= threshold:
        return SELL
    return HOLD
