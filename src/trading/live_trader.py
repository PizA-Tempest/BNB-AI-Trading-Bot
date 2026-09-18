"""Live trader — DISABLED BY DEFAULT. Requires explicit opt-in.

Runs only when BOTH hold:
  TRADING_MODE=live  AND  LIVE_TRADING=true
Anything else → hard exit. API keys come from env only, withdrawals must stay disabled.
"""
from __future__ import annotations

import os
import sys


def main() -> None:
    mode = os.getenv("TRADING_MODE", "paper").lower()
    live = os.getenv("LIVE_TRADING", "false").lower() == "true"
    if not (mode == "live" and live):
        print(f"Live trading disabled (TRADING_MODE={mode} LIVE_TRADING={live}). "
              "Set TRADING_MODE=live + LIVE_TRADING=true explicitly to proceed.")
        sys.exit(0)
    if not os.getenv("BINANCE_API_KEY") or not os.getenv("BINANCE_API_SECRET"):
        sys.exit("Missing BINANCE_API_KEY / BINANCE_API_SECRET in environment.")
    try:
        from binance.client import Client  # type: ignore  # python-binance
    except ImportError:
        sys.exit("python-binance not installed. pip install python-binance, then retry.")
    Client(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))
    print("LIVE MODE armed. Order execution wiring goes here — start with tiny sizes + emergency stop.")
    # NOTE: intentional stub — fill in order logic only after paper-trading validation.


if __name__ == "__main__":
    main()
