"""Shared config loader: config/config.yaml + .env overrides."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    # Env overrides (highest priority for secrets / deployment)
    cfg["confidence_threshold"] = float(os.getenv("CONFIDENCE_THRESHOLD", cfg.get("confidence_threshold", 0.95)))
    cfg.setdefault("trading", {})["mode"] = os.getenv("TRADING_MODE", cfg.get("trading", {}).get("mode", "paper"))
    cfg.setdefault("risk", {})["max_risk_per_trade"] = float(
        os.getenv("RISK_PER_TRADE", cfg.get("risk", {}).get("max_risk_per_trade", 0.01))
    )
    cfg["risk"]["max_daily_loss"] = float(
        os.getenv("MAX_DAILY_LOSS", cfg.get("risk", {}).get("max_daily_loss", 0.03))
    )
    cfg["live_trading"] = os.getenv("LIVE_TRADING", "false").lower() == "true"
    cfg["symbol"] = os.getenv("SYMBOL", cfg.get("symbol", "BNBUSDT"))
    cfg["interval"] = os.getenv("INTERVAL", cfg.get("interval", "15m"))
    return cfg
