# AGENTS.md — BNB AI Trading Bot

> Repo state: greenfield. Only `README.md` exists (spec + roadmap). No `src/`, `requirements.txt`, `config/`, `.env.example`, or tests yet. Treat `README.md` as the spec; build toward its Project Structure — don't invent a competing layout.

## Intended structure (from README — follow when creating files)

- `src/data/collector.py`, `src/data/preprocessing.py` — market data + cleaning
- `src/features/indicators.py` — RSI / MACD / EMA / Bollinger / ATR / volume / momentum / volatility
- `src/models/train.py`, `predict.py`, `evaluate.py` — baseline → LogReg → RF → XGBoost → LightGBM → LSTM/Transformer
- `src/strategy/signal.py` — BUY / SELL / HOLD gating
- `src/trading/paper_trader.py`, `src/trading/live_trader.py` — simulated vs real execution
- `src/risk/manager.py` — stop-loss, take-profit, position sizing, limits
- `dashboard/app.py` — Streamlit dashboard
- `backtests/` + `backtests/results/`, `reports/` — historical evaluation
- `config/config.yaml` + `.env` (never commit) — risk params, thresholds, keys

## Stack (intended)

Python + Pandas/NumPy + pandas-ta + XGBoost/LightGBM (+ LSTM/Transformer later) + VectorBT/Backtrader + FastAPI + Streamlit + Binance API. No lockfile/requirements yet — create `requirements.txt` when adding deps.

## Domain invariants (do not violate)

- Signal gating: `buy_prob >= CONFIDENCE_THRESHOLD (0.95)` → BUY; `sell_prob >= 0.95` → SELL; else HOLD. Threshold is configurable, lives in env/config, not hardcoded in multiple places.
- 95% target = precision on high-confidence signals with sufficient out-of-sample data — never claim it as overall accuracy or profit guarantee.
- Chronological splits only (train → validation → test → paper). Never randomly shuffle time-series data.
- Always report model metrics (precision/recall/F1/ROC-AUC) separately from trading metrics (win rate, profit factor, max drawdown, Sharpe/Sortino, net return).
- Backtests must include fees + slippage and state whether evaluation is out-of-sample.
- Live trading stays disabled by default (`TRADING_MODE=paper`). `LIVE_TRADING=true` requires explicit user request.

## Security

- Never hardcode or commit `BINANCE_API_KEY` / `BINANCE_API_SECRET` or `.env`. Use env vars + `.env.example` (template only).
- Recommend read/trade-only API permissions; no withdrawals.

## Expected commands (per README — verify before relying)

- Setup: `python -m venv venv` → `venv\Scripts\activate` (Windows) → `pip install -r requirements.txt` → copy `.env.example` to `.env`
- Train/evaluate: `python src/models/train.py`, `python src/models/evaluate.py`
- Backtest: `python -m backtests.run`
- Paper: `python src/trading/paper_trader.py` · Dashboard: `streamlit run dashboard/app.py`
- No lint/test/typecheck/CI configured yet — don't assume pytest, ruff, or mypy exist.
