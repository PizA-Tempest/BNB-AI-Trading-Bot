# 🤖 BNB AI Trading Bot

An AI-powered cryptocurrency trading bot designed to analyze **BNB market data**, predict potential price movements, generate high-confidence trading signals, and execute trades automatically.

> ⚠️ **Disclaimer:** This project is for educational and research purposes. A 95% prediction accuracy or profitable trading performance cannot be guaranteed. Cryptocurrency markets are highly volatile, and past or backtested performance does not guarantee future results.

---

## 📌 Overview

**BNB AI Trading Bot** combines machine learning, technical analysis, and automated risk management to identify potential trading opportunities in the BNB market.

The system analyzes historical and real-time market data, generates technical indicators, and uses a machine learning model to estimate the probability of future price movement.

The primary objective is to develop a system capable of producing **high-confidence trading signals**, with a target of achieving **95%+ precision on selected signals during evaluation**.

### Core Flow

```text
BNB Market Data
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
Technical Indicators
       ↓
Machine Learning Model
       ↓
Prediction Probability
       ↓
Signal Filter
       ↓
Risk Management
       ↓
Trading Decision
       ↓
Paper / Live Trading
       ↓
Performance Monitoring
```

---

## 🎯 Project Goals

- 📊 Collect and process BNB market data
- 🤖 Develop a machine learning model for price movement prediction
- 📈 Generate BUY / SELL / HOLD signals
- 🎯 Target **95%+ precision for high-confidence signals**
- 🧪 Backtest the strategy using historical data
- 🛡️ Implement risk management
- 💰 Calculate potential profit and loss
- 🧑‍💻 Support paper trading before live trading
- 📡 Monitor trading performance in real time
- 📊 Provide a dashboard for analysis and monitoring

---

## ✨ Features

### 1. Market Data Collection

Collect historical and real-time BNB market information:

- Open
- High
- Low
- Close
- Volume
- Price changes
- Market volatility
- Trading volume

Example:

```text
Timestamp    Open    High    Low     Close   Volume
10:00        650     655     648     653     1250
10:05        653     658     651     657     1430
10:10        657     660     654     659     1610
```

---

### 2. Technical Analysis

The system generates technical indicators that can be used as model features.

Potential indicators include:

- RSI
- MACD
- EMA 20
- EMA 50
- EMA 200
- Bollinger Bands
- ATR
- Volume Change
- Price Momentum
- Volatility
- Rate of Change

Example:

```text
RSI       = 63.4
MACD      = 2.31
EMA20     = 654.21
EMA50     = 648.72
ATR       = 8.12
Volume Δ  = +17.4%
```

---

## 🧠 AI Prediction

The bot uses machine learning to estimate the probability of future BNB price movement.

Example:

```text
Current Price: $657.42

Prediction:

BUY Probability  = 95.4%
SELL Probability = 4.6%

Signal = BUY
```

The prediction system should focus on **probability and signal quality**, rather than assuming that the model can always predict the market correctly.

---

## 🎯 95% High-Confidence Signal Target

The project uses a configurable confidence threshold.

Example:

```python
CONFIDENCE_THRESHOLD = 0.95
```

Trading logic:

```python
if buy_probability >= 0.95:
    signal = "BUY"

elif sell_probability >= 0.95:
    signal = "SELL"

else:
    signal = "HOLD"
```

This means the bot only considers trades when the model estimates a sufficiently high probability.

### Important

A **95% confidence threshold is not the same as 95% guaranteed accuracy**.

The system must evaluate the actual performance using:

- Precision
- Recall
- Accuracy
- F1 Score
- Win Rate
- Profit Factor
- Maximum Drawdown
- Sharpe Ratio
- Total Return

---

## 🧪 Backtesting

Before using real money, the strategy should be tested against historical BNB data.

Example:

```text
Historical Data
      ↓
Train Model
      ↓
Validation
      ↓
Out-of-Sample Test
      ↓
Backtest Strategy
      ↓
Performance Report
```

Example report:

```text
══════════════════════════════════
       BNB BACKTEST REPORT
══════════════════════════════════

Test Period          : 2025-01-01 → 2026-06-30

Total Signals        : 1,284
Executed Trades      : 327

Prediction Accuracy  : 91.8%
Signal Precision     : 95.1%
Win Rate              : 68.5%

Profit Factor         : 1.74
Maximum Drawdown      : 11.2%
Net Return            : +34.7%

Trading Fees          : Included
Slippage              : Included
Out-of-Sample Test    : Yes

══════════════════════════════════
```

The report should clearly distinguish **model prediction metrics** from **actual trading performance**.

---

# 🛡️ Risk Management

The bot should never rely only on the AI prediction.

Risk management features include:

- Stop loss
- Take profit
- Position sizing
- Maximum risk per trade
- Maximum daily loss
- Maximum drawdown
- Maximum open positions
- Emergency stop
- Trading cooldown

Example configuration:

```yaml
risk:
  max_risk_per_trade: 0.01
  max_daily_loss: 0.03
  max_drawdown: 0.15
  max_open_positions: 3
```

For example, with a `$1,000` account:

```text
Account Balance = $1,000
Risk Per Trade  = 1%

Maximum Risk = $10
```

---

# 🧪 Paper Trading

The bot should support paper trading before connecting to a live trading account.

```text
AI Prediction
      ↓
Trading Signal
      ↓
Paper Trading Engine
      ↓
Virtual Position
      ↓
Performance Tracking
```

This allows the strategy to be evaluated without risking real funds.

---

# 💱 Trading Modes

The system can support three modes:

### 🧪 Backtest

Uses historical data.

```text
BACKTEST=true
```

### 📝 Paper Trading

Uses real-time market data but simulated trades.

```text
PAPER_TRADING=true
```

### 🔴 Live Trading

Executes real orders.

```text
LIVE_TRADING=true
```

Live trading should be disabled by default.

---

# 🏗️ System Architecture

```text
┌─────────────────────────────┐
│       BNB Market Data       │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│      Data Preprocessing     │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│     Feature Engineering     │
│ RSI / MACD / EMA / ATR ...  │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│       ML Prediction         │
│      XGBoost / LightGBM     │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│      Signal Generator       │
│       BUY / SELL / HOLD     │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│      Risk Management        │
└──────────────┬──────────────┘
               ↓
       ┌───────┴────────┐
       ↓                ↓
┌─────────────┐  ┌─────────────┐
│ Paper Trade │  │ Live Trade  │
└─────────────┘  └─────────────┘
       ↓                ↓
       └───────┬────────┘
               ↓
┌─────────────────────────────┐
│   Performance Monitoring    │
└─────────────────────────────┘
```

---

# 🧰 Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | XGBoost / LightGBM |
| Deep Learning | LSTM / Transformer |
| Data Processing | Pandas / NumPy |
| Technical Analysis | pandas-ta |
| Backtesting | VectorBT / Backtrader |
| Database | MySQL / PostgreSQL |
| API | FastAPI |
| Dashboard | Streamlit |
| Containerization | Docker |
| Exchange API | Binance API |
| Monitoring | Prometheus / Grafana |

---

# 📁 Project Structure

```text
bnb-ai-trading-bot/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── models/
│   ├── trained/
│   └── checkpoints/
│
├── src/
│   ├── data/
│   │   ├── collector.py
│   │   └── preprocessing.py
│   │
│   ├── features/
│   │   └── indicators.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   │
│   ├── strategy/
│   │   └── signal.py
│   │
│   ├── trading/
│   │   ├── paper_trader.py
│   │   └── live_trader.py
│   │
│   ├── risk/
│   │   └── manager.py
│   │
│   └── utils/
│
├── backtests/
│   ├── results/
│   └── reports/
│
├── dashboard/
│   └── app.py
│
├── tests/
│
├── config/
│   └── config.yaml
│
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# 📊 Machine Learning Pipeline

## 1. Collect Data

```text
BNB/USDT
5m / 15m / 1h candles
```

## 2. Create Features

```text
Price
Volume
RSI
MACD
EMA
ATR
Bollinger Bands
Momentum
Volatility
```

## 3. Create Target

Example:

```python
future_return = (
    future_price - current_price
) / current_price
```

The model can then classify future movement:

```text
1 → Price increases above threshold
0 → Otherwise
```

---

# 🧠 Model Training

Initial models:

```text
Baseline
   ↓
Logistic Regression
   ↓
Random Forest
   ↓
XGBoost
   ↓
LightGBM
   ↓
LSTM / Transformer
```

The project should compare models using the same out-of-sample evaluation methodology.

---

# 📈 Evaluation Metrics

The system should track:

### Classification

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
```

### Trading

```text
Win Rate
Profit Factor
Total Return
Maximum Drawdown
Sharpe Ratio
Sortino Ratio
Average Trade
Number of Trades
```

---

# ⚠️ Avoiding Data Leakage

Financial machine learning is highly vulnerable to data leakage.

The project uses chronological data splitting:

```text
2021 ───────── 2023
       TRAIN

2024
       VALIDATION

2025
       TEST

2026
       PAPER TRADING
```

Randomly shuffling time-series data should be avoided when it allows future information to influence training.

---

# 🔐 Security

API keys must never be stored directly in the source code.

Use environment variables:

```env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

Never commit:

```text
.env
```

to Git.

Use API permissions that limit what the bot can do, and avoid enabling withdrawal permissions.

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/bnb-ai-trading-bot.git

cd bnb-ai-trading-bot
```

## Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ⚙️ Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Configure:

```env
BINANCE_API_KEY=
BINANCE_API_SECRET=

TRADING_MODE=paper

CONFIDENCE_THRESHOLD=0.95

RISK_PER_TRADE=0.01
MAX_DAILY_LOSS=0.03
```

---

# ▶️ Running the Project

### Train Model

```bash
python src/models/train.py
```

### Evaluate Model

```bash
python src/models/evaluate.py
```

### Run Backtest

```bash
python -m backtests.run
```

### Start Paper Trading

```bash
python src/trading/paper_trader.py
```

### Start Dashboard

```bash
streamlit run dashboard/app.py
```

---

# 📊 Dashboard

The dashboard will display:

```text
┌─────────────────────────────────────┐
│          BNB AI TRADING BOT         │
├─────────────────────────────────────┤
│ BNB Price          $657.42          │
│ Prediction         BUY              │
│ Confidence         95.4%            │
│                                     │
│ RSI                63.4             │
│ MACD               +2.31            │
│ EMA 20             $654.21          │
│ EMA 50             $648.72          │
│                                     │
│ Open Position      0.50 BNB         │
│ P/L                +$8.32           │
└─────────────────────────────────────┘
```

---

# 🗺️ Development Roadmap

## Phase 1 — Data

- [ ] Connect to market data API
- [ ] Download historical BNB data
- [ ] Store OHLCV data
- [ ] Clean missing data
- [ ] Build preprocessing pipeline

## Phase 2 — Features

- [ ] RSI
- [ ] MACD
- [ ] EMA
- [ ] Bollinger Bands
- [ ] ATR
- [ ] Volume indicators
- [ ] Momentum indicators
- [ ] Volatility features

## Phase 3 — AI Model

- [ ] Build baseline model
- [ ] Train XGBoost
- [ ] Train LightGBM
- [ ] Test LSTM
- [ ] Compare models
- [ ] Calibrate prediction probabilities
- [ ] Evaluate out-of-sample performance

## Phase 4 — Strategy

- [ ] BUY signal
- [ ] SELL signal
- [ ] HOLD signal
- [ ] Confidence threshold
- [ ] Stop loss
- [ ] Take profit
- [ ] Position sizing

## Phase 5 — Backtesting

- [ ] Historical backtesting
- [ ] Include trading fees
- [ ] Include slippage
- [ ] Calculate drawdown
- [ ] Calculate profit factor
- [ ] Generate performance reports

## Phase 6 — Paper Trading

- [ ] Real-time market data
- [ ] Simulated orders
- [ ] Position tracking
- [ ] Performance tracking
- [ ] Error handling

## Phase 7 — Live Trading

- [ ] Exchange API integration
- [ ] Order execution
- [ ] Position management
- [ ] Emergency stop
- [ ] Monitoring
- [ ] Logging

---

# 🎯 Success Criteria

The project will evaluate success using multiple metrics rather than prediction accuracy alone.

```text
Model
 ├── Precision
 ├── Recall
 ├── F1 Score
 └── Calibration

Trading Strategy
 ├── Win Rate
 ├── Profit Factor
 ├── Return
 ├── Maximum Drawdown
 └── Risk-adjusted performance
```

The **95% target applies to high-confidence signal precision**, where supported by sufficient out-of-sample data. It is not a guarantee that the model will correctly predict 95% of all BNB price movements or that the strategy will be profitable.

---

# ⚠️ Risk Disclaimer

Cryptocurrency trading involves substantial financial risk.

This software is an experimental and educational project. The developers do not guarantee:

- 95% prediction accuracy
- Profitable trades
- Positive returns
- Protection against losses
- Future performance based on historical results

Users are responsible for their own trading decisions and financial risk.

**Never use money you cannot afford to lose.**

---

# 📄 License

This project is intended for educational and research purposes.

Add your preferred license here, for example:

```text
MIT License
```

---

# 👨‍💻 Author

**BNB AI Trading Bot**

Built as an experimental project combining:

```text
Artificial Intelligence
+
Machine Learning
+
Technical Analysis
+
Algorithmic Trading
+
Risk Management
```