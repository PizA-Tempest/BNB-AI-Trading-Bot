"""Risk manager: sizing, stop-loss/take-profit, daily-loss, drawdown, cooldown."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskManager:
    max_risk_per_trade: float = 0.01
    max_daily_loss: float = 0.03
    max_drawdown: float = 0.15
    max_open_positions: int = 3
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    cooldown_bars: int = 3
    starting_balance: float = 1000.0
    balance: float = 1000.0
    peak: float = 1000.0
    day_start_balance: float = 1000.0
    open_positions: int = 0
    bars_since_close: int = 10**9
    halted: bool = False
    _log: list = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: dict) -> "RiskManager":
        r = cfg.get("risk", {})
        return cls(**{k: r[k] for k in
                      ("max_risk_per_trade", "max_daily_loss", "max_drawdown",
                       "max_open_positions", "stop_loss_pct", "take_profit_pct",
                       "cooldown_bars", "starting_balance") if k in r},
                   balance=r.get("starting_balance", 1000.0),
                   peak=r.get("starting_balance", 1000.0),
                   day_start_balance=r.get("starting_balance", 1000.0))

    # -- checks ---------------------------------------------------------
    def can_open(self) -> tuple[bool, str]:
        if self.halted:
            return False, "emergency stop engaged"
        if self.open_positions >= self.max_open_positions:
            return False, "max open positions reached"
        if self.bars_since_close < self.cooldown_bars:
            return False, "cooldown active"
        dd = (self.peak - self.balance) / self.peak if self.peak else 0
        if dd >= self.max_drawdown:
            return False, f"max drawdown hit ({dd:.1%})"
        day_loss = (self.day_start_balance - self.balance) / self.day_start_balance
        if day_loss >= self.max_daily_loss:
            return False, f"max daily loss hit ({day_loss:.1%})"
        return True, "ok"

    def position_size(self, price: float) -> float:
        """Quantity so that a stop-loss hit loses ~max_risk_per_trade of balance."""
        risk_amt = self.balance * self.max_risk_per_trade
        per_unit_risk = price * self.stop_loss_pct
        return max(risk_amt / per_unit_risk, 0.0) if per_unit_risk > 0 else 0.0

    def stop_take(self, entry: float, side: str) -> tuple[float, float]:
        if side == "BUY":
            return entry * (1 - self.stop_loss_pct), entry * (1 + self.take_profit_pct)
        return entry * (1 + self.stop_loss_pct), entry * (1 - self.take_profit_pct)

    # -- bookkeeping -----------------------------------------------------
    def on_fill(self) -> None:
        self.open_positions += 1

    def on_close(self, balance: float) -> None:
        self.balance = balance
        self.peak = max(self.peak, balance)
        self.open_positions = max(0, self.open_positions - 1)
        self.bars_since_close = 0

    def on_bar(self) -> None:
        self.bars_since_close += 1

    def emergency_stop(self, reason: str) -> None:
        self.halted = True
        self._log.append(reason)
