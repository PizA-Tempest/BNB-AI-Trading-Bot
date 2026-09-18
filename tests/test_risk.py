from src.risk.manager import RiskManager


def test_position_size_scales_with_balance():
    r = RiskManager(balance=1000.0, max_risk_per_trade=0.01, stop_loss_pct=0.02)
    assert r.position_size(500.0) > 0
    r2 = RiskManager(balance=2000.0, max_risk_per_trade=0.01, stop_loss_pct=0.02)
    assert r2.position_size(500.0) == r.position_size(500.0) * 2


def test_max_positions_blocks():
    r = RiskManager(max_open_positions=1)
    r.on_fill()
    ok, _ = r.can_open()
    assert not ok


def test_drawdown_blocks():
    r = RiskManager(balance=800.0, peak=1000.0, max_drawdown=0.15)
    ok, _ = r.can_open()
    assert not ok
