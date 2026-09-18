from src.strategy.signal import generate


def test_gating():
    assert generate(0.96, 0.04, 0.95) == "BUY"
    assert generate(0.02, 0.98, 0.95) == "SELL"
    assert generate(0.60, 0.40, 0.95) == "HOLD"
    assert generate(0.95, 0.05, 0.95) == "BUY"  # boundary inclusive
