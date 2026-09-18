import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.calibration import PlattCalibrated, sweep_thresholds


def _toy(n=400, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y


def test_platt_outputs_valid_probabilities():
    X, y = _toy()
    base = RandomForestClassifier(n_estimators=10, random_state=0).fit(X[:300], y[:300])
    cal = PlattCalibrated(base).fit(X[:300], y[:300])
    p = cal.predict_proba(X[300:])
    assert p.shape == (100, 2)
    assert ((p >= 0) & (p <= 1)).all()
    assert np.allclose(p.sum(axis=1), 1.0)


def test_sweep_picks_threshold_with_enough_signals():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    proba = rng.uniform(0, 1, 1000)
    out = sweep_thresholds(y, proba, thresholds=(0.8, 0.9, 0.95), min_signals=50)
    assert out["suggested_threshold"] in (0.8, 0.9, 0.95)
    assert len(out["rows"]) == 3
