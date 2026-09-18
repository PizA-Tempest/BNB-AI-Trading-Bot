"""Platt scaling wrapper: sigmoid-calibrates a fitted classifier's scores.

Fit the base model on train, then fit this wrapper on the chronological
validation slice. Version-proof (no CalibratedClassifierCV API dependence)
and picklable — unpickling auto-imports this module.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrated:
    def __init__(self, base, calibrator: LogisticRegression | None = None):
        self.base = base
        self.calibrator = calibrator or LogisticRegression(max_iter=1000)

    def fit(self, X, y):
        scores = self.base.predict_proba(X)[:, 1].reshape(-1, 1)
        self.calibrator.fit(scores, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        scores = self.base.predict_proba(X)[:, 1].reshape(-1, 1)
        p1 = self.calibrator.predict_proba(scores)[:, 1]
        return np.column_stack([1 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def sweep_thresholds(y_true: np.ndarray, proba: np.ndarray,
                     thresholds: tuple = (0.80, 0.85, 0.90, 0.95),
                     min_signals: int = 50) -> dict:
    """Pick the gate from data: highest signal precision with >= min_signals."""
    best: dict | None = None
    rows = []
    for thr in thresholds:
        mask = proba >= thr
        n = int(mask.sum())
        prec = float((y_true[mask] == 1).mean()) if n else 0.0
        rows.append({"threshold": thr, "signals": n, "signal_precision": round(prec, 4)})
        if n >= min_signals and (best is None or prec > best["signal_precision"]):
            best = rows[-1]
    if best is None:  # nothing cleared the bar — report the most precise option honestly
        best = max(rows, key=lambda r: (r["signal_precision"], r["signals"]))
        best = {**best, "note": f"no threshold reached min_signals={min_signals}"}
    return {"rows": rows, "suggested_threshold": best["threshold"], "detail": best}
