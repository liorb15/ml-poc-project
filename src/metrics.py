"""Student-owned metrics contract."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return metrics used to compare piano difficulty models."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
