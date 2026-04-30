from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_metrics_returns_numeric_accuracy_and_macro_f1():
    metrics_module = _load_module("project_metrics", SRC_DIR / "metrics.py")

    metrics = metrics_module.compute_metrics(
        ["beginner", "beginner", "intermediate", "advanced"],
        ["beginner", "intermediate", "intermediate", "advanced"],
    )

    assert set(metrics) == {"accuracy", "macro_f1"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
