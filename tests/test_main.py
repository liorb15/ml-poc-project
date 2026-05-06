from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluate_models_returns_holdout_and_cross_validation_metrics():
    config_module = _load_module("project_config_main", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    main_module = _load_module("project_main", SCRIPTS_DIR / "main.py")

    _, X_test, _, y_test = main_module._load_dataset()
    rows = main_module._evaluate_models(X_test, y_test)

    assert len(rows) >= 3
    model_keys = {row["model_key"] for row in rows}
    assert {"log_reg", "svm_rbf", "random_forest"}.issubset(model_keys)

    for row in rows:
        assert "accuracy" in row
        assert "macro_f1" in row
        assert "cv_accuracy_mean" in row
        assert "cv_accuracy_std" in row
        assert "cv_macro_f1_mean" in row
        assert "cv_macro_f1_std" in row
        assert 0.0 <= float(row["cv_accuracy_mean"]) <= 1.0
        assert 0.0 <= float(row["cv_macro_f1_mean"]) <= 1.0
        assert float(row["cv_accuracy_std"]) >= 0.0
        assert float(row["cv_macro_f1_std"]) >= 0.0
