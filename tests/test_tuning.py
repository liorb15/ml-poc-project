from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_optimize_random_forest_returns_study_with_best_params_and_metric():
    config_module = _load_module("project_config_tuning", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    data_module = _load_module("project_data_tuning", SRC_DIR / "data.py")
    tuning_module = _load_module("project_tuning", SRC_DIR / "tuning.py")

    X, y = data_module.load_feature_target_dataset()
    study = tuning_module.optimize_random_forest(
        X,
        y,
        n_trials=1,
        cv_splits=3,
        random_state=42,
    )

    assert len(study.trials) == 1
    assert 0.0 <= float(study.best_value) <= 1.0
    assert {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}.issubset(
        set(study.best_params)
    )


def test_build_tuned_random_forest_returns_configured_estimator():
    tuning_module = _load_module("project_tuning_build", SRC_DIR / "tuning.py")

    model = tuning_module.build_tuned_random_forest(
        {
            "n_estimators": 180,
            "max_depth": 9,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        random_state=42,
    )

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 180
    assert model.max_depth == 9
    assert model.min_samples_leaf == 2
    assert model.max_features == "sqrt"
    assert model.class_weight == "balanced_subsample"
    assert model.random_state == 42
