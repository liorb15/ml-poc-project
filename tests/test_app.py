from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_metrics_dataframe_returns_ranked_results():
    config_module = _load_module("project_config", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    app_module = _load_module("project_app", SRC_DIR / "app.py")

    metrics_df = app_module.load_metrics_dataframe()

    assert isinstance(metrics_df, pd.DataFrame)
    assert not metrics_df.empty
    assert list(metrics_df.columns) == [
        "model_key",
        "model_name",
        "model_path",
        "accuracy",
        "macro_f1",
    ]
    assert metrics_df.iloc[0]["macro_f1"] >= metrics_df.iloc[-1]["macro_f1"]


def test_project_context_mentions_piano_and_mikrokosmos():
    config_module = _load_module("project_config", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    app_module = _load_module("project_app", SRC_DIR / "app.py")

    context = app_module.get_project_context()

    assert "piano" in context["title"].lower()
    assert "mikrokosmos" in context["dataset_name"].lower()
    assert context["target_name"] == "difficulty_label"
