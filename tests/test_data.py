from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_feature_target_dataset_returns_explicit_X_and_y():
    config_module = _load_module("project_config", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    data_module = _load_module("project_data", SRC_DIR / "data.py")

    X, y = data_module.load_feature_target_dataset()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y) == 147
    assert "difficulty_label" not in X.columns
    assert set(y.unique()) <= {"beginner", "intermediate", "advanced"}


def test_load_dataset_split_returns_non_empty_train_test_split():
    config_module = _load_module("project_config", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    data_module = _load_module("project_data", SRC_DIR / "data.py")

    dataset_split = data_module.load_dataset_split()

    assert isinstance(dataset_split, tuple)
    assert len(dataset_split) == 4

    X_train, X_test, y_train, y_test = dataset_split

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert len(X_train) > len(X_test) > 0
    assert len(X_train) + len(X_test) == 147
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_load_dataset_split_uses_expected_coarse_difficulty_labels():
    config_module = _load_module("project_config", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    data_module = _load_module("project_data", SRC_DIR / "data.py")

    _, _, y_train, y_test = data_module.load_dataset_split()

    labels = set(y_train).union(set(y_test))

    assert labels <= {"beginner", "intermediate", "advanced"}
    assert "beginner" in labels
    assert "intermediate" in labels


def test_default_dataset_name_is_mikrokosmos(monkeypatch):
    monkeypatch.delenv("PIANO_DATASET", raising=False)
    config_module = _load_module("project_config_default", SRC_DIR / "config.py")

    assert config_module.DATASET_NAME == "mikrokosmos"


def test_load_dataset_split_rejects_unknown_dataset_name(monkeypatch):
    monkeypatch.setenv("PIANO_DATASET", "unknown-dataset")
    config_module = _load_module("project_config_unknown", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    data_module = _load_module("project_data_unknown", SRC_DIR / "data.py")

    with pytest.raises(ValueError, match="Unsupported dataset"):
        data_module.load_dataset_split()


def test_load_dataset_split_reports_missing_cipi_dataset_clearly(monkeypatch):
    monkeypatch.setenv("PIANO_DATASET", "cipi")
    monkeypatch.setenv("CIPI_DIR", str(PROJECT_ROOT / "data" / "external" / "CIPI-unavailable-test"))
    config_module = _load_module("project_config_cipi", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    data_module = _load_module("project_data_cipi", SRC_DIR / "data.py")

    with pytest.raises(FileNotFoundError, match="CIPI"):
        data_module.load_dataset_split()


def test_feature_matrix_contains_basic_symbolic_features():
    config_module = _load_module("project_config", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    data_module = _load_module("project_data", SRC_DIR / "data.py")

    X_train, _, _, _ = data_module.load_dataset_split()

    expected_columns = {
        "notes_played",
        "measures",
        "pitch_span",
        "chord_notes",
        "book_code",
        "rest_ratio",
        "unique_pitch_count",
        "avg_pitch_interval",
        "max_pitch_interval",
        "notes_per_second_proxy",
        "duration_std",
        "accidental_ratio",
        "key_signature_complexity",
        "time_signature_changes",
        "rhythmic_variety",
        "notes_per_pitch_class",
        "notes_per_measure_per_pitch_class",
        "duration_cv_proxy",
    }

    assert expected_columns.issubset(set(X_train.columns))


def test_engineered_features_are_numeric_and_non_negative():
    config_module = _load_module("project_config", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    data_module = _load_module("project_data", SRC_DIR / "data.py")

    X_train, _, _, _ = data_module.load_dataset_split()

    for column in [
        "rest_ratio",
        "unique_pitch_count",
        "avg_pitch_interval",
        "max_pitch_interval",
        "notes_per_second_proxy",
        "duration_std",
        "accidental_ratio",
        "key_signature_complexity",
        "time_signature_changes",
        "rhythmic_variety",
        "notes_per_pitch_class",
        "notes_per_measure_per_pitch_class",
        "duration_cv_proxy",
    ]:
        assert pd.api.types.is_numeric_dtype(X_train[column])
        assert (X_train[column] >= 0).all()


def test_exploratory_normalized_features_exist_in_raw_dataset_dataframe():
    config_module = _load_module("project_config", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    data_module = _load_module("project_data", SRC_DIR / "data.py")

    dataset_df = data_module._load_mikrokosmos_dataframe()

    for column in [
        "notes_per_second_per_pitch_class",
        "tempo_duration_interaction",
        "span_per_unique_pitch",
    ]:
        assert column in dataset_df.columns
        assert pd.api.types.is_numeric_dtype(dataset_df[column])
        assert (dataset_df[column] >= 0).all()
        assert dataset_df[column].nunique() > 1


def test_new_density_and_duration_features_have_signal_in_the_dataset():
    config_module = _load_module("project_config", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    data_module = _load_module("project_data", SRC_DIR / "data.py")

    X_train, X_test, _, _ = data_module.load_dataset_split()
    features_df = pd.concat([X_train, X_test], axis=0)

    for column in [
        "rhythmic_variety",
        "notes_per_pitch_class",
        "notes_per_measure_per_pitch_class",
        "duration_cv_proxy",
    ]:
        assert features_df[column].sum() > 0
        assert features_df[column].nunique() > 1
