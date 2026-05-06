from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate


def _load_module(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module `{module_name}` from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT_DIR = Path(__file__).resolve().parent

config = _load_module("project_config", SCRIPT_DIR.parent / "src" / "config.py")
sys.modules["config"] = config
load_dotenv(config.ENV_FILE)
PROJECT_ROOT = config.PROJECT_ROOT
SRC_DIR = config.SRC_DIR
APP_ENTRYPOINT = config.APP_ENTRYPOINT
MODELS = config.MODELS
STREAMLIT_HOST = config.STREAMLIT_HOST
STREAMLIT_PORT = config.STREAMLIT_PORT

data_module = _load_module("project_data", SRC_DIR / "data.py")
metrics_module = _load_module("project_metrics", SRC_DIR / "metrics.py")
results_module = _load_module("project_results", SRC_DIR / "results.py")
baselines_module = _load_module("project_baselines", SRC_DIR / "baselines.py")

load_dataset_split = data_module.load_dataset_split
compute_metrics = metrics_module.compute_metrics
write_metrics = results_module.write_metrics
build_models = baselines_module.build_models


def _validate_models_config() -> None:
    if not MODELS:
        raise ValueError("config.MODELS is empty. Add your trained models first.")

    for model_key, model_config in MODELS.items():
        if "path" not in model_config:
            raise ValueError(
                f"Missing `path` for model `{model_key}` in config.MODELS."
            )


def _validate_app_entrypoint() -> None:
    app_module = _load_module("project_app", APP_ENTRYPOINT)
    if not hasattr(app_module, "build_app") or not callable(app_module.build_app):
        raise TypeError("app.build_app must be a callable Streamlit entry point.")


def _streamlit_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [str(SRC_DIR)]
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def _load_dataset() -> tuple[Any, Any, Any, Any]:
    dataset_split = load_dataset_split()
    if not isinstance(dataset_split, tuple) or len(dataset_split) != 4:
        raise ValueError(
            "data.load_dataset_split() must return exactly four values: "
            "(X_train, X_test, y_train, y_test)."
        )

    return dataset_split


def _evaluate_models(X_test: Any, y_test: Any) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    X_train, _, y_train, _ = _load_dataset()
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "macro_f1": "f1_macro"}
    model_builders = build_models()

    for model_key, model_config in MODELS.items():
        if model_key not in model_builders:
            raise KeyError(
                f"No baseline builder found for model `{model_key}`."
            )

        holdout_model = clone(model_builders[model_key])
        holdout_model.fit(X_train, y_train)
        y_pred = holdout_model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)

        if not isinstance(metrics, dict) or not metrics:
            raise ValueError(
                "metrics.compute_metrics() must return a non-empty dictionary."
            )

        cv_model = clone(model_builders[model_key])
        cv_results = cross_validate(
            cv_model,
            X_train,
            y_train,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=None,
        )

        row: dict[str, object] = {
            "model_key": model_key,
            "model_name": model_config.get("name", model_key),
            "model_path": str(model_config["path"]),
        }

        for metric_name, metric_value in metrics.items():
            row[metric_name] = float(metric_value)

        row["cv_accuracy_mean"] = float(cv_results["test_accuracy"].mean())
        row["cv_accuracy_std"] = float(cv_results["test_accuracy"].std())
        row["cv_macro_f1_mean"] = float(cv_results["test_macro_f1"].mean())
        row["cv_macro_f1_std"] = float(cv_results["test_macro_f1"].std())

        rows.append(row)

    return rows


def _launch_streamlit() -> None:
    if not APP_ENTRYPOINT.exists():
        raise FileNotFoundError(f"Streamlit entry point not found: {APP_ENTRYPOINT}")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP_ENTRYPOINT),
            "--server.address",
            STREAMLIT_HOST,
            "--server.port",
            str(STREAMLIT_PORT),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=_streamlit_env(),
    )


def main() -> None:
    _validate_app_entrypoint()
    _validate_models_config()

    try:
        _, X_test, _, y_test = _load_dataset()
    except NotImplementedError as exc:
        raise NotImplementedError(
            "Dataset loading is still a template placeholder. "
            "Implement data.load_dataset_split()."
        ) from exc

    try:
        metrics_rows = _evaluate_models(X_test, y_test)
    except NotImplementedError as exc:
        raise NotImplementedError(
            "Metric computation is still a template placeholder. "
            "Implement metrics.compute_metrics()."
        ) from exc

    metrics_df = write_metrics(metrics_rows)

    print("Model evaluation completed. Metrics saved to results/model_metrics.csv")
    print(metrics_df.to_string(index=False))
    print(f"\nLaunching Streamlit on http://{STREAMLIT_HOST}:{STREAMLIT_PORT} ...")

    _launch_streamlit()


if __name__ == "__main__":
    main()
