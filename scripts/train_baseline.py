from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module `{module_name}` from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

config = _load_module("project_config", SRC_DIR / "config.py")
sys.modules["config"] = config

data_module = _load_module("project_data", SRC_DIR / "data.py")
load_dataset_split = data_module.load_dataset_split


MODELS_TO_TRAIN = {
    "log_reg": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
    ),
}


def main() -> None:
    X_train, _, y_train, _ = load_dataset_split()

    for model_key, model in MODELS_TO_TRAIN.items():
        model.fit(X_train, y_train)
        output_path = config.MODELS[model_key]["path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
        print(f"Saved {model_key} -> {output_path}")


if __name__ == "__main__":
    main()
