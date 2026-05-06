from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module `{module_name}` from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = _load_module("project_config", SRC_DIR / "config.py")
sys.modules["config"] = config

data_module = _load_module("project_data", SRC_DIR / "data.py")
tuning_module = _load_module("project_tuning", SRC_DIR / "tuning.py")

load_feature_target_dataset = data_module.load_feature_target_dataset
optimize_random_forest = tuning_module.optimize_random_forest


def main() -> None:
    X, y = load_feature_target_dataset()
    study = optimize_random_forest(X, y, n_trials=20, cv_splits=5, random_state=42)

    output_path = config.OPTUNA_RANDOM_FOREST_BEST_PARAMS_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_value_macro_f1": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved Optuna best params -> {output_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
