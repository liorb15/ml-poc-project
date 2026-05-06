from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

for dir in [
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]:
    dir.mkdir(exist_ok=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"
OPTUNA_RANDOM_FOREST_BEST_PARAMS_FILE = RESULTS_DIR / "optuna_random_forest_best_params.json"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

DATASET_NAME = os.getenv("PIANO_DATASET", "mikrokosmos").strip().lower()

MIKROKOSMOS_DIR = DATA_DIR / "external" / "Mikrokosmos-difficulty"
CIPI_DIR = Path(os.getenv("CIPI_DIR", str(DATA_DIR / "external" / "CIPI")))

MODELS = {
    "log_reg": {
        "name": "Logistic Regression",
        "description": "Baseline linear classifier on symbolic score features.",
        "path": MODELS_DIR / "mikrokosmos_log_reg.joblib",
    },
    "svm_rbf": {
        "name": "SVM (RBF)",
        "description": "Kernel SVM baseline on standardized symbolic score features.",
        "path": MODELS_DIR / "mikrokosmos_svm_rbf.joblib",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Tree ensemble baseline on symbolic score features.",
        "path": MODELS_DIR / "mikrokosmos_random_forest.joblib",
    },
}
