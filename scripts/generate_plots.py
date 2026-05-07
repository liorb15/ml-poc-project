from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")


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
model_io_module = _load_module("project_model_io", SRC_DIR / "model_io.py")


def _load_eda_dataframe() -> pd.DataFrame:
    X, y = data_module.load_feature_target_dataset()
    return pd.concat([X.reset_index(drop=True), y.rename("difficulty_label").reset_index(drop=True)], axis=1)


def save_eda_plot() -> Path:
    eda_df = _load_eda_dataframe()
    counts = eda_df["difficulty_label"].value_counts().reindex(["beginner", "intermediate", "advanced"], fill_value=0)

    output_path = config.PLOTS_DIR / "eda_class_distribution.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=["#4f46e5", "#0ea5e9", "#f97316"])
    ax.set_title("Répartition des classes de difficulté", fontsize=15, weight="bold")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Nombre de morceaux")
    ax.set_ylim(0, max(counts.values) + 15)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, str(int(value)), ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_notes_played_by_class_plot() -> Path:
    eda_df = _load_eda_dataframe()
    output_path = config.PLOTS_DIR / "eda_notes_played_by_class.png"
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    sns.boxplot(
        data=eda_df,
        x="difficulty_label",
        y="notes_played",
        hue="difficulty_label",
        order=["beginner", "intermediate", "advanced"],
        hue_order=["beginner", "intermediate", "advanced"],
        palette=["#4f46e5", "#0ea5e9", "#f97316"],
        dodge=False,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=eda_df,
        x="difficulty_label",
        y="notes_played",
        order=["beginner", "intermediate", "advanced"],
        color="#0f172a",
        alpha=0.35,
        size=3,
        jitter=0.22,
        ax=ax,
    )
    ax.set_title("Répartition de notes_played selon la classe", fontsize=15, weight="bold")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Nombre de notes jouées")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_notes_vs_measures_plot() -> Path:
    eda_df = _load_eda_dataframe()
    output_path = config.PLOTS_DIR / "eda_notes_vs_measures_by_class.png"
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    sns.scatterplot(
        data=eda_df,
        x="measures",
        y="notes_played",
        hue="difficulty_label",
        hue_order=["beginner", "intermediate", "advanced"],
        palette={"beginner": "#4f46e5", "intermediate": "#0ea5e9", "advanced": "#f97316"},
        s=60,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )
    ax.set_title("Notes jouées vs nombre de mesures", fontsize=15, weight="bold")
    ax.set_xlabel("Nombre de mesures")
    ax.set_ylabel("Nombre de notes jouées")
    ax.legend(title="Classe")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_model_comparison_plot() -> Path:
    metrics_df = pd.read_csv(config.MODEL_METRICS_FILE).sort_values("macro_f1", ascending=False)
    plot_df = metrics_df[["model_name", "macro_f1", "cv_macro_f1_mean"]].rename(
        columns={"macro_f1": "Holdout macro F1", "cv_macro_f1_mean": "CV macro F1"}
    )
    melted = plot_df.melt(id_vars="model_name", var_name="metric", value_name="score")

    output_path = config.PLOTS_DIR / "model_comparison_macro_f1.png"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=melted, x="model_name", y="score", hue="metric", ax=ax)
    ax.set_title("Comparaison des modèles sur le macro F1", fontsize=15, weight="bold")
    ax.set_xlabel("Modèle")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Métrique")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_best_model_plot() -> Path:
    X_train, X_test, y_train, y_test = data_module.load_dataset_split()
    model_path = config.MODELS["random_forest"]["path"]
    model = model_io_module.load_model(model_path)
    if hasattr(model, "fit") and not hasattr(model, "estimators_"):
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    labels = ["beginner", "intermediate", "advanced"]
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    output_path = config.PLOTS_DIR / "best_model_random_forest_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Random Forest — matrice de confusion", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        save_eda_plot(),
        save_notes_played_by_class_plot(),
        save_notes_vs_measures_plot(),
        save_model_comparison_plot(),
        save_best_model_plot(),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
