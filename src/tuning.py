"""Optuna-based hyperparameter tuning helpers for the piano difficulty prototype."""

from __future__ import annotations

from typing import Any

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def build_tuned_random_forest(
    best_params: dict[str, Any],
    random_state: int = 42,
) -> RandomForestClassifier:
    """Return a RandomForestClassifier configured from Optuna best params."""

    return RandomForestClassifier(
        n_estimators=int(best_params["n_estimators"]),
        max_depth=int(best_params["max_depth"]),
        min_samples_leaf=int(best_params["min_samples_leaf"]),
        max_features=best_params["max_features"],
        class_weight="balanced_subsample",
        random_state=random_state,
    )


def optimize_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 20,
    cv_splits: int = 5,
    random_state: int = 42,
) -> optuna.study.Study:
    """Optimize Random Forest hyperparameters with Optuna on macro-F1 CV."""

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        model = build_tuned_random_forest(params, random_state=random_state)
        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="f1_macro",
            n_jobs=None,
        )
        return float(scores.mean())

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study
