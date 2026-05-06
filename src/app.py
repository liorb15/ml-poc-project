"""Streamlit app for the piano difficulty prototype."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config import MIKROKOSMOS_DIR, MODEL_METRICS_FILE


def get_project_context() -> dict[str, object]:
    return {
        "title": "Piano Difficulty Prediction Prototype",
        "subtitle": "Estimate the difficulty of a piano piece from symbolic score features and prepare recommendation-ready metadata.",
        "dataset_name": "Mikrokosmos-difficulty",
        "target_name": "difficulty_label",
        "target_labels": ["beginner", "intermediate", "advanced"],
        "dataset_path": str(MIKROKOSMOS_DIR),
        "feature_groups": [
            "volume and density of notes",
            "rests and rhythmic pacing",
            "rhythmic variety across note durations",
            "normalized note density per pitch class",
            "notes per measure normalized by pitch variety",
            "tempo-normalized note density per pitch class",
            "duration variability normalized by average note length",
            "tempo × duration-variability interaction",
            "pitch span normalized by pitch variety",
            "pitch span and pitch variety",
            "melodic interval jumps",
            "tempo-derived density proxy",
            "piece book / progression level",
        ],
        "limitations": [
            "The prototype is trained only on Mikrokosmos, so stylistic diversity is limited.",
            "The advanced class is small, which makes difficult pieces harder to predict robustly.",
            "Recommendation logic is not implemented yet; for now the app focuses on difficulty modelling.",
        ],
    }


def load_metrics_dataframe() -> pd.DataFrame:
    if not MODEL_METRICS_FILE.exists():
        return pd.DataFrame(
            columns=[
                "model_key",
                "model_name",
                "model_path",
                "accuracy",
                "macro_f1",
                "cv_accuracy_mean",
                "cv_accuracy_std",
                "cv_macro_f1_mean",
                "cv_macro_f1_std",
            ]
        )

    metrics_df = pd.read_csv(MODEL_METRICS_FILE)
    if "macro_f1" in metrics_df.columns:
        metrics_df = metrics_df.sort_values(
            by=["macro_f1", "accuracy"], ascending=[False, False]
        ).reset_index(drop=True)
    return metrics_df


def _metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div style="padding:1rem 1.1rem;border:1px solid rgba(255,255,255,.08);border-radius:18px;"
             "background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));">
            <div style="font-size:.82rem;opacity:.75;margin-bottom:.35rem;">{label}</div>
            <div style="font-size:1.7rem;font-weight:700;line-height:1.1;">{value}</div>
            <div style="font-size:.82rem;opacity:.7;margin-top:.35rem;">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_app() -> None:
    context = get_project_context()
    metrics_df = load_metrics_dataframe()

    st.set_page_config(page_title="Piano Difficulty Prototype", layout="wide")

    st.markdown(
        """
        <style>
        .block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {background: rgba(255,255,255,0.03); border-radius: 16px; padding: 0.6rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title(context["title"])
    st.caption(context["subtitle"])

    top_left, top_mid, top_right = st.columns(3)
    with top_left:
        _metric_card("Dataset", str(context["dataset_name"]), "Current prototype source")
    with top_mid:
        _metric_card("Target", str(context["target_name"]), "Predicted difficulty label")
    with top_right:
        _metric_card("Models evaluated", str(len(metrics_df)), "Current trained baselines")

    st.divider()

    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Project objective")
        st.write(
            "This prototype studies whether symbolic information extracted from piano scores "
            "can predict an interpretable difficulty level. The long-term goal is to combine "
            "that prediction layer with recommendation logic so a pianist can discover pieces "
            "that match both their taste and their current level."
        )

        st.subheader("Current dataset")
        st.write(
            "The current working dataset is **Mikrokosmos-difficulty**, a structured collection "
            "of piano scores derived from Bartók's *Mikrokosmos*. It is useful for prototyping "
            "because it provides clean MusicXML scores and explicit difficulty progression."
        )
        st.code(str(Path(context["dataset_path"])), language="bash")

        st.subheader("Feature engineering used")
        for feature_group in context["feature_groups"]:
            st.markdown(f"- {feature_group}")

    with right:
        st.subheader("Target classes")
        st.write(
            "For the prototype, difficulty is grouped into three coarse categories to make the "
            "classification problem more stable on a small dataset."
        )
        target_df = pd.DataFrame(
            {
                "class": context["target_labels"],
                "meaning": [
                    "entry-level and easy progression pieces",
                    "moderately demanding pieces",
                    "most technically demanding pieces in the dataset",
                ],
            }
        )
        st.dataframe(target_df, use_container_width=True, hide_index=True)

        st.subheader("Known limitations")
        for limitation in context["limitations"]:
            st.markdown(f"- {limitation}")

    st.divider()
    st.subheader("Baseline model comparison")

    if metrics_df.empty:
        st.info("No metrics file found yet. Run `python scripts/train_baseline.py` then `python scripts/main.py`.")
    else:
        display_df = metrics_df.copy()
        for col in ["accuracy", "macro_f1"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda x: round(float(x), 4))
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        best_row = metrics_df.iloc[0]
        best_col1, best_col2 = st.columns(2)
        with best_col1:
            st.metric("Best current model", str(best_row["model_name"]))
        with best_col2:
            st.metric("Best macro F1", f"{float(best_row['macro_f1']):.4f}")

    st.divider()
    st.subheader("Next steps")
    st.markdown(
        """
        - deepen symbolic feature extraction on the current Mikrokosmos pipeline
        - strengthen the recommendation layer using metadata and similarity on the existing catalogue
        - optionally add a broader symbolic dataset later if access becomes possible
        - turn the prototype into a user-facing exploration tool
        """
    )


if __name__ == "__main__":
    build_app()
