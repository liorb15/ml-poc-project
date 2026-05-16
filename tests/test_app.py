from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
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
    main_module = _load_module("project_main_for_app_test", PROJECT_ROOT / "scripts" / "main.py")
    _, X_test, _, y_test = main_module._load_dataset()
    main_module.write_metrics(main_module._evaluate_models(X_test, y_test))
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
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
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
    assert "prototype" in context["subtitle"].lower()
    assert "147" in context["prototype_positioning"]
    assert "faisabilité" in context["prototype_positioning"].lower()
    assert len(context["why_147_is_still_useful"]) >= 3
    assert "petit" in context["why_147_is_still_useful"][0].lower()
    assert context["presentation_objective"].startswith("Montrer")
    assert "signal" in context["presentation_hypothesis"].lower()
    assert "prototype" in context["presentation_verdict"].lower()
    assert "split" in context["no_piece_split_title"].lower()
    assert len(context["no_piece_split_reasons"]) >= 3
    assert "cipi" in context["cipi_title"].lower()
    assert len(context["cipi_benefits"]) >= 3
    assert len(context["method_steps"]) == 4
    assert "musicxml" in context["current_scope"][0].lower()
    assert "xml inconnu" in context["current_scope"][1].lower()
    assert all(Path(path).suffix == ".png" for path in context["plot_paths"].values())


def test_get_presentation_sections_returns_clear_demo_order():
    config_module = _load_module("project_config_sections", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    app_module = _load_module("project_app_sections", SRC_DIR / "app.py")

    sections = app_module.get_presentation_sections()

    assert len(sections) == 3
    assert sections[0]["title"] == "1. Problème, contexte et données"
    assert sections[1]["title"] == "2. Modèles, métriques et résultats"
    assert sections[2]["title"] == "3. Démo et usage réel"


def test_get_section_progress_labels_returns_readable_story_progression():
    config_module = _load_module("project_config_progress", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    app_module = _load_module("project_app_progress", SRC_DIR / "app.py")

    progress = app_module.get_section_progress_labels()

    assert progress == {
        "story": "1/3",
        "models": "2/3",
        "demo": "3/3",
    }


def test_build_model_comparison_chart_dataframe_returns_presentation_ready_frame():
    config_module = _load_module("project_config_chart", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    app_module = _load_module("project_app_chart", SRC_DIR / "app.py")

    metrics_df = app_module.load_metrics_dataframe()
    chart_df = app_module.build_model_comparison_chart_dataframe(metrics_df)

    assert isinstance(chart_df, pd.DataFrame)
    assert list(chart_df.columns) == ["model_name", "Holdout macro F1", "CV macro F1"]
    assert len(chart_df) == 3
    assert chart_df.iloc[0]["Holdout macro F1"] >= chart_df.iloc[-1]["Holdout macro F1"]


def test_build_model_display_dataframe_hides_internal_paths():
    config_module = _load_module("project_config_display", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    app_module = _load_module("project_app_display", SRC_DIR / "app.py")

    metrics_df = app_module.load_metrics_dataframe()
    display_df = app_module.build_model_display_dataframe(metrics_df)

    assert "model_path" not in display_df.columns
    assert list(display_df.columns) == [
        "Modèle",
        "Accuracy",
        "Macro F1 holdout",
        "Accuracy CV",
        "Std Accuracy CV",
        "Macro F1 CV",
        "Std Macro F1 CV",
    ]


def test_build_target_distribution_dataframe_returns_expected_class_counts():
    config_module = _load_module("project_config_target_dist", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    app_module = _load_module("project_app_target_dist", SRC_DIR / "app.py")

    distribution_df = app_module.build_target_distribution_dataframe()

    assert isinstance(distribution_df, pd.DataFrame)
    assert list(distribution_df.columns) == ["difficulty_label", "pieces"]
    counts = dict(zip(distribution_df["difficulty_label"], distribution_df["pieces"]))
    assert counts == {"beginner": 92, "intermediate": 45, "advanced": 10}


class _DummyColumn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeStreamlit:
    def markdown(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def dataframe(self, *args, **kwargs):
        return None

    def plotly_chart(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def columns(self, spec):
        if isinstance(spec, int):
            return [_DummyColumn() for _ in range(spec)]
        return [_DummyColumn() for _ in spec]


def test_render_models_part_runs_without_name_error(monkeypatch):
    config_module = _load_module("project_config_render_models", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    app_module = _load_module("project_app_render_models", SRC_DIR / "app.py")

    metrics_df = app_module.load_metrics_dataframe()
    context = app_module.get_project_context()

    monkeypatch.setattr(app_module, "st", _FakeStreamlit())
    monkeypatch.setattr(app_module, "_section_header", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_module, "_metric_card", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_module, "_show_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_module, "load_optuna_summary", lambda: None)

    app_module._render_models_part(metrics_df, context)


def test_predict_musicxml_bytes_returns_prediction_payload():
    config_module = _load_module("project_config_predict_demo", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    app_module = _load_module("project_app_predict_demo", SRC_DIR / "app.py")

    xml_bytes = config_module.DEMO_MIKROKOSMOS_XML_FILE.read_bytes()

    payload = app_module.predict_musicxml_bytes(xml_bytes, filename="99.xml")

    assert payload["predicted_label"] in {"beginner", "intermediate", "advanced"}
    assert payload["model_name"] == "Random Forest"
    assert payload["source_filename"] == "99.xml"
    assert payload["top_probability"] >= 0.0
    assert payload["top_probability"] <= 1.0
    assert isinstance(payload["feature_frame"], pd.DataFrame)
    assert list(payload["feature_frame"].columns) == ["Feature", "Valeur"]
    assert len(payload["feature_frame"]) >= 5
    assert isinstance(payload["probability_frame"], pd.DataFrame)
    assert list(payload["probability_frame"].columns) == ["Classe", "Probabilité"]
    assert set(payload["probability_frame"]["Classe"]) == {"beginner", "intermediate", "advanced"}


def test_predict_musicxml_bytes_accepts_mxl_archive():
    config_module = _load_module("project_config_predict_mxl", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    app_module = _load_module("project_app_predict_mxl", SRC_DIR / "app.py")

    xml_bytes = config_module.DEMO_MIKROKOSMOS_XML_FILE.read_bytes()

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w") as archive:
        archive.writestr("score.xml", xml_bytes)

    payload = app_module.predict_musicxml_bytes(archive_buffer.getvalue(), filename="demo.mxl")

    assert payload["source_filename"] == "demo.mxl"
    assert payload["predicted_label"] in {"beginner", "intermediate", "advanced"}
    assert len(payload["probability_frame"]) == 3


def test_recommend_musicxml_bytes_returns_same_level_neighbors_with_similarity_scores():
    config_module = _load_module("project_config_recommend_demo", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    app_module = _load_module("project_app_recommend_demo", SRC_DIR / "app.py")

    xml_bytes = config_module.DEMO_MIKROKOSMOS_XML_FILE.read_bytes()

    payload = app_module.predict_musicxml_bytes(xml_bytes, filename="99.xml")
    recommendations_df = app_module.recommend_musicxml_bytes(xml_bytes, filename="99.xml", top_k=5)

    assert isinstance(recommendations_df, pd.DataFrame)
    assert len(recommendations_df) == 5
    assert list(recommendations_df.columns) == [
        "piece_id",
        "work",
        "book",
        "difficulty_label",
        "henle_difficulty",
        "distance",
        "similarity_score",
    ]
    assert set(recommendations_df["difficulty_label"]) == {payload["predicted_label"]}
    assert recommendations_df["distance"].is_monotonic_increasing
    assert recommendations_df["similarity_score"].between(0.0, 1.0).all()
    assert 99 not in set(recommendations_df["piece_id"])


def test_streamlit_demo_accepts_mxl_extension_in_uploader_definition():
    app_source = (SRC_DIR / "app.py").read_text(encoding="utf-8")
    assert 'type=["xml", "musicxml", "mxl"]' in app_source


def test_build_business_impact_dataframe_covers_advanced_user_risk():
    config_module = _load_module("project_config_business_risk", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    app_module = _load_module("project_app_business_risk", SRC_DIR / "app.py")

    business_df = app_module.build_business_impact_dataframe()

    assert isinstance(business_df, pd.DataFrame)
    assert list(business_df.columns) == ["Situation utilisateur", "Risque produit", "Comment le prototype le gère"]
    assert any("advanced" in value.lower() for value in business_df["Situation utilisateur"])
    assert any("facile" in value.lower() for value in business_df["Risque produit"])


def test_build_feature_importance_dataframe_returns_sorted_top_features():
    config_module = _load_module("project_config_feature_importance", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    app_module = _load_module("project_app_feature_importance", SRC_DIR / "app.py")

    importance_df = app_module.build_feature_importance_dataframe(top_n=6)

    assert isinstance(importance_df, pd.DataFrame)
    assert list(importance_df.columns) == ["Feature", "Importance", "Importance (%)"]
    assert len(importance_df) == 6
    assert importance_df.iloc[0]["Importance"] >= importance_df.iloc[-1]["Importance"]
    assert importance_df["Importance (%)"].sum() > 0
