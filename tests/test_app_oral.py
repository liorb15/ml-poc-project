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


def test_app_oral_sections_follow_simple_storytelling_flow():
    config_module = _load_module("project_config_oral", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    oral_module = _load_module("project_app_oral_sections", SRC_DIR / "app_oral.py")

    sections = oral_module.get_oral_sections()

    assert sections == [
        "Le problème",
        "Le dataset en un regard",
        "Les modèles en compétition",
        "Démo en direct",
        "Ce qu'on peut conclure",
    ]


def test_app_oral_context_uses_audience_facing_project_language():
    config_module = _load_module("project_config_oral_context", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    oral_module = _load_module("project_app_oral_context", SRC_DIR / "app_oral.py")

    context = oral_module.get_oral_context()

    assert "prototype ml" in context["subtitle"].lower()
    assert "musicxml" in context["subtitle"].lower()
    assert "preuve de concept" in context["hero_message"].lower()
    assert "verdict définitif" in context["story_beats"][2]["text"].lower()
    assert len(context["story_beats"]) == 3


def test_app_oral_demo_prediction_reuses_musicxml_pipeline():
    config_module = _load_module("project_config_oral_demo", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    oral_module = _load_module("project_app_oral_demo", SRC_DIR / "app_oral.py")

    payload = oral_module.build_oral_demo_payload(
        config_module.DEMO_MIKROKOSMOS_XML_FILE.read_bytes(), "99.xml"
    )

    assert payload["predicted_label"] in {"beginner", "intermediate", "advanced"}
    assert isinstance(payload["probability_frame"], pd.DataFrame)
    assert isinstance(payload["recommendations_frame"], pd.DataFrame)
    assert len(payload["probability_frame"]) == 3
    assert len(payload["recommendations_frame"]) > 0
    assert payload["model_name"] == "Random Forest"


def test_app_oral_eda_figures_expose_legends_or_clear_color_encoding():
    config_module = _load_module("project_config_oral_figs", SRC_DIR / "config.py")
    sys.modules["config"] = config_module
    oral_module = _load_module("project_app_oral_figs", SRC_DIR / "app_oral.py")

    notes_fig = oral_module.build_oral_notes_played_figure()
    measures_fig = oral_module.build_oral_notes_vs_measures_figure()

    assert notes_fig.layout.showlegend is not False
    assert measures_fig.layout.showlegend is not False
    assert len(notes_fig.data) >= 3
    assert len(measures_fig.data) >= 3
