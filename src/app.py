"""Streamlit app for the piano difficulty prototype."""

from __future__ import annotations

import io
import json
import importlib.util
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    MIKROKOSMOS_DIR,
    MODELS,
    MODEL_METRICS_FILE,
    OPTUNA_RANDOM_FOREST_BEST_PARAMS_FILE,
    PLOTS_DIR,
)


SRC_DIR = Path(__file__).resolve().parent


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module `{module_name}` from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data_module = _load_module("project_data_for_app", SRC_DIR / "data.py")
model_io_module = _load_module("project_model_io_for_app", SRC_DIR / "model_io.py")
FEATURE_COLUMNS = data_module.FEATURE_COLUMNS
load_feature_target_dataset = data_module.load_feature_target_dataset
load_catalog_dataframe = data_module.load_catalog_dataframe
extract_symbolic_features_from_musicxml_path = data_module._extract_symbolic_features
load_model = model_io_module.load_model

BEST_MODEL_KEY = "random_forest"


SECTION_ORDER = [
    {"key": "story", "title": "1. Problème, contexte et données"},
    {"key": "models", "title": "2. Modèles, métriques et résultats"},
    {"key": "demo", "title": "3. Démo et usage réel"},
]

FEATURE_EXPLANATIONS = [
    ("Notes et longueur", "Quelle quantité de matière musicale la pièce contient au total."),
    ("Rythme", "À quel point le rythme est varié et dense dans la partition."),
    ("Accords", "À quelle fréquence le pianiste doit gérer plusieurs notes en même temps."),
    ("Registre et sauts", "Jusqu’où les mains se déplacent et quelle largeur de clavier la pièce mobilise."),
    ("Contexte tonal et métrique", "Altérations, complexité de la tonalité, tempo et changements de signature rythmique."),
]


def get_project_context() -> dict[str, object]:
    return {
        "title": "Prototype de prédiction de difficulté au piano",
        "subtitle": "Prototype de prédiction à partir de partitions MusicXML : problème posé, données disponibles, résultats obtenus et portée réelle des conclusions.",
        "dataset_name": "Mikrokosmos-difficulty",
        "target_name": "difficulty_label",
        "target_labels": ["beginner", "intermediate", "advanced"],
        "dataset_path": str(MIKROKOSMOS_DIR),
        "retained_feature_count": len(FEATURE_COLUMNS),
        "presentation_objective": "Montrer qu’on peut approcher la difficulté d’un morceau à partir de sa partition, avec une démarche ML cohérente.",
        "presentation_hypothesis": "Même sur un petit corpus, il peut déjà exister un signal exploitable si les features musicales sont bien construites et si l’évaluation reste rigoureuse.",
        "presentation_verdict": "Le bon message final n’est pas “on a résolu le problème”, mais “on a validé un prototype crédible et une base sérieuse pour aller plus loin”.",
        "method_steps": [
            "partir des fichiers MusicXML et des métadonnées de difficulté disponibles",
            "extraire des features symboliques décrivant rythme, registre, accords, densité et contexte tonal",
            "entraîner plusieurs modèles supervisés sur ces variables",
            "comparer les résultats avec holdout et validation croisée pour juger la faisabilité du prototype",
        ],
        "plot_paths": {
            "eda": str(PLOTS_DIR / "eda_class_distribution.png"),
            "eda_notes_played_by_class": str(PLOTS_DIR / "eda_notes_played_by_class.png"),
            "eda_notes_vs_measures": str(PLOTS_DIR / "eda_notes_vs_measures_by_class.png"),
            "model_comparison": str(PLOTS_DIR / "model_comparison_macro_f1.png"),
            "best_model": str(PLOTS_DIR / "best_model_random_forest_confusion_matrix.png"),
        },
        "business_application": "L’application visée est un assistant de tri ou de recommandation pédagogique capable d’estimer rapidement si une partition correspond au niveau d’un élève, d’un catalogue pédagogique ou d’une plateforme musicale.",
        "demo_steps": [
            "on récupère une partition au format MusicXML",
            "on extrait automatiquement les mêmes features symboliques que dans le dataset d’entraînement",
            "on applique le meilleur modèle actuel pour proposer une classe de difficulté",
            "on utilise cette sortie comme aide à la décision pédagogique, pas comme verdict absolu",
        ],
        "current_scope": [
            "Le prototype modélise la difficulté à partir de partitions MusicXML déjà intégrées au corpus de travail.",
            "Il ne constitue pas encore une démonstration produit où l’on fournit un XML inconnu pour obtenir instantanément une prédiction finale.",
        ],
        "no_piece_split_title": "Pourquoi on n’a pas split les morceaux",
        "no_piece_split_reasons": [
            "L’unité prédite est le morceau entier : le label de difficulté disponible décrit la pièce globale, pas des segments locaux à l’intérieur du morceau.",
            "Découper un morceau en sous-parties aurait artificiellement multiplié le nombre de lignes sans créer de nouveaux labels fiables, donc avec un risque fort de faux gain statistique.",
            "Des segments venant du même morceau se ressemblent beaucoup ; si on les répartit entre train et test, on crée facilement une fuite d’information et une évaluation trop optimiste.",
            "Pour ce prototype, il était plus propre méthodologiquement de garder une ligne par morceau complet, même si cela limite la taille du dataset.",
        ],
        "cipi_title": "Ce que CIPI pourrait apporter concrètement",
        "cipi_benefits": [
            "Un corpus plus grand permettrait de tester le prototype sur davantage d’exemples réellement indépendants, au lieu de rester bloqué à 147 morceaux.",
            "Une plus grande diversité de compositeurs et de styles réduirait le risque que le modèle apprenne surtout le style Mikrokosmos plutôt que la difficulté pianistique en général.",
            "Des classes de difficulté potentiellement mieux représentées rendraient l’évaluation plus crédible, en particulier pour les morceaux difficiles aujourd’hui trop rares.",
            "Avec plus de données, on pourrait comparer des modèles plus ambitieux et tester plus sérieusement des raffinements comme la séparation main gauche / main droite ou des features plus riches.",
        ],
        "prototype_positioning": "Ce travail est un prototype : 147 pièces ne suffisent pas pour revendiquer un modèle final universel, mais elles suffisent pour tester la faisabilité, comparer plusieurs baselines et montrer que les features extraites de la partition portent un vrai signal prédictif.",
        "why_147_is_still_useful": [
            "Le dataset est petit, mais le problème est cadré comme une étude de faisabilité, pas comme un benchmark industriel final.",
            "L’entrée du modèle est un jeu de données tabulaire structuré avec des features musicales construites à la main, donc des modèles classiques restent adaptés malgré un échantillon limité.",
            "L’évaluation ne repose pas sur un seul split chanceux : le prototype utilise aussi une validation croisée stratifiée.",
            "L’objectif est de valider une chaîne ML complète et de détecter un signal utile avant un passage à une base plus large comme CIPI.",
        ],
        "limitations": [
            "Le dataset est petit : seulement 147 pièces solo sont utilisées actuellement.",
            "La classe advanced est rare, donc les morceaux difficiles restent les plus compliqués à prédire correctement.",
            "Toutes les pièces viennent de Mikrokosmos, donc la diversité stylistique reste limitée.",
            "La couche de recommandation n’est pas encore implémentée : le prototype se concentre sur la prédiction de difficulté.",
        ],
    }


def get_presentation_sections() -> list[dict[str, str]]:
    return SECTION_ORDER.copy()


def get_section_progress_labels() -> dict[str, str]:
    total = len(SECTION_ORDER)
    return {
        section["key"]: f"{index}/{total}"
        for index, section in enumerate(SECTION_ORDER, start=1)
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


@lru_cache(maxsize=1)
def load_best_model() -> object:
    return load_model(Path(MODELS[BEST_MODEL_KEY]["path"]))


def build_business_impact_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Situation utilisateur": "Un élève cherche un morceau advanced mais reçoit en réalité un morceau trop facile.",
                "Risque produit": "L’utilisateur peut juger la recommandation trop facile ou peu crédible si le niveau annoncé semble artificiellement gonflé.",
                "Comment le prototype le gère": "Le prototype doit être présenté comme une aide au tri initial : la sortie du modèle complète un jugement pédagogique, elle ne remplace pas la validation finale par un enseignant ou un curateur.",
            },
            {
                "Situation utilisateur": "Un morceau difficile est sous-évalué et proposé à un pianiste intermédiaire.",
                "Risque produit": "L’utilisateur risque de se décourager, d’abandonner le morceau ou de considérer la plateforme comme peu fiable sur les cas exigeants.",
                "Comment le prototype le gère": "Le macro F1 est suivi pour éviter de masquer les erreurs sur la classe advanced, et la recommandation doit rester accompagnée d’un avertissement de prudence sur les cas limites.",
            },
            {
                "Situation utilisateur": "Le morceau est hors du style Mikrokosmos mais on demande quand même une estimation.",
                "Risque produit": "Le modèle peut produire une réponse plausible en apparence alors qu’il extrapole hors de son domaine d’entraînement.",
                "Comment le prototype le gère": "La démo montre ce comportement comme un test exploratoire : intéressant pour voir la réaction du pipeline, mais insuffisant pour revendiquer une généralisation réelle.",
            },
        ]
    )


def build_feature_importance_dataframe(top_n: int = 8) -> pd.DataFrame:
    model = load_best_model()
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance", "Importance (%)"])

    importance_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": [float(value) for value in model.feature_importances_],
        }
    )
    total_importance = float(importance_df["Importance"].sum())
    importance_df["Importance (%)"] = importance_df["Importance"].map(
        lambda value: (value / total_importance * 100.0) if total_importance else 0.0
    )
    importance_df = importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    return importance_df.head(top_n).copy()


def build_probability_figure(probability_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        probability_df,
        x="Classe",
        y="Probabilité",
        color="Classe",
        text="Probabilité",
        category_orders={"Classe": ["beginner", "intermediate", "advanced"]},
        color_discrete_map={
            "beginner": "#4f46e5",
            "intermediate": "#0ea5e9",
            "advanced": "#f97316",
        },
    )
    fig.update_layout(
        showlegend=False,
        height=340,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.88)",
        xaxis_title="Classe prédite possible",
        yaxis_title="Probabilité",
        yaxis=dict(range=[0, 1]),
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside", cliponaxis=False)
    return fig


def build_feature_importance_figure(importance_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        importance_df.sort_values(by="Importance", ascending=True),
        x="Importance (%)",
        y="Feature",
        orientation="h",
        color="Importance (%)",
        text="Importance (%)",
        color_continuous_scale=["#dbeafe", "#60a5fa", "#4f46e5"],
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.88)",
        xaxis_title="Poids relatif dans la Random Forest (%)",
        yaxis_title="Feature",
        coloraxis_showscale=False,
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    return fig


def _extract_musicxml_payload(xml_bytes: bytes, filename: str) -> tuple[bytes, str]:
    suffix = Path(filename).suffix.lower()
    if suffix != ".mxl":
        return xml_bytes, suffix or ".musicxml"

    with zipfile.ZipFile(io.BytesIO(xml_bytes)) as archive:
        xml_members = [
            name for name in archive.namelist()
            if not name.endswith("/") and name.lower().endswith((".xml", ".musicxml"))
        ]
        preferred_members = [name for name in xml_members if "meta-inf/" not in name.lower()]
        member_name = (preferred_members or xml_members or [None])[0]
        if member_name is None:
            raise ValueError("Archive .mxl invalide : aucun fichier MusicXML exploitable trouvé.")
        return archive.read(member_name), Path(member_name).suffix or ".musicxml"


def predict_musicxml_bytes(xml_bytes: bytes, filename: str = "uploaded.musicxml") -> dict[str, object]:
    xml_payload, suffix = _extract_musicxml_payload(xml_bytes, filename)
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(xml_payload)
            temp_path = Path(temp_file.name)

        feature_dict = extract_symbolic_features_from_musicxml_path(temp_path)
        feature_row = {column: float(feature_dict.get(column, 0.0)) for column in FEATURE_COLUMNS}
        feature_matrix = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)

        model = load_best_model()
        predicted_label = str(model.predict(feature_matrix)[0])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(feature_matrix)[0]
            class_labels = [str(label) for label in model.classes_]
        else:
            class_labels = [predicted_label]
            probabilities = [1.0]

        probability_frame = pd.DataFrame(
            {
                "Classe": class_labels,
                "Probabilité": [float(value) for value in probabilities],
            }
        ).sort_values(by="Probabilité", ascending=False).reset_index(drop=True)

        top_feature_names = build_feature_importance_dataframe(top_n=6)["Feature"].tolist()
        feature_frame = pd.DataFrame(
            {
                "Feature": top_feature_names,
                "Valeur": [round(float(feature_row.get(feature_name, 0.0)), 4) for feature_name in top_feature_names],
            }
        )

        return {
            "source_filename": filename,
            "model_name": str(MODELS[BEST_MODEL_KEY]["name"]),
            "predicted_label": predicted_label,
            "top_probability": float(probability_frame.iloc[0]["Probabilité"]),
            "probability_frame": probability_frame,
            "feature_frame": feature_frame,
            "feature_row": feature_row,
        }
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _piece_id_from_filename(filename: str) -> int | None:
    stem = Path(filename).stem
    return int(stem) if stem.isdigit() else None


def recommend_musicxml_bytes(xml_bytes: bytes, filename: str = "uploaded.musicxml", top_k: int = 5) -> pd.DataFrame:
    """Recommend similar catalog pieces from the predicted difficulty level.

    The recommendation is intentionally simple for the prototype: predict the
    uploaded score level, keep catalog pieces with that same level, and rank
    them by normalized distance on the same symbolic features used by training.
    """

    payload = predict_musicxml_bytes(xml_bytes, filename=filename)
    predicted_label = str(payload["predicted_label"])
    uploaded_features = pd.Series(payload["feature_row"], dtype=float).reindex(FEATURE_COLUMNS).fillna(0.0)

    catalog_df = load_catalog_dataframe()
    candidate_df = catalog_df[catalog_df["difficulty_label"] == predicted_label].copy()
    uploaded_piece_id = _piece_id_from_filename(filename)
    if uploaded_piece_id is not None and "piece_id" in candidate_df.columns:
        candidate_df = candidate_df[candidate_df["piece_id"] != uploaded_piece_id]

    if candidate_df.empty:
        return pd.DataFrame(
            columns=[
                "piece_id",
                "work",
                "book",
                "difficulty_label",
                "henle_difficulty",
                "distance",
                "similarity_score",
            ]
        )

    feature_df = catalog_df[FEATURE_COLUMNS].astype(float)
    scale = feature_df.std(axis=0).replace(0, 1.0)
    center = feature_df.mean(axis=0)
    candidate_features = (candidate_df[FEATURE_COLUMNS].astype(float) - center) / scale
    uploaded_vector = (uploaded_features - center) / scale
    distances = ((candidate_features - uploaded_vector) ** 2).sum(axis=1) ** 0.5

    recommendations = candidate_df[
        ["piece_id", "work", "book", "difficulty_label", "henle_difficulty"]
    ].copy()
    recommendations["distance"] = distances.astype(float)
    recommendations = recommendations.sort_values("distance", ascending=True).head(top_k).reset_index(drop=True)
    max_distance = float(recommendations["distance"].max()) if not recommendations.empty else 0.0
    if max_distance > 0:
        recommendations["similarity_score"] = 1.0 - (recommendations["distance"] / max_distance)
    else:
        recommendations["similarity_score"] = 1.0
    recommendations["similarity_score"] = recommendations["similarity_score"].clip(0.0, 1.0)
    return recommendations


def build_target_distribution_dataframe() -> pd.DataFrame:
    _, y = load_feature_target_dataset()
    ordered_labels = ["beginner", "intermediate", "advanced"]
    counts = y.value_counts().reindex(ordered_labels, fill_value=0)
    return pd.DataFrame(
        {
            "difficulty_label": counts.index,
            "pieces": counts.values,
        }
    )


def build_model_comparison_chart_dataframe(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(columns=["model_name", "Holdout macro F1", "CV macro F1"])

    chart_df = metrics_df[["model_name", "macro_f1", "cv_macro_f1_mean"]].copy()
    chart_df = chart_df.rename(
        columns={
            "macro_f1": "Holdout macro F1",
            "cv_macro_f1_mean": "CV macro F1",
        }
    )
    chart_df[["Holdout macro F1", "CV macro F1"]] = chart_df[[
        "Holdout macro F1",
        "CV macro F1",
    ]].astype(float)
    chart_df = chart_df.sort_values(by="Holdout macro F1", ascending=False).reset_index(drop=True)
    return chart_df


def build_model_display_dataframe(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "Modèle",
                "Accuracy",
                "Macro F1 holdout",
                "Accuracy CV",
                "Std Accuracy CV",
                "Macro F1 CV",
                "Std Macro F1 CV",
            ]
        )

    display_df = metrics_df[
        [
            "model_name",
            "accuracy",
            "macro_f1",
            "cv_accuracy_mean",
            "cv_accuracy_std",
            "cv_macro_f1_mean",
            "cv_macro_f1_std",
        ]
    ].copy()
    display_df = display_df.rename(
        columns={
            "model_name": "Modèle",
            "accuracy": "Accuracy",
            "macro_f1": "Macro F1 holdout",
            "cv_accuracy_mean": "Accuracy CV",
            "cv_accuracy_std": "Std Accuracy CV",
            "cv_macro_f1_mean": "Macro F1 CV",
            "cv_macro_f1_std": "Std Macro F1 CV",
        }
    )
    for col in [
        "Accuracy",
        "Macro F1 holdout",
        "Accuracy CV",
        "Std Accuracy CV",
        "Macro F1 CV",
        "Std Macro F1 CV",
    ]:
        display_df[col] = display_df[col].map(lambda value: round(float(value), 4))
    return display_df


def build_target_distribution_figure(distribution_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        distribution_df,
        x="difficulty_label",
        y="pieces",
        color="difficulty_label",
        text="pieces",
        category_orders={"difficulty_label": ["beginner", "intermediate", "advanced"]},
        color_discrete_map={
            "beginner": "#4f46e5",
            "intermediate": "#0ea5e9",
            "advanced": "#f97316",
        },
    )
    fig.update_layout(
        showlegend=False,
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.88)",
        xaxis_title="Classe",
        yaxis_title="Nombre de morceaux",
    )
    fig.update_traces(textposition="outside", marker_line_color="#ffffff", marker_line_width=1.5)
    return fig


def build_model_comparison_figure(chart_df: pd.DataFrame) -> go.Figure:
    melted_df = chart_df.melt(id_vars="model_name", var_name="metric", value_name="score")
    fig = px.bar(
        melted_df,
        x="model_name",
        y="score",
        color="metric",
        barmode="group",
        text="score",
        color_discrete_map={
            "Holdout macro F1": "#4f46e5",
            "CV macro F1": "#06b6d4",
        },
    )
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.88)",
        xaxis_title="Modèle",
        yaxis_title="Score",
        legend_title="Métrique",
        yaxis=dict(range=[0, 1]),
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
    return fig


def load_optuna_summary() -> dict[str, object] | None:
    if not OPTUNA_RANDOM_FOREST_BEST_PARAMS_FILE.exists():
        return None

    return json.loads(OPTUNA_RANDOM_FOREST_BEST_PARAMS_FILE.read_text(encoding="utf-8"))


def _show_plot(plot_path: str, caption: str) -> None:
    path = Path(plot_path)
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Plot manquant pour l’instant : {path.name}")


def _metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section_header(title: str, intro: str, progress_label: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading-row">
            <div>
                <div class="section-chip">Section {progress_label}</div>
                <h2>{title}</h2>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(intro)


def _render_header(context: dict[str, object], metrics_df: pd.DataFrame) -> None:
    st.markdown('<div class="hero-shell">', unsafe_allow_html=True)
    st.title(str(context["title"]))
    st.caption(str(context["subtitle"]))
    st.markdown(
        f"""
        <div class="hero-banner">
            <div>
                <div class="hero-kicker">Positionnement du travail</div>
                <div class="hero-text">{context['prototype_positioning']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_left, top_mid, top_right = st.columns(3)
    with top_left:
        _metric_card("Dataset", str(context["dataset_name"]), "Jeu de données utilisé pour le prototype")
    with top_mid:
        _metric_card("Cible", str(context["target_name"]), "Classe de difficulté prédite")
    with top_right:
        _metric_card("Modèles comparés", str(len(metrics_df)), "Baselines évaluées actuellement")

    story_left, story_mid, story_right = st.columns(3)
    with story_left:
        st.markdown(
            f"""
            <div class="story-card">
                <div class="story-label">Question testée</div>
                <div class="story-text">{context['presentation_objective']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with story_mid:
        st.markdown(
            f"""
            <div class="story-card">
                <div class="story-label">Hypothèse</div>
                <div class="story-text">{context['presentation_hypothesis']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with story_right:
        st.markdown(
            f"""
            <div class="story-card">
                <div class="story-label">Conclusion attendue</div>
                <div class="story-text">{context['presentation_verdict']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_goal_section(metrics_df: pd.DataFrame, context: dict[str, object]) -> None:
    progress = get_section_progress_labels()
    _section_header(
        "1. Objectif du projet",
        "Cette première section pose le problème, explicite la méthode suivie et fixe clairement ce que le prototype permet — ou non — de conclure à ce stade.",
        progress["goal"],
    )
    left, right = st.columns([1.2, 1])
    with left:
        st.write(
            "La question centrale est la suivante : **peut-on estimer la difficulté d’un morceau de piano à partir de sa partition, sans passer par l’audio ?**"
        )
        st.write(
            "Le projet part donc d’une partition MusicXML, transforme son contenu en variables numériques, puis apprend à relier ces variables à une difficulté globale du morceau."
        )
        st.markdown("### Méthode suivie")
        for index, step in enumerate(context["method_steps"], start=1):
            st.markdown(f"**Étape {index}.** {step}.")
        st.markdown("### Ce que montre ce prototype")
        st.markdown(
            """
            - il ne s’agit pas d’un produit final ;
            - le travail montre qu’une **démarche ML cohérente** est possible sur ce problème ;
            - et qu’il existe déjà un **signal prédictif exploitable** dans les partitions.
            """
        )
        st.markdown("### Ce que le périmètre couvre aujourd’hui")
        for item in context["current_scope"]:
            st.markdown(f"- {item}")
        st.markdown("### Pourquoi 147 pièces suffisent quand même pour un prototype sérieux")
        for reason in context["why_147_is_still_useful"]:
            st.markdown(f"- {reason}")
    with right:
        st.markdown(
            """
            <div class="callout-card">
                <div class="callout-title">Résultat principal à retenir</div>
                <div class="callout-text">Il s’agit d’un <strong>prototype</strong>, pas d’un benchmark final universel. Le résultat principal est la validation d’une faisabilité méthodologique et d’un pipeline déjà exploitable.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if metrics_df.empty:
            st.info("Les métriques des modèles ne sont pas encore disponibles.")
        else:
            best_row = metrics_df.iloc[0]
            _metric_card("Meilleure baseline actuelle", str(best_row["model_name"]), "Modèle le plus solide pour l’instant")
            _metric_card("Meilleur macro F1 holdout", f"{float(best_row['macro_f1']):.4f}", "Premier chiffre fort à commenter")
            _metric_card("Pourquoi cette métrique compte", "Macro F1", "Elle donne le même poids aux classes beginner, intermediate et advanced malgré le déséquilibre du dataset")


def _render_dataset_section(context: dict[str, object]) -> None:
    progress = get_section_progress_labels()
    _section_header(
        "2. Jeu de données et cible",
        "Ici, on montre sur quoi repose concrètement le prototype, et pourquoi la taille du dataset est à la fois une limite et une information importante à interpréter.",
        progress["dataset"],
    )
    distribution_df = build_target_distribution_dataframe()
    left, right = st.columns([1.05, 1])
    with left:
        st.write(
            "Le dataset actuel est **Mikrokosmos-difficulty**. Après nettoyage, on conserve **147 pièces de piano solo**. "
            "Les labels d’origine sont regroupés en trois classes plus simples afin de rendre l’apprentissage possible sur un corpus réduit."
        )
        st.write(
            "À ce stade, ce dataset ne sert pas à prouver une généralisation forte sur tout le répertoire piano ; il sert à vérifier si une première modélisation sérieuse est possible."
        )
        st.markdown(f"### {context['no_piece_split_title']}")
        for reason in context["no_piece_split_reasons"]:
            st.markdown(f"- {reason}")
        st.markdown(f"### {context['cipi_title']}")
        for benefit in context["cipi_benefits"]:
            st.markdown(f"- {benefit}")
        st.code(str(Path(context["dataset_path"])), language="bash")
        st.dataframe(distribution_df, width="stretch", hide_index=True)
    with right:
        st.write("Visuel utile : répartition des classes.")
        chart_df = distribution_df.set_index("difficulty_label")
        st.bar_chart(chart_df)
        st.caption(
            "Ce graphique est important car la classe `advanced` est beaucoup plus petite que les autres. "
            "Il explique tout de suite pourquoi cette classe sera la plus délicate à prédire correctement."
        )


def _render_features_section(context: dict[str, object]) -> None:
    progress = get_section_progress_labels()
    _section_header(
        "3. Ce que voit le modèle",
        "Cette section explique concrètement comment une partition devient une ligne de données : quelles informations sont extraites, ce qu’elles représentent musicalement, et ce qui n’est pas encore modélisé.",
        progress["features"],
    )
    st.write(
        f"Le pipeline d’entraînement actuel conserve **{context['retained_feature_count']} features principales**. "
        "L’idée n’est pas de toutes les détailler à l’oral, mais de montrer qu’on capture plusieurs dimensions plausibles de la difficulté pianistique."
    )
    st.markdown(
        "Le modèle ne lit donc pas une image de partition et n’écoute pas d’interprétation : il exploite une représentation symbolique du morceau, convertie en variables numériques."
    )

    feature_cols = st.columns(len(FEATURE_EXPLANATIONS))
    for column, (title, explanation) in zip(feature_cols, FEATURE_EXPLANATIONS):
        with column:
            st.markdown(
                f"""
                <div style="height:100%;padding:1rem;border:1px solid rgba(15,23,42,.08);border-radius:18px;
                     background:white;box-shadow:0 10px 24px rgba(15,23,42,.05);">
                    <div style="font-weight:700;color:#0f172a;margin-bottom:.45rem;">{title}</div>
                    <div style="font-size:.92rem;color:#475569;line-height:1.45;">{explanation}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Exemples de features conservées")
    examples_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "signification simple": [
                "nombre total d’événements de notes",
                "notes jouées uniquement",
                "nombre de silences",
                "nombre de mesures",
                "notes jouées dans des accords",
                "largeur de registre au clavier",
                "durée moyenne des notes",
                "densité moyenne de notes par mesure",
                "part des notes jouées en accord",
                "part des silences",
                "nombre de hauteurs distinctes",
                "densité de notes rapportée à la variété de hauteurs",
                "densité par mesure normalisée par la variété de hauteurs",
                "saut mélodique moyen",
                "plus grand saut mélodique",
                "information moyenne de tempo",
                "proxy de densité basé sur le tempo",
                "variation des durées de notes",
                "variation rythmique normalisée",
                "nombre de durées distinctes",
                "part des altérations",
                "complexité moyenne de l’armure",
                "nombre de changements de signature rythmique",
                "groupe de cahier Mikrokosmos",
            ],
        }
    )
    st.dataframe(examples_df, width="stretch", hide_index=True, height=420)


def _render_model_section(metrics_df: pd.DataFrame) -> None:
    progress = get_section_progress_labels()
    _section_header(
        "4. Comparaison des modèles",
        "Ici, on répond à la vraie question expérimentale : est-ce qu’un modèle arrive déjà à capter quelque chose d’utile, et lequel s’en sort le mieux dans ce cadre de prototype ?",
        progress["models"],
    )
    if metrics_df.empty:
        st.info("Aucun fichier de métriques trouvé pour l’instant. Lance `python scripts/main.py` pour les régénérer.")
        return

    comparison_df = build_model_comparison_chart_dataframe(metrics_df)

    left, right = st.columns([1.05, 1])
    with left:
        display_df = metrics_df.copy()
        numeric_cols = [
            "accuracy",
            "macro_f1",
            "cv_accuracy_mean",
            "cv_accuracy_std",
            "cv_macro_f1_mean",
            "cv_macro_f1_std",
        ]
        for col in numeric_cols:
            display_df[col] = display_df[col].map(lambda value: round(float(value), 4))
        st.dataframe(display_df, width="stretch", hide_index=True)
    with right:
        st.write("Visuel utile : macro F1 holdout vs macro F1 en validation croisée.")
        st.bar_chart(comparison_df.set_index("model_name"))
        st.caption(
            "Ce graphique est utile car il compare le résultat final sur le test avec le résultat moyen en validation. "
            "Il donne rapidement une idée du modèle le plus solide et de celui qui paraît moins stable."
        )
        st.markdown(
            "**Comment lire cette section :** l’accuracy indique la part de bonnes prédictions au total, tandis que le macro F1 vérifie si le modèle reste équilibré entre les classes, y compris la classe `advanced`, beaucoup plus rare."
        )

    best_row = metrics_df.iloc[0]
    st.markdown(
        f"""
        <div class="callout-card soft">
            <div class="callout-title">Lecture rapide du résultat</div>
            <div class="callout-text">Le point important à commenter n’est pas seulement le meilleur score absolu, mais le fait qu’un modèle comme <strong>{best_row['model_name']}</strong> fasse déjà ressortir un signal cohérent malgré un jeu de données petit et déséquilibré.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    a, b, c = st.columns(3)
    with a:
        _metric_card("Meilleure baseline", str(best_row["model_name"]), "Modèle en tête actuellement")
    with b:
        _metric_card("Macro F1 holdout", f"{float(best_row['macro_f1']):.4f}", "Score final sur le jeu de test")
    with c:
        _metric_card("Macro F1 CV", f"{float(best_row['cv_macro_f1_mean']):.4f}", "Score moyen en validation croisée")


def _render_takeaways_section(context: dict[str, object]) -> None:
    progress = get_section_progress_labels()
    _section_header(
        "5. Conclusion et limites",
        "On termine en séparant clairement ce que le prototype permet déjà de dire, et ce qu’il ne permet pas encore d’affirmer sérieusement.",
        progress["takeaways"],
    )
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("### Ce qu’il faut retenir")
        st.markdown(
            """
            - Le projet dispose déjà d’une chaîne complète, depuis le parsing MusicXML jusqu’à l’évaluation des modèles.
            - Trois modèles ont été comparés, et **Random Forest** est la baseline la plus solide pour l’instant.
            - L’app garde un récit simple : problème → données → features → modèles → conclusion.
            - Le principal goulot d’étranglement vient maintenant davantage du **dataset** que d’un manque de sophistication du modèle.
            """
        )
        st.markdown("### Ce que cette version permet réellement d’affirmer")
        st.markdown(
            """
            - la difficulté pianistique laisse déjà apparaître un signal exploitable dans la partition ;
            - un pipeline supervisé simple permet de comparer plusieurs approches de manière cohérente ;
            - en revanche, cette version ne suffit pas encore à affirmer une généralisation robuste à tout le répertoire piano.
            """
        )
        optuna_summary = load_optuna_summary()
        if optuna_summary is not None:
            st.markdown("### Suite Optuna")
            st.write(
                "Une expérience de tuning Optuna a été lancée séparément sur la Random Forest pour chercher une configuration plus performante. "
                f"Meilleur macro F1 CV trouvé : **{float(optuna_summary['best_value_macro_f1']):.4f}**."
            )
            st.json(optuna_summary)
    with right:
        st.markdown("### Limites du prototype")
        for limitation in context["limitations"]:
            st.markdown(f"- {limitation}")
        st.markdown(
            """
            <div class="callout-card soft">
                <div class="callout-title">Conclusion</div>
                <div class="callout-text">Le prochain vrai gain viendra probablement d’un dataset plus riche, pas seulement d’une complexité de modélisation cosmétique. Le résultat actuel valide donc surtout un prototype prometteur, plutôt qu’un modèle définitivement généralisable.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_story_part(context: dict[str, object]) -> None:
    progress = get_section_progress_labels()
    _section_header(
        "1. Problème, contexte et données",
        "Cette première partie présente le problème, l’intérêt concret du projet et la manière dont une partition MusicXML devient un exemple exploitable pour le modèle.",
        progress["story"],
    )

    top_left, top_right = st.columns([1.15, 1])
    with top_left:
        st.markdown("### Problème posé")
        st.write(
            "L’objectif est d’estimer la difficulté d’un morceau de piano à partir de sa **partition**. Le travail se place donc du côté de la représentation symbolique du morceau, et non de l’audio interprété."
        )
        st.markdown("### Application visée")
        st.write(str(context["business_application"]))
        st.markdown("### Méthode suivie")
        for index, step in enumerate(context["method_steps"], start=1):
            st.markdown(f"**Étape {index}.** {step}.")
        st.markdown("### Ce que le périmètre couvre aujourd’hui")
        for item in context["current_scope"]:
            st.markdown(f"- {item}")
    with top_right:
        distribution_df = build_target_distribution_dataframe()
        distribution_fig = build_target_distribution_figure(distribution_df)
        st.plotly_chart(distribution_fig, width="stretch")
        st.caption(
            "Ce premier visuel d’EDA montre immédiatement le déséquilibre du corpus : la classe `advanced` est beaucoup plus rare que les classes `beginner` et `intermediate`."
        )

    st.markdown("### Pourquoi ce dataset reste utile malgré sa petite taille")
    for reason in context["why_147_is_still_useful"]:
        st.markdown(f"- {reason}")

    st.markdown("### À quoi ressemble concrètement le dataset ?")
    st.write(
        "Au-delà du simple comptage des classes, on peut regarder comment les morceaux se répartissent selon quelques variables très structurantes. Ici, les visuels montrent surtout que les classes se séparent déjà partiellement par **taille de pièce**, **nombre de notes jouées** et **nombre de mesures**."
    )
    eda_left, eda_mid, eda_right = st.columns(3)
    with eda_left:
        _show_plot(
            context["plot_paths"]["eda"],
            "EDA — répartition des classes de difficulté",
        )
    with eda_mid:
        _show_plot(
            context["plot_paths"]["eda_notes_played_by_class"],
            "EDA — distribution de notes_played par classe",
        )
    with eda_right:
        _show_plot(
            context["plot_paths"]["eda_notes_vs_measures"],
            "EDA — notes jouées vs nombre de mesures",
        )
    st.caption(
        "Lecture rapide : les morceaux `advanced` se concentrent nettement sur des pièces plus longues et plus denses. Ce n’est pas une définition universelle de la difficulté, mais c’est bien un signal présent dans ce corpus."
    )

    dataset_left, dataset_right = st.columns([1.1, 1])
    with dataset_left:
        st.markdown("### Jeu de données utilisé")
        st.write(
            "Le prototype s’appuie sur **Mikrokosmos-difficulty**. Après filtrage des pièces à quatre mains et vérification des fichiers MusicXML disponibles, le corpus de travail contient **147 morceaux solo**."
        )
        st.code(str(Path(context["dataset_path"])), language="bash")
        st.dataframe(distribution_df, width="stretch", hide_index=True)
    with dataset_right:
        st.markdown("### Comment une partition devient une ligne de données")
        st.write(
            f"Chaque morceau est transformé en **{context['retained_feature_count']} features principales** décrivant notamment la densité de notes, la variété rythmique, les accords, l’étendue du registre, les sauts et le contexte tonal."
        )
        for title, explanation in FEATURE_EXPLANATIONS:
            st.markdown(f"- **{title}** : {explanation}")

    st.markdown(f"### {context['no_piece_split_title']}")
    for reason in context["no_piece_split_reasons"]:
        st.markdown(f"- {reason}")


def _render_models_part(metrics_df: pd.DataFrame, context: dict[str, object]) -> None:
    progress = get_section_progress_labels()
    _section_header(
        "2. Modèles, métriques et résultats",
        "Cette deuxième partie explique quels modèles sont testés, pourquoi le macro F1 est suivi de près et quels résultats ressortent réellement du prototype.",
        progress["models"],
    )

    st.markdown("### Modèles comparés")
    st.write(
        "Trois baselines ont été comparées sur les mêmes features symboliques : **Logistic Regression**, **SVM (RBF)** et **Random Forest**. L’objectif n’est pas de chercher immédiatement le modèle le plus complexe possible, mais d’évaluer proprement la faisabilité du problème."
    )

    comparison_df = build_model_comparison_chart_dataframe(metrics_df) if not metrics_df.empty else pd.DataFrame()

    model_left, model_right = st.columns([1.05, 1])
    with model_left:
        if metrics_df.empty:
            st.info("Aucun fichier de métriques trouvé pour l’instant. Lance `python scripts/main.py` pour les régénérer.")
        else:
            display_df = build_model_display_dataframe(metrics_df)
            st.dataframe(display_df, width="stretch", hide_index=True)
            st.markdown(
                "**Pourquoi suivre le macro F1 ?** Parce qu’il donne le même poids aux classes `beginner`, `intermediate` et `advanced`, ce qui évite qu’un bon score global masque un mauvais comportement sur la classe rare `advanced`."
            )
    with model_right:
        if comparison_df.empty:
            st.info("Le graphique de comparaison apparaîtra dès que les métriques seront disponibles.")
        else:
            comparison_fig = build_model_comparison_figure(comparison_df)
            st.plotly_chart(comparison_fig, width="stretch")
            st.caption(
                "Ce plot compare directement les performances sur le test final et la validation croisée. Il permet de voir à la fois le niveau de performance et la cohérence générale du comportement du modèle."
            )

    if not metrics_df.empty:
        best_row = metrics_df.iloc[0]
        st.markdown("### Résultat le plus convaincant à ce stade")
        a, b, c = st.columns(3)
        with a:
            _metric_card("Meilleure baseline", str(best_row["model_name"]), "Modèle en tête actuellement")
        with b:
            _metric_card("Macro F1 holdout", f"{float(best_row['macro_f1']):.4f}", "Score final sur le jeu de test")
        with c:
            _metric_card("Macro F1 CV", f"{float(best_row['cv_macro_f1_mean']):.4f}", "Score moyen en validation croisée")

    st.markdown("### Lecture produit avant la matrice de confusion")
    st.write(
        "Avant même de regarder la matrice de confusion, il faut traduire les erreurs en conséquences concrètes. Une erreur de niveau ne crée pas seulement un mauvais score : elle crée une mauvaise recommandation, donc potentiellement une frustration utilisateur ou une perte de confiance dans la plateforme."
    )
    business_df = build_business_impact_dataframe()
    st.dataframe(business_df, width="stretch", hide_index=True)

    impact_left, impact_right = st.columns([1.05, 1])
    with impact_left:
        st.markdown("### Quelles features pèsent le plus dans la décision ?")
        st.write(
            "Sur la meilleure baseline actuelle, on peut déjà quantifier quelles variables comptent le plus. Ici, on lit une importance de features issue de la **Random Forest** : cela ne donne pas une causalité absolue, mais cela montre quels signaux structurent le plus la décision du modèle."
        )
        feature_importance_df = build_feature_importance_dataframe(top_n=8)
        st.dataframe(feature_importance_df, width="stretch", hide_index=True)
        if "book_code" in set(feature_importance_df["Feature"]):
            st.caption(
                "La présence de `book_code` parmi les variables importantes rappelle qu’une partie du signal est liée à l’organisation pédagogique interne de Mikrokosmos. C’est utile pour le prototype, mais c’est aussi une limite de généralisation hors de ce corpus."
            )
    with impact_right:
        feature_importance_fig = build_feature_importance_figure(build_feature_importance_dataframe(top_n=8))
        st.plotly_chart(feature_importance_fig, width="stretch")
        st.caption(
            "Ces importances montrent surtout quels signaux symboliques sont les plus utilisés par la Random Forest dans ce prototype : densité de notes, rythme, registre, accords ou contexte tonal."
        )

    best_left, best_right = st.columns([1.05, 1])
    with best_left:
        st.markdown("### Lecture du meilleur modèle")
        st.write(
            "Le meilleur modèle actuel est la **Random Forest**. Ce résultat ne prouve pas une généralisation définitive, mais il montre qu’un signal exploitable ressort déjà des partitions malgré la petite taille du corpus."
        )
        optuna_summary = load_optuna_summary()
        if optuna_summary is not None:
            st.write(
                f"Un tuning Optuna complémentaire a également trouvé une configuration Random Forest atteignant **{float(optuna_summary['best_value_macro_f1']):.4f}** de macro F1 moyenne en validation croisée."
            )
    with best_right:
        _show_plot(
            context["plot_paths"]["best_model"],
            "Meilleur modèle — matrice de confusion de la Random Forest sur le holdout set",
        )


def _render_demo_part(context: dict[str, object]) -> None:
    progress = get_section_progress_labels()
    _section_header(
        "3. Démo et usage réel",
        "Cette dernière partie montre comment le meilleur modèle pourrait être utilisé dans un contexte réel, tout en rappelant clairement les limites actuelles du prototype.",
        progress["demo"],
    )

    demo_left, demo_right = st.columns([1.05, 1])
    with demo_left:
        st.markdown("### Scénario d’usage réaliste")
        st.write(
            "Dans un usage réel, l’idée serait de proposer un assistant capable d’analyser une nouvelle partition MusicXML et de renvoyer une estimation de difficulté utilisable dans un cadre pédagogique ou de recommandation."
        )
        for index, step in enumerate(context["demo_steps"], start=1):
            st.markdown(f"**Étape {index}.** {step}.")
        st.markdown("### Comment interpréter cette démo honnêtement")
        st.write(
            "Le modèle a été entraîné sur un corpus **très homogène**, centré sur Mikrokosmos de Bartók. Autrement dit, on n’est pas en train de prouver une généralisation à tout le répertoire piano : on regarde surtout **ce que le pipeline fait quand on lui donne une nouvelle partition**, y compris si elle sort un peu du cadre habituel."
        )
        st.markdown("### Ce que CIPI pourrait apporter concrètement")
        for benefit in context["cipi_benefits"]:
            st.markdown(f"- {benefit}")
    with demo_right:
        st.markdown(
            """
            <div class="callout-card soft">
                <div class="callout-title">Ce que le prototype fait déjà</div>
                <div class="callout-text">Le projet dispose déjà d’une chaîne complète : parsing MusicXML, construction des features, entraînement de plusieurs modèles et comparaison de leurs résultats.</div>
            </div>
            <div class="callout-card soft">
                <div class="callout-title">Ce qu’il ne fait pas encore</div>
                <div class="callout-text">La version actuelle n’est pas encore un produit final branché sur un flux d’entrée utilisateur temps réel. Elle sert surtout à valider la démarche, la qualité des features et la pertinence d’un futur passage à plus grande échelle.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Limites du prototype")
        for limitation in context["limitations"]:
            st.markdown(f"- {limitation}")

    st.markdown("### Démo MusicXML : que se passe-t-il si on teste quand même une nouvelle partition ?")
    st.write(
        "Cette démo n’est pas une promesse produit finale. Elle sert à montrer que, même avec un entraînement sur un corpus homogène, on peut déjà injecter un **fichier MusicXML**, extraire ses features et obtenir une estimation de difficulté lisible."
    )
    uploaded_file = st.file_uploader(
        "Téléverser une partition MusicXML (.xml, .musicxml ou .mxl)",
        type=["xml", "musicxml", "mxl"],
        help="L’idée ici est de tester le comportement du pipeline sur une nouvelle partition, pas de prétendre que la généralisation est déjà acquise.",
    )

    if uploaded_file is None:
        st.info("Ajoute un fichier MusicXML pour lancer la démonstration complète du pipeline.")
        return

    try:
        prediction_payload = predict_musicxml_bytes(uploaded_file.getvalue(), filename=uploaded_file.name)
    except Exception as exc:  # pragma: no cover - defensive UI path
        st.error(f"Impossible d’analyser ce fichier MusicXML : {exc}")
        return

    st.success(
        f"Prédiction obtenue pour **{prediction_payload['source_filename']}** : **{prediction_payload['predicted_label']}** (confiance max ≈ {prediction_payload['top_probability']:.1%})."
    )

    result_left, result_right = st.columns([1.05, 1])
    with result_left:
        st.markdown("### Ce que renvoie le modèle")
        st.write(
            "La sortie doit être lue comme une **estimation exploratoire**. Si le résultat semble cohérent sur une partition hors entraînement, c’est encourageant pour la faisabilité ; si le résultat paraît étrange, cela rappelle précisément pourquoi un corpus plus large reste indispensable."
        )
        st.dataframe(prediction_payload["probability_frame"], width="stretch", hide_index=True)
        st.markdown("### Valeurs observées sur quelques features importantes")
        st.dataframe(prediction_payload["feature_frame"], width="stretch", hide_index=True)
    with result_right:
        probability_fig = build_probability_figure(prediction_payload["probability_frame"])
        st.plotly_chart(probability_fig, width="stretch")
        st.caption(
            "On visualise ici la répartition des probabilités entre `beginner`, `intermediate` et `advanced`, ce qui est plus informatif qu’un simple label brut."
        )


def build_app() -> None:
    context = get_project_context()
    metrics_df = load_metrics_dataframe()

    st.set_page_config(page_title="Prototype difficulté piano", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(99, 102, 241, 0.10), transparent 32%),
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.10), transparent 28%),
                linear-gradient(180deg, #f8fafc 0%, #eef2ff 48%, #f8fafc 100%);
        }
        header[data-testid="stHeader"] {
            display: none;
        }
        [data-testid="stToolbar"] {
            display: none;
        }
        .block-container {
            max-width: 1250px;
            padding-top: 1.1rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: #0f172a;
            letter-spacing: -0.02em;
        }
        h1 {
            font-size: 3.15rem !important;
            line-height: 0.98 !important;
            margin-bottom: 0.35rem !important;
        }
        h2 {
            font-size: 2rem !important;
            margin: 0 !important;
        }
        p, li, div {
            color: #1f2937;
        }
        [data-testid="stSidebar"] {
            background: rgba(255,255,255,0.92);
            border-left: 1px solid rgba(99,102,241,0.08);
        }
        [data-testid="stSidebar"] ul {
            padding-left: 0.8rem;
        }
        .hero-shell {
            padding: 1.2rem 0 0.6rem 0;
        }
        .hero-banner {
            margin: 0.8rem 0 1.2rem 0;
            padding: 1.2rem 1.25rem;
            border-radius: 22px;
            border: 1px solid rgba(99,102,241,0.12);
            background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(224,231,255,0.88));
            box-shadow: 0 18px 42px rgba(15,23,42,0.08);
        }
        .hero-kicker {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #4338ca;
            margin-bottom: 0.45rem;
        }
        .hero-text {
            font-size: 1.03rem;
            line-height: 1.6;
            color: #172554;
            font-weight: 500;
        }
        .story-card {
            margin-top: 0.85rem;
            padding: 1rem 1.05rem;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(255,255,255,.96), rgba(248,250,252,.94));
            border: 1px solid rgba(15,23,42,.08);
            box-shadow: 0 10px 24px rgba(15,23,42,.05);
            min-height: 154px;
        }
        .story-label {
            font-size: .76rem;
            color: #4338ca;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: .42rem;
        }
        .story-text {
            color: #1f2937;
            line-height: 1.55;
            font-size: .96rem;
            font-weight: 500;
        }
        .callout-card {
            margin-bottom: 1rem;
            padding: 1rem 1.05rem;
            border-radius: 18px;
            border: 1px solid rgba(14,165,233,0.14);
            background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(240,249,255,0.92));
            box-shadow: 0 12px 28px rgba(15,23,42,0.05);
        }
        .callout-card.soft {
            border-color: rgba(99,102,241,0.12);
            background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(238,242,255,0.92));
        }
        .callout-title {
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.35rem;
        }
        .callout-text {
            color: #334155;
            line-height: 1.55;
        }
        .metric-card {
            padding: 1rem 1.1rem;
            border: 1px solid rgba(99,102,241,.12);
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(255,255,255,.99), rgba(248,250,252,.96));
            box-shadow: 0 14px 36px rgba(15,23,42,.07);
            position: relative;
            overflow: hidden;
        }
        .metric-card::before {
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #4f46e5, #0ea5e9);
        }
        .metric-label {
            font-size: .8rem;
            color: #334155;
            margin-bottom: .38rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            font-weight: 700;
        }
        .metric-value {
            font-size: 1.95rem;
            font-weight: 800;
            line-height: 1.05;
            color: #111827;
        }
        .metric-help {
            font-size: .84rem;
            color: #475569;
            margin-top: .42rem;
        }
        .section-heading-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 0.25rem;
            margin-bottom: 0.15rem;
        }
        .section-chip {
            display: inline-flex;
            align-items: center;
            gap: .35rem;
            padding: .28rem .62rem;
            border-radius: 999px;
            background: rgba(79,70,229,.10);
            color: #3730a3;
            font-size: .78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: .45rem;
        }
        .sidebar-step {
            margin: 0.3rem 0;
            padding: 0.6rem 0.7rem;
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(255,255,255,.96), rgba(238,242,255,.85));
            border: 1px solid rgba(99,102,241,0.10);
        }
        .sidebar-step strong {
            color: #0f172a;
        }
        .sidebar-step span {
            display: block;
            color: #4338ca;
            font-size: .74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: .15rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Structure")
        progress_labels = get_section_progress_labels()
        for section in get_presentation_sections():
            st.markdown(
                f"""
                <div class="sidebar-step">
                    <span>{progress_labels[section['key']]}</span>
                    <strong>{section['title']}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.caption("Lecture de haut en bas.")

    _render_header(context, metrics_df)
    st.divider()
    _render_story_part(context)
    st.divider()
    _render_models_part(metrics_df, context)
    st.divider()
    _render_demo_part(context)


if __name__ == "__main__":
    build_app()
