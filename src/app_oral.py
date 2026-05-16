"""Simplified oral Streamlit app for the piano difficulty prototype."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DEMO_MIKROKOSMOS_XML_FILE, PLOTS_DIR


SRC_DIR = Path(__file__).resolve().parent


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module `{module_name}` from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


app_module = _load_module("project_full_app_for_oral", SRC_DIR / "app.py")

CLASS_ORDER = ["beginner", "intermediate", "advanced"]
CLASS_COLORS = {
    "beginner": "#6D5EF9",
    "intermediate": "#12B5E5",
    "advanced": "#FF8A3D",
}
CLASS_LABELS_FR = {
    "beginner": "Débutant",
    "intermediate": "Intermédiaire",
    "advanced": "Avancé",
}
FEATURE_LABELS_FR = {
    "notes_played": "Notes jouées",
    "notes_total": "Éléments notés",
    "unique_pitch_count": "Notes différentes",
    "rests": "Silences",
    "book_code": "Volume Mikrokosmos",
    "measures": "Mesures",
    "chord_notes": "Notes en accords",
    "pitch_span": "Amplitude du registre",
    "pitch_std": "Variété du registre",
    "avg_duration": "Durée moyenne",
}


ORAL_PLOT_PATHS = {
    "class_distribution": str(PLOTS_DIR / "eda_class_distribution.png"),
    "notes_played": str(PLOTS_DIR / "eda_notes_played_by_class.png"),
    "notes_vs_measures": str(PLOTS_DIR / "eda_notes_vs_measures_by_class.png"),
    "model_comparison": str(PLOTS_DIR / "model_comparison_macro_f1.png"),
    "confusion_matrix": str(PLOTS_DIR / "best_model_random_forest_confusion_matrix.png"),
}


def get_oral_sections() -> list[str]:
    return [
        "Le problème",
        "Le dataset en un regard",
        "Les modèles en compétition",
        "Démo en direct",
        "Ce qu'on peut conclure",
    ]


def get_oral_context() -> dict[str, object]:
    return {
        "title": "Une partition peut-elle raconter son niveau ?",
        "subtitle": "Prototype ML de difficulté piano : partir d'une partition MusicXML, extraire des signaux musicaux, puis estimer un niveau pédagogique.",
        "hero_message": "Le résultat n'est pas un juge absolu du niveau pianistique. C'est une preuve de concept : les partitions contiennent déjà assez d'indices mesurables pour produire une estimation utile et explicable.",
        "story_beats": [
            {
                "label": "Départ",
                "text": "Un professeur ou une plateforme reçoit une partition et doit savoir rapidement à quel élève elle peut convenir.",
            },
            {
                "label": "Signal",
                "text": "Le fichier MusicXML devient une série d'indices : densité de notes, rythme, registre, accords et sauts.",
            },
            {
                "label": "Décision",
                "text": "Le modèle propose un niveau et des morceaux proches, comme aide au tri plutôt que comme verdict définitif.",
            },
        ],
    }


def build_oral_demo_payload(xml_bytes: bytes, filename: str) -> dict[str, object]:
    payload = app_module.predict_musicxml_bytes(xml_bytes, filename)
    payload["recommendations_frame"] = app_module.recommend_musicxml_bytes(xml_bytes, filename, top_k=4)
    return payload


def _format_percent(value: float) -> str:
    return f"{float(value) * 100:.1f} %".replace(".", ",")


def _localized_probability_frames(probability_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    chart_df = probability_frame.copy()
    chart_df["Classe"] = chart_df["Classe"].map(CLASS_LABELS_FR).fillna(chart_df["Classe"])

    table_df = chart_df.copy()
    table_df["Probabilité"] = table_df["Probabilité"].map(_format_percent)
    return chart_df, table_df


def _localized_feature_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    display_df = feature_frame.copy()
    display_df["Feature"] = display_df["Feature"].map(FEATURE_LABELS_FR).fillna(display_df["Feature"])
    display_df = display_df.rename(columns={"Feature": "Indice musical"})
    return display_df


def _localized_label(label: str) -> str:
    return CLASS_LABELS_FR.get(str(label), str(label))


def _render_static_table(dataframe: pd.DataFrame, *, max_rows: int | None = None) -> None:
    """Render a static HTML table instead of Streamlit's dynamic dataframe widget.

    The public LocalTunnel link can fail to fetch Streamlit's lazily-loaded
    DataFrame JavaScript chunk. A static table is enough for the oral demo and
    avoids the fragile dynamic import entirely.
    """
    display_df = dataframe.head(max_rows) if max_rows is not None else dataframe
    st.markdown(
        display_df.to_html(index=False, escape=True, classes="oral-static-table"),
        unsafe_allow_html=True,
    )


def _load_eda_dataframe() -> pd.DataFrame:
    X, y = app_module.load_feature_target_dataset()
    return pd.concat([X.reset_index(drop=True), y.rename("difficulty_label").reset_index(drop=True)], axis=1)


def _oralize_figure(fig: go.Figure, *, height: int = 420, legend_title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.82)",
        font=dict(color="#102033", size=14),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="rgba(15,23,42,0.08)",
            borderwidth=1,
            title=legend_title,
        ),
        xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.16)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.16)", zeroline=False),
    )
    return fig


def build_oral_class_distribution_figure() -> go.Figure:
    distribution_df = app_module.build_target_distribution_dataframe()
    fig = px.bar(
        distribution_df,
        x="difficulty_label",
        y="pieces",
        color="difficulty_label",
        text="pieces",
        category_orders={"difficulty_label": CLASS_ORDER},
        color_discrete_map=CLASS_COLORS,
    )
    fig.update_traces(textposition="outside", marker_line_color="#ffffff", marker_line_width=1.4)
    fig.update_layout(showlegend=True, xaxis_title="Classe", yaxis_title="Nombre de morceaux")
    return _oralize_figure(fig, height=360, legend_title="Classe")


def build_oral_notes_played_figure() -> go.Figure:
    eda_df = _load_eda_dataframe()
    fig = px.box(
        eda_df,
        x="difficulty_label",
        y="notes_played",
        color="difficulty_label",
        points="all",
        category_orders={"difficulty_label": CLASS_ORDER},
        color_discrete_map=CLASS_COLORS,
    )
    fig.update_traces(jitter=0.28, pointpos=0, marker=dict(size=5, opacity=0.45))
    fig.update_layout(xaxis_title="Classe", yaxis_title="Notes jouées", showlegend=True)
    return _oralize_figure(fig, height=390, legend_title="Classe")


def build_oral_notes_vs_measures_figure() -> go.Figure:
    eda_df = _load_eda_dataframe()
    fig = px.scatter(
        eda_df,
        x="measures",
        y="notes_played",
        color="difficulty_label",
        size="unique_pitch_count",
        hover_data=["notes_total", "rests"],
        category_orders={"difficulty_label": CLASS_ORDER},
        color_discrete_map=CLASS_COLORS,
    )
    fig.update_traces(marker=dict(opacity=0.82, line=dict(width=1, color="#ffffff")))
    fig.update_layout(xaxis_title="Nombre de mesures", yaxis_title="Notes jouées", showlegend=True)
    return _oralize_figure(fig, height=390, legend_title="Classe")


def _show_plot(path: str, caption: str, width: str | int = "stretch") -> None:
    image_path = Path(path)
    if image_path.exists():
        st.image(str(image_path), caption=caption, width=width)
    else:
        st.warning(f"Visuel manquant : {image_path.name}")


def _section_title(kicker: str, title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="oral-section-heading">
            <div class="oral-kicker">{kicker}</div>
            <h2>{title}</h2>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _story_card(label: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="oral-story-card">
            <div class="oral-story-label">{label}</div>
            <div class="oral-story-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _hero(context: dict[str, object], metrics_df) -> None:
    best_model = metrics_df.iloc[0]["model_name"] if not metrics_df.empty else "Random Forest"
    best_macro_f1 = float(metrics_df.iloc[0]["macro_f1"]) if not metrics_df.empty else 0.0

    hero_left, hero_right = st.columns([1.25, 0.95])
    with hero_left:
        st.markdown(
            f"""
            <div class="oral-hero">
                <div class="oral-kicker">Prototype ML · Difficulté piano</div>
                <h1>{context['title']}</h1>
                <p>{context['subtitle']}</p>
                <div class="oral-hero-message">{context['hero_message']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_right:
        st.markdown(
            """
            <div class="oral-side-panel">
                <div class="oral-side-panel-label">Résultat principal</div>
                <div class="oral-side-panel-title">Un signal exploitable existe déjà</div>
                <div class="oral-side-panel-text">Le pipeline MusicXML → features → modèle retrouve une structure de difficulté crédible, même avec 147 pièces. La suite consiste à tester cette intuition sur un corpus plus large.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        _story_card("Dataset", "147 morceaux solo issus de Mikrokosmos")
    with col2:
        _story_card("Meilleure baseline", str(best_model))
    with col3:
        _story_card("Macro F1 holdout", f"{best_macro_f1:.3f}")

    beats = context["story_beats"]
    beat_cols = st.columns(3)
    for col, beat in zip(beat_cols, beats):
        with col:
            _story_card(str(beat["label"]), str(beat["text"]))


def _dataset_part() -> None:
    _section_title(
        "2. Données",
        "Le dataset en un regard",
        "Ces visualisations résument la structure du corpus : répartition des niveaux, densité des partitions et relation entre longueur des pièces et volume de notes.",
    )
    st.markdown(
        "**Signal observé :** les morceaux `advanced` sont plus rares, plus longs et souvent plus denses. Ce n'est pas toute la difficulté pianistique, mais c'est déjà une structure nette dans ce corpus."
    )
    left, middle, right = st.columns(3)
    with left:
        st.plotly_chart(build_oral_class_distribution_figure(), width="stretch")
    with middle:
        st.plotly_chart(build_oral_notes_played_figure(), width="stretch")
    with right:
        st.plotly_chart(build_oral_notes_vs_measures_figure(), width="stretch")


def _models_part(metrics_df) -> None:
    _section_title(
        "3. Modèles",
        "Les modèles en compétition",
        "On compare trois baselines simples et défendables sur petit dataset. L'idée n'est pas de faire du deep learning décoratif, mais de vérifier proprement si le problème est déjà apprenable.",
    )
    comparison_df = app_module.build_model_comparison_chart_dataframe(metrics_df)
    comparison_fig = app_module.build_model_comparison_figure(comparison_df)
    comparison_fig.update_layout(
        font=dict(color="#102033", size=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.82)",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="rgba(15,23,42,0.08)",
            borderwidth=1,
            title="Métrique",
        ),
    )
    left, right = st.columns([1.0, 1.05])
    with left:
        st.plotly_chart(comparison_fig, width="stretch")
    with right:
        importance_df = app_module.build_feature_importance_dataframe(top_n=6)
        importance_fig = app_module.build_feature_importance_figure(importance_df)
        importance_fig.update_layout(
            font=dict(color="#102033", size=14),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.82)",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(importance_fig, width="stretch")
    st.markdown(
        "**Conclusion modèle :** la Random Forest ressort comme meilleure baseline, et les signaux dominants sont liés à la taille de pièce, à la densité et à la variété musicale."
    )
    st.markdown(
        "**Lecture des erreurs :** la matrice montre où le modèle confond encore les niveaux. Elle complète les métriques sans remplacer l'analyse des signaux musicaux."
    )
    matrix_left, matrix_right = st.columns([0.72, 0.28])
    with matrix_left:
        _show_plot(
            ORAL_PLOT_PATHS["confusion_matrix"],
            "Matrice de confusion du meilleur modèle",
            width=520,
        )


def _demo_part() -> None:
    _section_title(
        "4. Démo",
        "Du fichier au conseil pédagogique",
        "La démo montre le passage complet : une partition entre dans le pipeline, le modèle estime un niveau, puis le catalogue propose des morceaux proches pour rendre le résultat actionnable.",
    )
    sample_xml_path = DEMO_MIKROKOSMOS_XML_FILE
    uploaded_file = st.file_uploader(
        "Déposer un fichier .xml, .musicxml ou .mxl",
        type=["xml", "musicxml", "mxl"],
        help="La démo sert à observer le comportement du pipeline, pas à prouver une généralisation définitive.",
    )

    if uploaded_file is not None:
        filename = uploaded_file.name
        xml_bytes = uploaded_file.getvalue()
        source_label = f"Partition importée — `{filename}`"
    elif sample_xml_path.exists():
        filename = sample_xml_path.name
        xml_bytes = sample_xml_path.read_bytes()
        source_label = "Exemple intégré — Mikrokosmos n°99"
        st.caption("Aucun fichier déposé : le prototype utilise automatiquement un exemple intégré pour afficher le pipeline complet.")
    else:
        st.info("Ajoute une partition MusicXML pour lancer la démo.")
        return

    try:
        payload = build_oral_demo_payload(xml_bytes, filename)
    except Exception as exc:  # pragma: no cover - UI safety net
        st.error("Impossible d'analyser ce fichier MusicXML pour l'instant.")
        st.caption(
            "Vérifie que le fichier est une partition MusicXML valide. L'exemple intégré reste disponible pour afficher le pipeline complet."
        )
        with st.expander("Détail technique"):
            st.code(str(exc))
        return

    probability_chart_df, probability_table_df = _localized_probability_frames(payload["probability_frame"])
    feature_display_df = _localized_feature_frame(payload["feature_frame"])
    predicted_label = _localized_label(payload["predicted_label"])
    top_probability = _format_percent(payload["top_probability"])

    st.markdown(f"**Source analysée :** {source_label}")
    st.success(f"Niveau estimé : **{predicted_label}** · confiance maximale **{top_probability}**")

    left, right = st.columns([0.95, 1.05])
    with left:
        st.markdown("**Probabilités par niveau**")
        _render_static_table(probability_table_df)
        st.markdown("**Indices musicaux les plus parlants**")
        _render_static_table(feature_display_df)
    with right:
        probability_fig = app_module.build_probability_figure(probability_chart_df)
        probability_fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=40),
            yaxis=dict(range=[0, 1], tickformat=".0%", title="Probabilité"),
        )
        st.plotly_chart(
            probability_fig,
            width="stretch",
            config={"displayModeBar": False},
        )

    recommendations_df = payload.get("recommendations_frame", pd.DataFrame())
    if not recommendations_df.empty:
        display_recommendations = recommendations_df.copy()
        display_recommendations["difficulty_label"] = display_recommendations["difficulty_label"].map(_localized_label)
        display_recommendations["similarity_score"] = display_recommendations["similarity_score"].map(_format_percent)
        display_recommendations = display_recommendations.rename(
            columns={
                "piece_id": "Morceau",
                "work": "Œuvre",
                "book": "Volume",
                "difficulty_label": "Niveau",
                "henle_difficulty": "Henle",
                "similarity_score": "Similarité",
            }
        )
        st.markdown("**Morceaux proches dans le catalogue**")
        expected_columns = ["Morceau", "Œuvre", "Volume", "Niveau", "Henle", "Similarité"]
        _render_static_table(
            display_recommendations[[col for col in expected_columns if col in display_recommendations.columns]]
        )
        st.caption("La recommandation compare les morceaux du même niveau estimé sur les mêmes features musicales que le modèle.")


def _conclusion_part() -> None:
    _section_title(
        "5. Conclusion",
        "Ce qu'on peut conclure",
        "La bonne conclusion n'est pas 'le problème est résolu', mais 'le prototype montre une faisabilité crédible et justifie une montée en échelle avec un corpus plus large'.",
    )
    col1, col2 = st.columns(2)
    with col1:
        _story_card("Ce qui est validé", "Le pipeline MusicXML → features → modèle → prédiction fonctionne de bout en bout.")
        _story_card("Ce que ça raconte", "Il existe déjà un signal exploitable dans les partitions symboliques.")
    with col2:
        _story_card("Ce qui manque", "Plus de diversité musicale pour vérifier la vraie généralisation hors Mikrokosmos.")
        _story_card("Étape logique suivante", "Passer à un corpus plus large comme CIPI et raffiner les features musicales.")


def build_app() -> None:
    st.set_page_config(page_title="Oral · difficulté piano", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(109,94,249,.18), transparent 24%),
                radial-gradient(circle at 100% 0%, rgba(18,181,229,.16), transparent 22%),
                linear-gradient(180deg, #f6f7fb 0%, #edf3ff 52%, #f8fafc 100%);
        }
        header[data-testid="stHeader"], [data-testid="stToolbar"] { display:none; }
        .block-container { max-width: 1280px; padding-top: 1.15rem; padding-bottom: 2.4rem; }
        h1, h2, p, div { color: #0f172a; }
        .oral-kicker {
            text-transform: uppercase; letter-spacing: .11em; font-size: .72rem; font-weight: 900; color:#5b4cf4;
        }
        .oral-hero {
            background: linear-gradient(135deg, rgba(255,255,255,.98), rgba(230,236,255,.96));
            border:1px solid rgba(109,94,249,.16); border-radius: 32px; padding:1.45rem 1.55rem; box-shadow: 0 28px 70px rgba(50,60,110,.12);
            margin-bottom: 1rem;
        }
        .oral-hero h1 { font-size: 3.35rem !important; margin: .15rem 0 .55rem 0 !important; line-height: .96 !important; letter-spacing:-.04em; }
        .oral-hero p { font-size: 1.04rem; line-height: 1.6; color:#334155; max-width: 90%; }
        .oral-hero-message {
            margin-top:.85rem; background: linear-gradient(135deg, rgba(255,255,255,.95), rgba(245,247,255,.75)); border:1px solid rgba(15,23,42,.06);
            border-radius:20px; padding:1rem 1.05rem; color:#111827; font-weight:700; box-shadow: inset 0 1px 0 rgba(255,255,255,.9);
        }
        .oral-side-panel {
            height: 100%; min-height: 100%; background: linear-gradient(160deg, #0f172a 0%, #172554 55%, #0f3b68 100%);
            border-radius: 28px; padding:1.35rem 1.2rem; box-shadow: 0 30px 70px rgba(15,23,42,.22); border:1px solid rgba(255,255,255,.08);
        }
        .oral-side-panel-label { text-transform:uppercase; letter-spacing:.11em; font-size:.7rem; font-weight:900; color:#93c5fd; margin-bottom:.5rem; }
        .oral-side-panel-title { color:#f8fafc; font-size:1.3rem; line-height:1.1; font-weight:800; margin-bottom:.65rem; }
        .oral-side-panel-text { color:rgba(226,232,240,.92); line-height:1.6; font-size:.98rem; }
        .oral-story-card {
            height:100%; background: linear-gradient(180deg, rgba(255,255,255,.99), rgba(246,248,255,.95));
            border:1px solid rgba(15,23,42,.07); border-radius:22px; padding:1.05rem 1.1rem; box-shadow: 0 16px 34px rgba(15,23,42,.06);
        }
        .oral-story-label {
            font-size:.7rem; font-weight:900; text-transform:uppercase; letter-spacing:.11em; color:#5b4cf4; margin-bottom:.4rem;
        }
        .oral-story-text { font-size:1rem; line-height:1.52; color:#18212f; font-weight:600; }
        .oral-section-heading { margin-top: 1.45rem; margin-bottom: .7rem; }
        .oral-section-heading h2 { margin: .18rem 0 .34rem 0 !important; font-size: 2.15rem !important; letter-spacing:-.03em; }
        .oral-section-heading p { margin:0; color:#475569; max-width: 900px; font-size:1rem; }
        [data-testid="stSidebar"] { background: rgba(255,255,255,.86); border-left:1px solid rgba(109,94,249,.10); backdrop-filter: blur(12px); }
        [data-testid="stSidebar"] .stMarkdown strong { color:#172554; }
        [data-testid="stSidebar"] p { color:#475569; }
        table.oral-static-table {
            width: 100%; border-collapse: collapse; margin: .35rem 0 1rem 0;
            background: rgba(255,255,255,.92); border: 1px solid rgba(15,23,42,.08);
            border-radius: 16px; overflow: hidden; box-shadow: 0 12px 28px rgba(15,23,42,.05);
            font-size: .92rem;
        }
        table.oral-static-table th {
            background: rgba(237,242,255,.96); color: #172554; text-align: left;
            padding: .68rem .75rem; font-weight: 900; border-bottom: 1px solid rgba(15,23,42,.08);
        }
        table.oral-static-table td {
            padding: .62rem .75rem; color: #1e293b; border-bottom: 1px solid rgba(15,23,42,.055);
        }
        table.oral-static-table tr:last-child td { border-bottom: 0; }
        table.oral-static-table tr:nth-child(even) td { background: rgba(248,250,252,.72); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    metrics_df = app_module.load_metrics_dataframe()
    context = get_oral_context()

    with st.sidebar:
        st.markdown("## Fil narratif")
        for index, section in enumerate(get_oral_sections(), start=1):
            st.markdown(f"**{index}. {section}**")
        st.caption("Une lecture linéaire : problème → données → modèle → démonstration → conclusion.")

    _hero(context, metrics_df)
    _dataset_part()
    st.divider()
    _models_part(metrics_df)
    st.divider()
    _demo_part()
    st.divider()
    _conclusion_part()


if __name__ == "__main__":
    build_app()
