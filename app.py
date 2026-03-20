"""
analysis_app.py
---------------
Flask application for the Recipe Bloom **Analysis** page.

Serves the interactive carousel of visualisations (ingredient network,
nutritional landscape, cooking duration) at routes / and /analysis.

Run with:
    python analysis_app.py
"""

from flask import Flask, render_template
from pathlib import Path
import importlib.util
import json

# ── Local source imports ───────────────────────────────────────────────────

from src.plot_ingredients import (
    plot_ingredient_network_from_export,
    plot_ingredient_leiden_graph_from_export,
    plot_leiden_community_size_bar_from_export,
    plot_ingredient_cooccurrence_heatmap_clustered_from_export,
    plot_top_ingredient_pairs_from_export,
)
from src.plot_nutrition import replot_exported_nutrition_figure
from src.content import (
    NETWORK_CAPTION, NETWORK_INSIGHT_TITLE, NETWORK_INSIGHT_SUBTITLE,
    NETWORK_INSIGHT_FINDINGS, NETWORK_INSIGHT_READ, NETWORK_INSIGHT_TOPPAIR,
    NETWORK_INSIGHT_METHOD, NETWORK_INSIGHT_HEATMAP,
    NETWORK_HL_1_VALUE, NETWORK_HL_1_LABEL,
    NETWORK_HL_2_VALUE, NETWORK_HL_2_LABEL,
    NETWORK_HL_3_VALUE, NETWORK_HL_3_LABEL,
    LEIDEN_CAPTION, LEIDEN_INSIGHT_TITLE, LEIDEN_INSIGHT_SUBTITLE,
    LEIDEN_INSIGHT_FINDINGS, LEIDEN_INSIGHT_WHAT, LEIDEN_INSIGHT_INTERPRET,
    LEIDEN_INSIGHT_READ, LEIDEN_INSIGHT_METHOD,
    NUTRITION_CAPTION, NUTRITION_INSIGHT_TITLE, NUTRITION_INSIGHT_SUBTITLE,
    NUTRITION_INSIGHT_WHAT, NUTRITION_INSIGHT_LOADING, NUTRITION_INSIGHT_FINDINGS,
    NUTRITION_INSIGHT_INTERPRET, NUTRITION_INSIGHT_METHOD,
    NUTHEAT_CAPTION, NUTHEAT_INSIGHT_WHAT, NUTHEAT_INSIGHT_READ,
    NUTHEAT_INSIGHT_WHY, NUTHEAT_INSIGHT_FINDINGS, NUTHEAT_INSIGHT_INTERPRET,
    NUTHEAT_INSIGHT_METHOD,
    WINDROSE_CAPTION, WINDROSE_INSIGHT_TITLE, WINDROSE_INSIGHT_SUBTITLE,
    WINDROSE_INSIGHT_FINDINGS, WINDROSE_INSIGHT_METHOD,
    FEATURE_CAPTION, FEATURE_INSIGHT_SUBTITLE, FEATURE_INSIGHT_TITLE,
    FEATURE_INSIGHT_CONTEXT, FEATURE_INSIGHT_EXPECT, FEATURE_INSIGHT_METHOD,
    FEATURE_INSIGHT_FINDINGS_ORANGE, FEATURE_INSIGHT_FINDINGS_BLUE, FEATURE_INSIGHT_FINDINGS_GREEN, FEATURE_INSIGHT_FINDINGS_RED, 
    FEATURE_INSIGHT_INTERPRET, FEATURE_INSIGHT_SUMMARY,
    RELIABLE_INSIGHT_TITLE, RELIABLE_CAPTION, RELIABLE_INSIGHT_SUBTITLE,
    RELIABLE_INSIGHT_INTERPRET, RELIABLE_INSIGHT_FINDINGS, RELIABLE_INSIGHT_WHAT,
)

# ── Paths ──────────────────────────────────────────────────────────────────

BASE_DIR            = Path(__file__).resolve().parent
WINDROSE_OUTPUT_DIR = BASE_DIR / "plots" / "cooking_time_outputs"
NUTRITION_OUTPUT_DIR = BASE_DIR / "plots" / "nutritional_landscape_outputs"

FEATURE_OUTPUT_DIR = BASE_DIR / "plots" / "review_level_outputs"
FEATURE_JSON_PATH  = FEATURE_OUTPUT_DIR / "plot_feature_review_level.json"

# ── Dynamic import for plot_duration (avoids package collision) ────────────

def _load_windrose_module():
    spec = importlib.util.spec_from_file_location(
        "plot_effort_reward",
        BASE_DIR / "src" / "plot_duration.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

_windrose_module      = _load_windrose_module()
replot_exported_windrose = _windrose_module.replot_exported_windrose

# ── Pre-build all figures at startup (cached as JSON strings) ──────────────

def _load_feature_payload():
    if not FEATURE_JSON_PATH.exists():
        return {}

    with open(FEATURE_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _build_figures():
    """
    Render every Plotly figure once at startup and serialise to JSON.
    Storing raw JSON strings avoids repeated serialisation on each request.
    """
    figs = {}

    # Ingredient network figures
    figs["network"]      = plot_ingredient_network_from_export().to_json()
    figs["heatmap"]      = plot_ingredient_cooccurrence_heatmap_clustered_from_export().to_json()
    figs["top_pairs"]    = plot_top_ingredient_pairs_from_export().to_json()
    figs["leiden"]       = plot_ingredient_leiden_graph_from_export().to_json()
    figs["leiden_comm"]  = plot_leiden_community_size_bar_from_export().to_json()

    # Nutrition figures
    for key in (
        "nutrition_pca_landscape",
        "nutrition_cluster_heatmap",
        "nutrition_pca_loadings",
        "nutrition_pca_categories",
        "nutrition_cluster_categories",
    ):
        figs[key] = replot_exported_nutrition_figure(
            fig_key=key, output_dir=str(NUTRITION_OUTPUT_DIR)
        ).to_json()

    # Windrose
    figs["windrose"] = replot_exported_windrose(
        fig_key="windrose_total_time_population",
        output_dir=WINDROSE_OUTPUT_DIR,
    ).to_json()

    # Feature-importance figures (from precomputed export JSON)
    feature_payload = _load_feature_payload()
    standalone = feature_payload.get("standalone_figures", {})
    webapp_panels = feature_payload.get("webapp_panels", {})

    main_key = webapp_panels.get("feature_importance_main", "ternary_feature_role")
    figs["feature_main"] = json.dumps(standalone.get(main_key, {}))
    figs["feature_reliability"] = json.dumps(standalone.get("category_reliability", {}))

    # optional alternates for later panel switching / insight view
    figs["feature_ridge"]   = json.dumps(standalone.get("ridge_shap_distribution", {}))
    figs["feature_grouped"] = json.dumps(standalone.get("grouped_cross_model_shap", {}))
    figs["feature_rating_distribution"] = json.dumps(standalone.get("rating_distribution", {}))
    figs["feature_decomp_combined"] = json.dumps(standalone.get("decomp_combined", {}))
    figs["feature_meta"] = feature_payload.get("meta", {})

    return figs

_FIGS = _build_figures()

# ── Flask app ──────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
@app.route("/analysis")
def index():
    """Render the Analysis carousel page."""
    return render_template(
        "index.html",
        active_page="analysis",

        # ── Figure JSON ──
        network_fig_json                    = _FIGS["network"],
        leiden_fig_json                     = _FIGS["leiden"],
        nutrition_fig_json                  = _FIGS["nutrition_pca_landscape"],
        network_heatmap_fig_json            = _FIGS["heatmap"],
        nutrition_heatmap_fig_json          = _FIGS["nutrition_cluster_heatmap"],
        network_top_pairs_fig_json          = _FIGS["top_pairs"],
        nutrition_loadings_fig_json         = _FIGS["nutrition_pca_loadings"],
        nutrition_categories_fig_json       = _FIGS["nutrition_pca_categories"],
        nutrition_cluster_categories_fig_json = _FIGS["nutrition_cluster_categories"],
        windrose_fig_json                   = _FIGS["windrose"],
        leiden_community_fig_json           = _FIGS["leiden_comm"],
        feature_fig_json       = _FIGS.get("feature_main", "{}"),
        feature_reliability_fig_json = _FIGS.get("feature_reliability", "{}"),
        feature_ridge_fig_json = _FIGS.get("feature_ridge", "{}"),
        feature_grouped_fig_json = _FIGS.get("feature_grouped", "{}"),
        feature_rating_distribution_fig_json = _FIGS.get("feature_rating_distribution", "{}"),
        feature_decomp_combined_fig_json = _FIGS.get("feature_decomp_combined", "{}"),
        feature_meta = _FIGS.get("feature_meta", {}),

        # ── Network / Leiden captions & insights ──
        network_caption             = NETWORK_CAPTION,
        network_insight_title       = NETWORK_INSIGHT_TITLE,
        network_insight_subtitle    = NETWORK_INSIGHT_SUBTITLE,
        network_insight_findings    = NETWORK_INSIGHT_FINDINGS,
        network_insight_read        = NETWORK_INSIGHT_READ,
        network_insight_toppair     = NETWORK_INSIGHT_TOPPAIR,
        network_insight_method      = NETWORK_INSIGHT_METHOD,
        network_insight_heatmap     = NETWORK_INSIGHT_HEATMAP,
        network_hl_1_value          = NETWORK_HL_1_VALUE,
        network_hl_1_label          = NETWORK_HL_1_LABEL,
        network_hl_2_value          = NETWORK_HL_2_VALUE,
        network_hl_2_label          = NETWORK_HL_2_LABEL,
        network_hl_3_value          = NETWORK_HL_3_VALUE,
        network_hl_3_label          = NETWORK_HL_3_LABEL,
        leiden_caption              = LEIDEN_CAPTION,
        leiden_insight_title        = LEIDEN_INSIGHT_TITLE,
        leiden_insight_subtitle     = LEIDEN_INSIGHT_SUBTITLE,
        leiden_insight_findings     = LEIDEN_INSIGHT_FINDINGS,
        leiden_insight_what         = LEIDEN_INSIGHT_WHAT,
        leiden_insight_interpret    = LEIDEN_INSIGHT_INTERPRET,
        leiden_insight_read         = LEIDEN_INSIGHT_READ,
        leiden_insight_method       = LEIDEN_INSIGHT_METHOD,

        # ── Nutrition captions & insights ──
        nutrition_caption               = NUTRITION_CAPTION,
        nutrition_insight_title         = NUTRITION_INSIGHT_TITLE,
        nutrition_insight_subtitle      = NUTRITION_INSIGHT_SUBTITLE,
        nutrition_insight_what          = NUTRITION_INSIGHT_WHAT,
        nutrition_insight_loading       = NUTRITION_INSIGHT_LOADING,
        nutrition_insight_findings      = NUTRITION_INSIGHT_FINDINGS,
        nutrition_insight_interpret     = NUTRITION_INSIGHT_INTERPRET,
        nutrition_insight_method        = NUTRITION_INSIGHT_METHOD,
        nutheat_caption                 = NUTHEAT_CAPTION,
        nutheat_insight_title           = NUTRITION_INSIGHT_TITLE,
        nutheat_insight_subtitle        = NUTRITION_INSIGHT_SUBTITLE,
        nutheat_insight_what            = NUTHEAT_INSIGHT_WHAT,
        nutheat_insight_read            = NUTHEAT_INSIGHT_READ,
        nutheat_insight_why             = NUTHEAT_INSIGHT_WHY,
        nutheat_insight_findings        = NUTHEAT_INSIGHT_FINDINGS,
        nutheat_insight_interpret       = NUTHEAT_INSIGHT_INTERPRET,
        nutheat_insight_method          = NUTHEAT_INSIGHT_METHOD,

        # ── Windrose captions & insights ──
        windrose_caption            = WINDROSE_CAPTION,
        windrose_insight_title      = WINDROSE_INSIGHT_TITLE,
        windrose_insight_subtitle   = WINDROSE_INSIGHT_SUBTITLE,
        windrose_insight_findings   = WINDROSE_INSIGHT_FINDINGS,
        windrose_insight_method     = WINDROSE_INSIGHT_METHOD,

        # ── Feature importance captions & insights ──
        feature_caption            = FEATURE_CAPTION,
        feature_insight_title      = FEATURE_INSIGHT_TITLE,
        feature_insight_subtitle   = FEATURE_INSIGHT_SUBTITLE,
        feature_insight_context   = FEATURE_INSIGHT_CONTEXT,
        feature_insight_expect     = FEATURE_INSIGHT_EXPECT,
        feature_insight_method     = FEATURE_INSIGHT_METHOD,
        feature_insight_findings_red     = FEATURE_INSIGHT_FINDINGS_RED,
        feature_insight_findings_green     = FEATURE_INSIGHT_FINDINGS_GREEN,
        feature_insight_findings_blue     = FEATURE_INSIGHT_FINDINGS_BLUE,
        feature_insight_findings_orange     = FEATURE_INSIGHT_FINDINGS_ORANGE,
        feature_insight_interpret     = FEATURE_INSIGHT_INTERPRET,
        feature_insight_summary     = FEATURE_INSIGHT_SUMMARY,
        reliable_caption    = RELIABLE_CAPTION,
        reliable_insight_title    = RELIABLE_INSIGHT_TITLE,
        reliable_insight_subtitle    = RELIABLE_INSIGHT_SUBTITLE,
        reliable_insight_what    = RELIABLE_INSIGHT_WHAT,
        reliable_insight_findings    = RELIABLE_INSIGHT_FINDINGS,
        reliable_insight_interpret    = RELIABLE_INSIGHT_INTERPRET,

    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)