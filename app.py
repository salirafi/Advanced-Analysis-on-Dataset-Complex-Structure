from flask import Flask, render_template
from src.plot_ingredients import (
    plot_ingredient_network_from_export,
    plot_ingredient_leiden_graph_from_export,
    plot_leiden_community_size_bar_from_export,
)
from src.plot_nutrition import replot_exported_nutrition_figure
from src.content import *

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

_network_fig = plot_ingredient_network_from_export()
_network_fig_json = _network_fig.to_json()

_leiden_fig = plot_ingredient_leiden_graph_from_export()
_leiden_fig_json = _leiden_fig.to_json()

_leiden_comm_fig = plot_leiden_community_size_bar_from_export()
_leiden_comm_fig_json = _leiden_comm_fig.to_json()

_nutrition_fig = replot_exported_nutrition_figure(
    fig_key="nutrition_pca_landscape",
    output_dir=str(BASE_DIR / "plots" / "nutritional_landscape_outputs")
)
_nutrition_fig_json = _nutrition_fig.to_json()

@app.route("/")
def index():
    return render_template(
        "index.html",
        network_fig_json=_network_fig_json,
        leiden_fig_json=_leiden_fig_json,
        nutrition_fig_json=_nutrition_fig_json,
        leiden_hl_communities=getattr(__import__('content'), 'leiden_HL_COMMUNITIES', '—'),
        leiden_hl_nodes=getattr(__import__('content'), 'leiden_HL_NODES', '—'),
        leiden_hl_resolution=getattr(__import__('content'), 'leiden_HL_RESOLUTION', '1.0'),

        network_caption=NETWORK_CAPTION,
        network_insight_title=NETWORK_INSIGHT_TITLE,
        network_insight_subtitle=NETWORK_INSIGHT_SUBTITLE,
        network_insight_findings=NETWORK_INSIGHT_FINDINGS,
        network_insight_read=NETWORK_INSIGHT_READ,
        network_insight_method=NETWORK_INSIGHT_METHOD,
        network_hl_1_value=NETWORK_HL_1_VALUE,
        network_hl_1_label=NETWORK_HL_1_LABEL,
        network_hl_2_value=NETWORK_HL_2_VALUE,
        network_hl_2_label=NETWORK_HL_2_LABEL,
        network_hl_3_value=NETWORK_HL_3_VALUE,
        network_hl_3_label=NETWORK_HL_3_LABEL,

        leiden_caption=LEIDEN_CAPTION,
        leiden_insight_title=LEIDEN_INSIGHT_TITLE,
        leiden_insight_subtitle=LEIDEN_INSIGHT_SUBTITLE,
        leiden_insight_findings=LEIDEN_INSIGHT_FINDINGS,
        leiden_insight_what=LEIDEN_INSIGHT_WHAT,
        leiden_insight_interpret=LEIDEN_INSIGHT_INTERPRET,
        leiden_insight_read=LEIDEN_INSIGHT_READ,
        leiden_insight_method=LEIDEN_INSIGHT_METHOD,
        leiden_community_fig_json=_leiden_comm_fig_json,
    )

if __name__ == "__main__":
    app.run(debug=True)
