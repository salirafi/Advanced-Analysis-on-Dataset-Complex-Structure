#!/usr/bin/env python3
"""
Standalone script to build an ingredient co-occurrence network,
plot it with Plotly, and export lightweight plotting data for a web app.

Exports
-------
- ingredient_network_nodes.parquet
- ingredient_network_edges.parquet
- ingredient_network_meta.json
- ingredient_louvain_nodes.parquet
- ingredient_louvain_edges.parquet
- ingredient_louvain_communities.parquet
- ingredient_louvain_meta.json
- ingredient_leiden_nodes.parquet
- ingredient_leiden_edges.parquet
- ingredient_leiden_communities.parquet
- ingredient_leiden_meta.json

Optional
--------
- network_preview.html
- ingredient_network.png
- ingredient_louvain_graph.html
- ingredient_louvain_graph.png
- ingredient_leiden_graph.html
- ingredient_leiden_graph.png

Requirements
------------
pip install pandas numpy networkx plotly pyarrow kaleido scipy igraph leidenalg
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster.hierarchy import linkage, leaves_list

try:
    import igraph as ig
    import leidenalg as la
except ImportError:
    ig = None
    la = None

from src.ingredient_standardization import standardize_ingredient

BASE_DIR = Path(__file__).resolve().parents[1]

# ============================================================
# Loading / preprocessing
# ============================================================

def parse_ingredient_cell(value):
    """
    Parse one ingredient cell into a Python list.

    Handles:
    - already-a-list
    - stringified Python lists, e.g. "['salt', 'pepper']"
    - JSON-like strings
    - invalid / missing values -> empty list
    """
    if isinstance(value, list):
        return value

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []

        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    return []


def normalize_ingredient(text: str) -> Optional[str]:
    """
    Normalize ingredient strings for graph construction.

    Current normalization:
    - strip whitespace
    - lowercase
    - collapse internal spaces
    - apply ingredient standardization map
    """
    if not isinstance(text, str):
        return None

    s = " ".join(text.strip().split()).lower()
    if not s:
        return None

    s = standardize_ingredient(s)

    if s == "unknown":
        return None

    return s


def preprocess_ingredient_lists(
    df: pd.DataFrame,
    ingredients_col: str = "RecipeIngredientParts",
) -> pd.DataFrame:
    """
    Ensure ingredient column is a clean list[str] per row.
    """
    df = df.copy()

    df[ingredients_col] = df[ingredients_col].apply(parse_ingredient_cell)

    def clean_list(lst):
        cleaned = []
        for item in lst:
            norm = normalize_ingredient(item)
            if norm is not None:
                cleaned.append(norm)
        return cleaned

    df[ingredients_col] = df[ingredients_col].apply(clean_list)
    return df


def load_recipes_from_sqlite(
    db_path: str | Path,
    table_name: str = "recipes",
    recipe_col: str = "Name",
    ingredients_col: str = "RecipeIngredientParts",
    where: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load recipes from SQLite.
    """
    db_path = Path(db_path)

    query = f"SELECT {recipe_col}, {ingredients_col} FROM {table_name}"
    if where:
        query += f" WHERE {where}"

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    return preprocess_ingredient_lists(df, ingredients_col=ingredients_col)


def load_recipes_from_parquet(
    parquet_path: str | Path,
    recipe_col: str = "Name",
    ingredients_col: str = "RecipeIngredientParts",
) -> pd.DataFrame:
    """
    Load recipes from a Parquet file.
    """
    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path, columns=[recipe_col, ingredients_col])
    return preprocess_ingredient_lists(df, ingredients_col=ingredients_col)


def build_discrete_colorscale(n, base_colors="Turbo"):
    """
    Build a discrete Plotly colorscale with n distinct colors.
    """
    if n <= 0:
        return [(0.0, "rgb(0,0,0)"), (1.0, "rgb(0,0,0)")]

    palette = px.colors.sample_colorscale(
        base_colors,
        [i / (n - 1) if n > 1 else 0 for i in range(n)]
    )

    colorscale = []
    for i, c in enumerate(palette):
        colorscale.append((i / n, c))
        colorscale.append(((i + 1) / n, c))

    return colorscale


# ============================================================
# Core graph + plotting logic
# ============================================================

def scale_log(values: Iterable[float], out_min: float, out_max: float, multiplier: float = 1.5):
    """
    Log-scale values into [out_min, out_max], optionally multiplied at the end.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = np.log1p(arr)

    if arr.size == 0:
        return np.array([], dtype=float)

    if np.allclose(arr.min(), arr.max()):
        return np.full_like(arr, (out_min + out_max) / 2, dtype=float)

    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return (out_min + arr * (out_max - out_min)) * multiplier


GRAPH_STYLE = {
    "height": 1000,
    "width": 1000,
    "margin": dict(l=0, r=0, t=0, b=10),
    "plot_bgcolor": "white",
    "paper_bgcolor": "white",
    "node_size_min": 18,
    "node_size_max": 55,
    "edge_width_min": 0.8,
    "edge_width_max": 5.0,
    "text_font_size": 10,
    "text_font_family": "Arial Black",
    "text_font_color": "black",
    "node_outline_width": 1.0,
    "node_outline_color": "black",
    "node_opacity": 0.95,
    "edge_color": "rgba(120,120,120,0.30)",
}

GRAPH_DEFAULTS = {
    "top_n_ingredients": 100,
    "min_edge_weight": 1,
    "min_node_degree": 1,
    "max_nodes": None,
    "layout_seed": 42,
    "layout_k": 1.5,
    "capitalize_labels": True,
}


def _build_filtered_ingredient_graph(
    df,
    recipe_col="Name",
    ingredients_col="RecipeIngredientParts",
    top_n_ingredients=50,
    min_edge_weight=10,
    min_node_degree=1,
    max_nodes=None,
):
    """
    Shared graph-building pipeline used by the original network,
    Louvain network, and Leiden network.
    """
    ingredient_freq = Counter()

    for _, row in df[[recipe_col, ingredients_col]].dropna().iterrows():
        ingredients = row[ingredients_col]
        if not isinstance(ingredients, list):
            continue

        ingredients_unique = sorted(
            set(i for i in ingredients if isinstance(i, str) and i.strip())
        )
        ingredient_freq.update(ingredients_unique)

    if top_n_ingredients is not None:
        allowed_ingredients = {
            ingredient
            for ingredient, _ in ingredient_freq.most_common(top_n_ingredients)
        }
    else:
        allowed_ingredients = None

    pair_freq = Counter()
    filtered_ingredient_freq = Counter()

    for _, row in df[[recipe_col, ingredients_col]].dropna().iterrows():
        ingredients = row[ingredients_col]
        if not isinstance(ingredients, list):
            continue

        ingredients_unique = sorted(
            set(i for i in ingredients if isinstance(i, str) and i.strip())
        )

        if allowed_ingredients is not None:
            ingredients_unique = [i for i in ingredients_unique if i in allowed_ingredients]

        filtered_ingredient_freq.update(ingredients_unique)

        for pair in itertools.combinations(ingredients_unique, 2):
            pair_freq[pair] += 1

    edge_rows = [
        {"source": a, "target": b, "weight": w}
        for (a, b), w in pair_freq.items()
        if w >= min_edge_weight
    ]
    edge_df = pd.DataFrame(edge_rows)

    if edge_df.empty:
        raise ValueError(
            "No edges remain after filtering. "
            "Try lowering min_edge_weight or increasing top_n_ingredients."
        )

    G = nx.Graph()
    for _, row in edge_df.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["weight"])

    for node in G.nodes():
        G.nodes[node]["recipe_count"] = filtered_ingredient_freq[node]

    low_degree_nodes = [n for n, d in dict(G.degree()).items() if d < min_node_degree]
    G.remove_nodes_from(low_degree_nodes)

    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    if G.number_of_nodes() == 0:
        raise ValueError(
            "No nodes remain after filtering. "
            "Try lowering min_edge_weight/min_node_degree."
        )

    if max_nodes is not None and G.number_of_nodes() > max_nodes:
        weighted_degree_tmp = dict(G.degree(weight="weight"))
        top_nodes = sorted(
            weighted_degree_tmp,
            key=weighted_degree_tmp.get,
            reverse=True,
        )[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    edge_rows_filtered = []
    for u, v, data in G.edges(data=True):
        edge_rows_filtered.append({
            "source": u,
            "target": v,
            "weight": int(data["weight"]),
        })
    edge_df = pd.DataFrame(edge_rows_filtered)

    weighted_degree = dict(G.degree(weight="weight"))
    simple_degree = dict(G.degree())

    node_rows = []
    for node in G.nodes():
        node_rows.append({
            "ingredient": node,
            "degree": int(simple_degree[node]),
            "weighted_degree": float(weighted_degree[node]),
            "recipe_count": int(G.nodes[node]["recipe_count"]),
        })

    node_df = (
        pd.DataFrame(node_rows)
        .sort_values("weighted_degree", ascending=False)
        .reset_index(drop=True)
    )

    return G, edge_df, node_df, weighted_degree, simple_degree


def _compute_shared_network_geometry(
    G,
    edge_df,
    weighted_degree,
    layout_seed=42,
    layout_k=0.35,
):
    """
    Shared layout + node/edge scaling for all graph types.
    """
    pos = nx.spring_layout(G, seed=layout_seed, k=layout_k, weight="weight")

    recipe_counts = [G.nodes[n]["recipe_count"] for n in G.nodes()]
    node_sizes_scaled = scale_log(
        recipe_counts,
        GRAPH_STYLE["node_size_min"],
        GRAPH_STYLE["node_size_max"],
        multiplier=1.0,
    )

    edge_weights = edge_df["weight"].to_numpy(dtype=float)
    edge_widths_scaled = scale_log(
        edge_weights,
        GRAPH_STYLE["edge_width_min"],
        GRAPH_STYLE["edge_width_max"],
        multiplier=0.5,
    )

    edge_width_map = {
        (row["source"], row["target"]): width
        for (_, row), width in zip(edge_df.iterrows(), edge_widths_scaled)
    }
    edge_width_map.update({
        (row["target"], row["source"]): width
        for (_, row), width in zip(edge_df.iterrows(), edge_widths_scaled)
    })

    wdeg_values = np.array([weighted_degree[n] for n in G.nodes()], dtype=float)
    wdeg_log = np.log1p(wdeg_values)

    if len(wdeg_log) > 1:
        lo = np.percentile(wdeg_log, 5)
        hi = np.percentile(wdeg_log, 95)
        wdeg_log = np.clip(wdeg_log, lo, hi)

    return pos, node_sizes_scaled, edge_width_map, wdeg_log


def _make_edge_traces(
    G,
    pos,
    edge_width_map,
    capitalize_labels=True,
    edge_color_map=None,
):
    """
    Build edge traces with consistent hover and width handling.
    """
    edge_traces = []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = data["weight"]

        u_label = u.title() if capitalize_labels else u
        v_label = v.title() if capitalize_labels else v

        color = (
            edge_color_map[(u, v)]
            if edge_color_map is not None
            else GRAPH_STYLE["edge_color"]
        )

        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(
                    width=float(edge_width_map[(u, v)]),
                    color=color,
                ),
                hoverinfo="text",
                text=f"{u_label} ↔ {v_label}<br>Co-occurrence: {int(w)}",
                showlegend=False,
            )
        )

    return edge_traces


def _make_graph_layout(title):
    return dict(
        showlegend=False,
        hovermode="closest",
        margin=GRAPH_STYLE["margin"],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor=GRAPH_STYLE["plot_bgcolor"],
        paper_bgcolor=GRAPH_STYLE["paper_bgcolor"],
    )


def _build_igraph_from_networkx(G: nx.Graph):
    """
    Convert NetworkX graph to igraph while preserving:
    - vertex names
    - edge weights
    """
    if ig is None or la is None:
        raise ImportError(
            "Leiden dependencies are missing. Install with:\n"
            "pip install igraph leidenalg"
        )

    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    weights = [float(G[u][v]["weight"]) for u, v in G.edges()]

    g = ig.Graph()
    g.add_vertices(len(nodes))
    g.vs["name"] = nodes

    if edges:
        g.add_edges(edges)
        g.es["weight"] = weights

    return g


def _detect_leiden_communities(
    G: nx.Graph,
    resolution: float = 1.0,
    seed: int = 42,
):
    """
    Detect Leiden communities and return:
    - communities: list[set[str]]
    - node_to_community: dict[str, int]
    """
    g = _build_igraph_from_networkx(G)

    np.random.seed(seed)

    partition = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights=g.es["weight"] if g.ecount() > 0 else None,
        resolution_parameter=resolution,
        seed=seed,
    )

    communities = []
    node_to_community = {}

    for cid, comm in enumerate(partition):
        comm_names = {g.vs[idx]["name"] for idx in comm}
        communities.append(comm_names)
        for node in comm_names:
            node_to_community[node] = cid

    return communities, node_to_community


def _plot_partitioned_ingredient_graph(
    G,
    edge_df,
    weighted_degree,
    simple_degree,
    node_to_community,
    layout_seed,
    layout_k,
    title,
    capitalize_labels,
    colorscale,
    chart_type,
    method_name,
    resolution_value,
    resolution_key,
):
    """
    Shared plotting/export logic for partition-based community graphs
    such as Louvain and Leiden.
    """
    pos, node_sizes_scaled, edge_width_map, _ = _compute_shared_network_geometry(
        G=G,
        edge_df=edge_df,
        weighted_degree=weighted_degree,
        layout_seed=layout_seed,
        layout_k=layout_k,
    )

    edge_color_map = {}
    for u, v in G.edges():
        same_comm = node_to_community[u] == node_to_community[v]
        color = "rgba(120,120,120,0.30)" if same_comm else "rgba(170,170,170,0.15)"
        edge_color_map[(u, v)] = color
        edge_color_map[(v, u)] = color

    edge_traces = _make_edge_traces(
        G=G,
        pos=pos,
        edge_width_map=edge_width_map,
        capitalize_labels=capitalize_labels,
        edge_color_map=edge_color_map,
    )

    node_x, node_y = [], []
    node_text, node_size, node_color, node_label = [], [], [], []
    export_node_rows = []

    for i, node in enumerate(G.nodes()):
        x, y = pos[node]
        deg = simple_degree[node]
        wdeg = weighted_degree[node]
        rcount = G.nodes[node]["recipe_count"]
        cid = node_to_community[node]
        label = node.title() if capitalize_labels else node

        hovertext = (
            f"<b>{label}</b><br>"
            f"Community: {cid}<br>"
            f"Recipes containing ingredient: {rcount}<br>"
            f"Connected ingredients: {deg}<br>"
            f"Weighted degree: {wdeg}"
        )

        node_x.append(x)
        node_y.append(y)
        node_text.append(hovertext)
        node_size.append(float(node_sizes_scaled[i]))
        node_color.append(int(cid))
        node_label.append(label)

        export_node_rows.append({
            "ingredient": node,
            "label": label,
            "x": float(x),
            "y": float(y),
            "degree": int(deg),
            "weighted_degree": float(wdeg),
            "recipe_count": int(rcount),
            "community": int(cid),
            "node_size": float(node_sizes_scaled[i]),
            "node_color": int(cid),
            "hovertext": hovertext,
        })

    export_node_df = pd.DataFrame(export_node_rows)

    n_communities = int(export_node_df["community"].nunique())
    discrete_colorscale = build_discrete_colorscale(n_communities, colorscale)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_label,
        textposition="middle center",
        textfont=dict(
            size=GRAPH_STYLE["text_font_size"],
            color=GRAPH_STYLE["text_font_color"],
            family=GRAPH_STYLE["text_font_family"],
        ),
        hoverinfo="text",
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale=discrete_colorscale,
            showscale=True,
            colorbar=dict(
                title="Community",
                orientation="h",
                y=-0.15,
                x=0.5,
                xanchor="center",
                yanchor="top",
                len=0.6,
            ),
            line=dict(
                width=GRAPH_STYLE["node_outline_width"],
                color=GRAPH_STYLE["node_outline_color"],
            ),
            opacity=GRAPH_STYLE["node_opacity"],
            cmin=-0.5,
            cmax=n_communities - 0.5,
        ),
        showlegend=False,
    )

    export_edge_rows = []
    for u, v, data in G.edges(data=True):
        export_edge_rows.append({
            "source": u,
            "target": v,
            "weight": int(data["weight"]),
            "edge_width": float(edge_width_map[(u, v)]),
            "same_community": bool(node_to_community[u] == node_to_community[v]),
            "edge_color": edge_color_map[(u, v)],
        })
    export_edge_df = pd.DataFrame(export_edge_rows)

    export_community_df = (
        export_node_df.groupby("community")
        .agg(
            n_ingredients=("ingredient", "count"),
            total_recipe_count=("recipe_count", "sum"),
            total_weighted_degree=("weighted_degree", "sum"),
        )
        .sort_values("n_ingredients", ascending=False)
        .reset_index()
    )

    export_meta = {
        "chart_type": chart_type,
        "title": title,
        "method": method_name,
        "colorscale": discrete_colorscale,
        "base_colorscale": colorscale,
        "colorbar_title": "Community",
        "node_color_min": -0.5,
        "node_color_max": n_communities - 0.5,
        "n_communities": n_communities,
        "layout_seed": int(layout_seed),
        "layout_k": float(layout_k),
        "height": GRAPH_STYLE["height"],
        "width": GRAPH_STYLE["width"],
        "node_outline_color": GRAPH_STYLE["node_outline_color"],
        "node_outline_width": GRAPH_STYLE["node_outline_width"],
        "text_font_color": GRAPH_STYLE["text_font_color"],
        "text_font_size": GRAPH_STYLE["text_font_size"],
        "text_font_family": GRAPH_STYLE["text_font_family"],
        "node_opacity": GRAPH_STYLE["node_opacity"],
        resolution_key: float(resolution_value),
    }

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(**_make_graph_layout(title))

    return fig, export_edge_df, export_node_df, export_community_df, export_meta


def plot_ingredient_network(
    df,
    recipe_col="Name",
    ingredients_col="RecipeIngredientParts",
    top_n_ingredients=GRAPH_DEFAULTS["top_n_ingredients"],
    min_edge_weight=GRAPH_DEFAULTS["min_edge_weight"],
    min_node_degree=GRAPH_DEFAULTS["min_node_degree"],
    max_nodes=GRAPH_DEFAULTS["max_nodes"],
    layout_seed=GRAPH_DEFAULTS["layout_seed"],
    layout_k=GRAPH_DEFAULTS["layout_k"],
    title="Ingredient Co-occurrence Network",
    capitalize_labels=GRAPH_DEFAULTS["capitalize_labels"],
    colorscale="Blues",
):
    """
    Build and plot the original ingredient co-occurrence network.
    Node color = log(weighted degree).
    """
    G, edge_df, node_df, weighted_degree, simple_degree = _build_filtered_ingredient_graph(
        df=df,
        recipe_col=recipe_col,
        ingredients_col=ingredients_col,
        top_n_ingredients=top_n_ingredients,
        min_edge_weight=min_edge_weight,
        min_node_degree=min_node_degree,
        max_nodes=max_nodes,
    )

    pos, node_sizes_scaled, edge_width_map, wdeg_log = _compute_shared_network_geometry(
        G=G,
        edge_df=edge_df,
        weighted_degree=weighted_degree,
        layout_seed=layout_seed,
        layout_k=layout_k,
    )

    edge_traces = _make_edge_traces(
        G=G,
        pos=pos,
        edge_width_map=edge_width_map,
        capitalize_labels=capitalize_labels,
        edge_color_map=None,
    )

    node_x, node_y = [], []
    node_text, node_size, node_color, node_label = [], [], [], []
    export_node_rows = []

    for i, node in enumerate(G.nodes()):
        x, y = pos[node]
        deg = simple_degree[node]
        wdeg = weighted_degree[node]
        rcount = G.nodes[node]["recipe_count"]
        label = node.title() if capitalize_labels else node

        hovertext = (
            f"<b>{label}</b><br>"
            f"Recipes containing ingredient: {rcount}<br>"
            f"Connected ingredients: {deg}<br>"
            f"Weighted degree: {wdeg}"
        )

        node_x.append(x)
        node_y.append(y)
        node_text.append(hovertext)
        node_size.append(float(node_sizes_scaled[i]))
        node_color.append(float(wdeg_log[i]))
        node_label.append(label)

        export_node_rows.append({
            "ingredient": node,
            "label": label,
            "x": float(x),
            "y": float(y),
            "degree": int(deg),
            "weighted_degree": float(wdeg),
            "recipe_count": int(rcount),
            "node_size": float(node_sizes_scaled[i]),
            "node_color": float(wdeg_log[i]),
            "hovertext": hovertext,
        })

    export_node_df = pd.DataFrame(export_node_rows)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_label,
        textposition="middle center",
        textfont=dict(
            size=GRAPH_STYLE["text_font_size"],
            color=GRAPH_STYLE["text_font_color"],
            family=GRAPH_STYLE["text_font_family"],
        ),
        hoverinfo="text",
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title="Log Weighted Degree",
                orientation="h",
                y=-0.15,
                x=0.5,
                xanchor="center",
                yanchor="top",
                len=0.6,
            ),
            line=dict(
                width=GRAPH_STYLE["node_outline_width"],
                color=GRAPH_STYLE["node_outline_color"],
            ),
            opacity=GRAPH_STYLE["node_opacity"],
            cmin=float(np.min(wdeg_log)),
            cmax=float(np.max(wdeg_log)),
        ),
        showlegend=False,
    )

    export_edge_rows = []
    for u, v, data in G.edges(data=True):
        export_edge_rows.append({
            "source": u,
            "target": v,
            "weight": int(data["weight"]),
            "edge_width": float(edge_width_map[(u, v)]),
            "edge_color": GRAPH_STYLE["edge_color"],
        })
    export_edge_df = pd.DataFrame(export_edge_rows)

    export_meta = {
        "chart_type": "ingredient_network",
        "title": title,
        "colorscale": colorscale,
        "colorbar_title": "Log weighted degree",
        "node_color_min": float(np.min(wdeg_log)),
        "node_color_max": float(np.max(wdeg_log)),
        "layout_seed": int(layout_seed),
        "layout_k": float(layout_k),
        "top_n_ingredients": None if top_n_ingredients is None else int(top_n_ingredients),
        "min_edge_weight": int(min_edge_weight),
        "min_node_degree": int(min_node_degree),
        "max_nodes": None if max_nodes is None else int(max_nodes),
        "height": GRAPH_STYLE["height"],
        "width": GRAPH_STYLE["width"],
        "node_outline_color": GRAPH_STYLE["node_outline_color"],
        "node_outline_width": GRAPH_STYLE["node_outline_width"],
        "text_font_color": GRAPH_STYLE["text_font_color"],
        "text_font_size": GRAPH_STYLE["text_font_size"],
        "text_font_family": GRAPH_STYLE["text_font_family"],
        "node_opacity": GRAPH_STYLE["node_opacity"],
    }

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(**_make_graph_layout(title))

    return G, fig, export_edge_df, export_node_df, export_meta


def plot_ingredient_louvain_graph(
    df,
    recipe_col="Name",
    ingredients_col="RecipeIngredientParts",
    top_n_ingredients=GRAPH_DEFAULTS["top_n_ingredients"],
    min_edge_weight=GRAPH_DEFAULTS["min_edge_weight"],
    min_node_degree=GRAPH_DEFAULTS["min_node_degree"],
    max_nodes=GRAPH_DEFAULTS["max_nodes"],
    layout_seed=GRAPH_DEFAULTS["layout_seed"],
    layout_k=GRAPH_DEFAULTS["layout_k"],
    title="Ingredient Community Graph (Louvain)",
    capitalize_labels=GRAPH_DEFAULTS["capitalize_labels"],
    colorscale="Turbo",
    louvain_resolution=1.0,
):
    """
    Build and plot the Louvain ingredient network.
    Node color = community ID.
    Everything else follows the same shared UI/settings as the original graph.
    """
    G, edge_df, node_df, weighted_degree, simple_degree = _build_filtered_ingredient_graph(
        df=df,
        recipe_col=recipe_col,
        ingredients_col=ingredients_col,
        top_n_ingredients=top_n_ingredients,
        min_edge_weight=min_edge_weight,
        min_node_degree=min_node_degree,
        max_nodes=max_nodes,
    )

    communities = nx.community.louvain_communities(
        G,
        weight="weight",
        resolution=louvain_resolution,
        seed=layout_seed,
    )

    node_to_community = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            node_to_community[node] = cid

    fig, export_edge_df, export_node_df, export_community_df, export_meta = _plot_partitioned_ingredient_graph(
        G=G,
        edge_df=edge_df,
        weighted_degree=weighted_degree,
        simple_degree=simple_degree,
        node_to_community=node_to_community,
        layout_seed=layout_seed,
        layout_k=layout_k,
        title=title,
        capitalize_labels=capitalize_labels,
        colorscale=colorscale,
        chart_type="ingredient_louvain_graph",
        method_name="louvain",
        resolution_value=louvain_resolution,
        resolution_key="louvain_resolution",
    )

    export_meta.update({
        "top_n_ingredients": None if top_n_ingredients is None else int(top_n_ingredients),
        "min_edge_weight": int(min_edge_weight),
        "min_node_degree": int(min_node_degree),
        "max_nodes": None if max_nodes is None else int(max_nodes),
    })

    return G, fig, export_edge_df, export_node_df, export_community_df, export_meta


def plot_ingredient_leiden_graph(
    df,
    recipe_col="Name",
    ingredients_col="RecipeIngredientParts",
    top_n_ingredients=GRAPH_DEFAULTS["top_n_ingredients"],
    min_edge_weight=GRAPH_DEFAULTS["min_edge_weight"],
    min_node_degree=GRAPH_DEFAULTS["min_node_degree"],
    max_nodes=GRAPH_DEFAULTS["max_nodes"],
    layout_seed=GRAPH_DEFAULTS["layout_seed"],
    layout_k=GRAPH_DEFAULTS["layout_k"],
    title="Ingredient Community Graph (Leiden)",
    capitalize_labels=GRAPH_DEFAULTS["capitalize_labels"],
    colorscale="Turbo",
    leiden_resolution=1.0,
):
    """
    Build and plot the Leiden ingredient network.
    Node color = community ID.
    Everything else follows the same shared UI/settings as the original graph.
    """
    G, edge_df, node_df, weighted_degree, simple_degree = _build_filtered_ingredient_graph(
        df=df,
        recipe_col=recipe_col,
        ingredients_col=ingredients_col,
        top_n_ingredients=top_n_ingredients,
        min_edge_weight=min_edge_weight,
        min_node_degree=min_node_degree,
        max_nodes=max_nodes,
    )

    communities, node_to_community = _detect_leiden_communities(
        G,
        resolution=leiden_resolution,
        seed=layout_seed,
    )

    fig, export_edge_df, export_node_df, export_community_df, export_meta = _plot_partitioned_ingredient_graph(
        G=G,
        edge_df=edge_df,
        weighted_degree=weighted_degree,
        simple_degree=simple_degree,
        node_to_community=node_to_community,
        layout_seed=layout_seed,
        layout_k=layout_k,
        title=title,
        capitalize_labels=capitalize_labels,
        colorscale=colorscale,
        chart_type="ingredient_leiden_graph",
        method_name="leiden",
        resolution_value=leiden_resolution,
        resolution_key="leiden_resolution",
    )

    export_meta.update({
        "top_n_ingredients": None if top_n_ingredients is None else int(top_n_ingredients),
        "min_edge_weight": int(min_edge_weight),
        "min_node_degree": int(min_node_degree),
        "max_nodes": None if max_nodes is None else int(max_nodes),
    })

    return G, fig, export_edge_df, export_node_df, export_community_df, export_meta


def plot_top_ingredient_pairs(
    edge_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Top Ingredient Pair Co-occurrences",
    capitalize_labels: bool = True,
):
    """
    Plot a horizontal bar chart of the strongest ingredient pairs
    and return exportable lightweight plotting data.
    """
    if edge_df.empty:
        raise ValueError("edge_df is empty.")

    plot_df = edge_df.copy()

    if capitalize_labels:
        def make_pair_label(a, b, capitalize=True):
            vals = sorted([a, b])
            if capitalize:
                vals = [v.title() for v in vals]
            return f"{vals[0]} + {vals[1]}"

        plot_df["pair"] = [
            make_pair_label(a, b, capitalize=capitalize_labels)
            for a, b in zip(plot_df["source"], plot_df["target"])
        ]
    else:
        plot_df["pair"] = plot_df["source"] + " + " + plot_df["target"]

    plot_df = (
        plot_df.sort_values("weight", ascending=False)
        .head(top_n)
        .copy()
    )

    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    plot_df["hovertext"] = (
        "<b>" + plot_df["pair"] + "</b><br>"
        + "Recipes containing both ingredients: "
        + plot_df["weight"].astype(str)
    )

    fig = px.bar(
        plot_df,
        x="weight",
        y="pair",
        orientation="h",
        text="weight",
        title=title,
    )

    fig.update_traces(
        textposition="outside",
        hovertext=plot_df["hovertext"],
        hovertemplate="%{hovertext}<extra></extra>",
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        xaxis_title="Number of recipes containing both ingredients",
        yaxis_title="Ingredient pair",
        height=650,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    export_pair_df = plot_df[[
        "source", "target", "pair", "weight", "hovertext"
    ]].copy()

    export_meta = {
        "title": title,
        "xaxis_title": "Number of recipes containing both ingredients",
        "yaxis_title": "Ingredient pair",
        "height": 650,
        "orientation": "h",
        "chart_type": "top_ingredient_pairs",
        "top_n": int(top_n),
        "capitalize_labels": bool(capitalize_labels),
        "textposition": "outside",
        "template": "plotly_white",
    }

    return fig, export_pair_df, export_meta


def compute_ingredient_frequency_df(
    df: pd.DataFrame,
    ingredients_col: str = "RecipeIngredientParts",
) -> pd.DataFrame:
    """
    Flatten ingredient lists and compute ingredient frequency table.
    """
    all_ingredients = [
        ingredient
        for ingredients in df[ingredients_col]
        if isinstance(ingredients, list)
        for ingredient in ingredients
        if isinstance(ingredient, str) and ingredient.strip()
    ]

    ingredient_counts = Counter(all_ingredients)

    freq_df = (
        pd.DataFrame(ingredient_counts.items(), columns=["ingredient", "count"])
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    return freq_df


def plot_top_ingredient_frequencies(
    freq_df: pd.DataFrame,
    top_n: int = 500,
    title: str | None = None,
    capitalize_labels: bool = True,
    marker_color: str = "#2E86AB",
):
    """
    Plot a horizontal bar chart of the most common ingredients
    and return exportable lightweight plotting data.
    """
    if freq_df.empty:
        raise ValueError("freq_df is empty.")

    if title is None:
        title = f"Top {top_n} Most Common Ingredients"

    plot_df = freq_df.head(top_n).copy()

    if capitalize_labels:
        plot_df["label"] = plot_df["ingredient"].str.title()
    else:
        plot_df["label"] = plot_df["ingredient"]

    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    plot_df["hovertext"] = (
        "<b>" + plot_df["label"] + "</b><br>"
        + "Number of recipes: "
        + plot_df["count"].astype(str)
    )

    fig = px.bar(
        plot_df,
        x="count",
        y="label",
        orientation="h",
        text="count",
        title=title,
    )

    fig.update_traces(
        textposition="outside",
        marker_color=marker_color,
        hovertext=plot_df["hovertext"],
        hovertemplate="%{hovertext}<extra></extra>",
    )

    fig.update_layout(
        xaxis_title="Number of Recipes",
        yaxis_title="Ingredient",
        title_x=0.5,
        template="plotly_white",
        height=700,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    export_freq_df = plot_df[[
        "ingredient", "label", "count", "hovertext"
    ]].copy()

    export_meta = {
        "title": title,
        "xaxis_title": "Number of Recipes",
        "yaxis_title": "Ingredient",
        "height": 700,
        "orientation": "h",
        "chart_type": "top_ingredient_frequencies",
        "top_n": int(top_n),
        "capitalize_labels": bool(capitalize_labels),
        "textposition": "outside",
        "template": "plotly_white",
        "marker_color": marker_color,
    }

    return fig, export_freq_df, export_meta


def plot_ingredient_cooccurrence_heatmap(
    df,
    ingredients_col="RecipeIngredientParts",
    top_n=30,
    colorscale="Magma",
    capitalize_labels=True,
):
    """
    Build ingredient co-occurrence heatmaps (plain + clustered),
    and return exportable data.
    """
    ingredient_freq = Counter()

    for ingredients in df[ingredients_col].dropna():
        if isinstance(ingredients, list):
            ingredients_unique = set(
                i for i in ingredients if isinstance(i, str) and i.strip()
            )
            ingredient_freq.update(ingredients_unique)

    top_ingredients = [i for i, _ in ingredient_freq.most_common(top_n)]
    top_set = set(top_ingredients)

    pair_freq = Counter()

    for ingredients in df[ingredients_col].dropna():
        if not isinstance(ingredients, list):
            continue

        ingredients_filtered = sorted(
            set(i for i in ingredients if isinstance(i, str) and i.strip()) & top_set
        )

        for a, b in itertools.combinations(ingredients_filtered, 2):
            pair_freq[(a, b)] += 1

    co_matrix = pd.DataFrame(
        0,
        index=top_ingredients,
        columns=top_ingredients,
        dtype=float,
    )

    for (a, b), w in pair_freq.items():
        co_matrix.loc[a, b] = w
        co_matrix.loc[b, a] = w

    for ing in top_ingredients:
        co_matrix.loc[ing, ing] = 0.0

    co_matrix_log = np.log1p(co_matrix)

    Z = linkage(co_matrix_log, method="average")
    order = leaves_list(Z)
    co_matrix_log_clustered = co_matrix_log.iloc[order, order]

    if capitalize_labels:
        co_matrix_display = co_matrix_log.copy()
        co_matrix_display.index = [x.title() for x in co_matrix_display.index]
        co_matrix_display.columns = [x.title() for x in co_matrix_display.columns]

        co_matrix_clustered_display = co_matrix_log_clustered.copy()
        co_matrix_clustered_display.index = [x.title() for x in co_matrix_clustered_display.index]
        co_matrix_clustered_display.columns = [x.title() for x in co_matrix_clustered_display.columns]
    else:
        co_matrix_display = co_matrix_log
        co_matrix_clustered_display = co_matrix_log_clustered

    fig = px.imshow(
        co_matrix_display,
        color_continuous_scale=colorscale,
        title=f"Ingredient Co-occurrence Heatmap (Top {top_n}, Log Scale)",
        aspect="auto",
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        xaxis_title="Ingredient",
        yaxis_title="Ingredient",
        width=900,
        height=900,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig_clustered = px.imshow(
        co_matrix_clustered_display,
        color_continuous_scale=colorscale,
        title="Ingredient Co-occurrence Heatmap (Clustered, Log Scale)",
        aspect="auto",
    )

    fig_clustered.update_layout(
        template="plotly_white",
        title_x=0.5,
        xaxis_title="Ingredient",
        yaxis_title="Ingredient",
        width=900,
        height=900,
    )
    fig_clustered.update_yaxes(scaleanchor="x", scaleratio=1)

    export_meta = {
        "chart_type": "ingredient_cooccurrence_heatmap",
        "title": f"Ingredient Co-occurrence Heatmap (Top {top_n}, Log Scale)",
        "clustered_title": "Ingredient Co-occurrence Heatmap (Clustered, Log Scale)",
        "top_n": int(top_n),
        "colorscale": colorscale,
        "capitalize_labels": bool(capitalize_labels),
        "template": "plotly_white",
        "width": 900,
        "height": 900,
        "xaxis_title": "Ingredient",
        "yaxis_title": "Ingredient",
        "scaleanchor": "x",
        "scaleratio": 1,
    }

    return fig, fig_clustered, co_matrix_log, co_matrix_log_clustered, export_meta


# ============================================================
# Export helpers
# ============================================================

def export_network_data(
    export_node_df: pd.DataFrame,
    export_edge_df: pd.DataFrame,
    export_meta: dict,
    out_dir: str | Path = "ingredient_network_export",
):
    """
    Export lightweight plotting data for the original graph.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    node_path = out_dir / "ingredient_network_nodes.parquet"
    edge_path = out_dir / "ingredient_network_edges.parquet"
    meta_path = out_dir / "ingredient_network_meta.json"

    export_node_df.to_parquet(node_path, index=False)
    export_edge_df.to_parquet(edge_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(export_meta, f, indent=2)

    return node_path, edge_path, meta_path


def export_partition_network_data(
    export_node_df: pd.DataFrame,
    export_edge_df: pd.DataFrame,
    export_community_df: pd.DataFrame,
    export_meta: dict,
    out_dir: str | Path,
    prefix: str,
):
    """
    Generic export helper for community graphs (Louvain / Leiden).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    node_path = out_dir / f"{prefix}_nodes.parquet"
    edge_path = out_dir / f"{prefix}_edges.parquet"
    community_path = out_dir / f"{prefix}_communities.parquet"
    meta_path = out_dir / f"{prefix}_meta.json"

    export_node_df.to_parquet(node_path, index=False)
    export_edge_df.to_parquet(edge_path, index=False)
    export_community_df.to_parquet(community_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(export_meta, f, indent=2)

    return node_path, edge_path, community_path, meta_path


def export_louvain_network_data(
    export_node_df: pd.DataFrame,
    export_edge_df: pd.DataFrame,
    export_community_df: pd.DataFrame,
    export_meta: dict,
    out_dir: str | Path,
):
    return export_partition_network_data(
        export_node_df=export_node_df,
        export_edge_df=export_edge_df,
        export_community_df=export_community_df,
        export_meta=export_meta,
        out_dir=out_dir,
        prefix="ingredient_louvain",
    )


def export_leiden_network_data(
    export_node_df: pd.DataFrame,
    export_edge_df: pd.DataFrame,
    export_community_df: pd.DataFrame,
    export_meta: dict,
    out_dir: str | Path,
):
    return export_partition_network_data(
        export_node_df=export_node_df,
        export_edge_df=export_edge_df,
        export_community_df=export_community_df,
        export_meta=export_meta,
        out_dir=out_dir,
        prefix="ingredient_leiden",
    )


def export_top_pairs_data(
    export_pair_df: pd.DataFrame,
    export_meta: dict,
    out_dir: str | Path,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_path = out_dir / "top_ingredient_pairs.parquet"
    meta_path = out_dir / "top_ingredient_pairs_meta.json"

    export_pair_df.to_parquet(pair_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(export_meta, f, indent=2)

    return pair_path, meta_path


def export_top_frequencies_data(
    export_freq_df: pd.DataFrame,
    export_meta: dict,
    out_dir: str | Path,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    freq_path = out_dir / "top_ingredient_frequencies.parquet"
    meta_path = out_dir / "top_ingredient_frequencies_meta.json"

    export_freq_df.to_parquet(freq_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(export_meta, f, indent=2)

    return freq_path, meta_path


def export_cooccurrence_heatmap_data(
    co_matrix: pd.DataFrame,
    co_matrix_clustered: pd.DataFrame,
    export_meta: dict,
    out_dir: str | Path,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = out_dir / "ingredient_cooccurrence_heatmap.parquet"
    clustered_path = out_dir / "ingredient_cooccurrence_heatmap_clustered.parquet"
    meta_path = out_dir / "ingredient_cooccurrence_heatmap_meta.json"

    co_matrix.to_parquet(heatmap_path)
    co_matrix_clustered.to_parquet(clustered_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(export_meta, f, indent=2)

    return heatmap_path, clustered_path, meta_path


# ============================================================
# Lightweight replot functions
# ============================================================

def plot_ingredient_network_from_export(
    node_path=BASE_DIR / "plots/ingredient_network_export/ingredient_network_nodes.parquet",
    edge_path=BASE_DIR / "plots/ingredient_network_export/ingredient_network_edges.parquet",
    meta_path=BASE_DIR / "plots/ingredient_network_export/ingredient_network_meta.json",
):
    """
    Replot the original ingredient network using exported lightweight files.
    """
    nodes = pd.read_parquet(node_path)
    edges = pd.read_parquet(edge_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    node_lookup = nodes.set_index("ingredient")

    edge_traces = []
    for _, row in edges.iterrows():
        u = row["source"]
        v = row["target"]

        x0 = node_lookup.at[u, "x"]
        y0 = node_lookup.at[u, "y"]
        x1 = node_lookup.at[v, "x"]
        y1 = node_lookup.at[v, "y"]

        u_label = node_lookup.at[u, "label"]
        v_label = node_lookup.at[v, "label"]

        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(
                    width=float(row["edge_width"]),
                    color=row.get("edge_color", GRAPH_STYLE["edge_color"]),
                ),
                hoverinfo="text",
                text=f"{u_label} ↔ {v_label}<br>Co-occurrence: {int(row['weight'])}",
                showlegend=False,
            )
        )

    node_trace = go.Scatter(
        x=nodes["x"],
        y=nodes["y"],
        mode="markers+text",
        text=nodes["label"],
        textposition="middle center",
        textfont=dict(
            size=meta.get("text_font_size", GRAPH_STYLE["text_font_size"]),
            color=meta.get("text_font_color", GRAPH_STYLE["text_font_color"]),
            family=meta.get("text_font_family", GRAPH_STYLE["text_font_family"]),
        ),
        hoverinfo="text",
        hovertext=nodes["hovertext"],
        marker=dict(
            size=nodes["node_size"],
            color=nodes["node_color"],
            colorscale=meta["colorscale"],
            showscale=True,
            colorbar=dict(
                title=meta["colorbar_title"],
                orientation="h",
                y=-0.15,
                x=0.5,
                xanchor="center",
                yanchor="top",
                len=0.6,
            ),
            line=dict(
                width=meta.get("node_outline_width", GRAPH_STYLE["node_outline_width"]),
                color=meta.get("node_outline_color", GRAPH_STYLE["node_outline_color"]),
            ),
            opacity=meta.get("node_opacity", GRAPH_STYLE["node_opacity"]),
            cmin=meta["node_color_min"],
            cmax=meta["node_color_max"],
        ),
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(**_make_graph_layout(meta["title"]))

    return fig


def _plot_partition_graph_from_export(
    node_path,
    edge_path,
    meta_path,
):
    """
    Generic replot helper for Louvain / Leiden exported graphs.
    """
    nodes = pd.read_parquet(node_path)
    edges = pd.read_parquet(edge_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    node_lookup = nodes.set_index("ingredient")

    edge_traces = []
    for _, row in edges.iterrows():
        u = row["source"]
        v = row["target"]

        x0 = node_lookup.at[u, "x"]
        y0 = node_lookup.at[u, "y"]
        x1 = node_lookup.at[v, "x"]
        y1 = node_lookup.at[v, "y"]

        u_label = node_lookup.at[u, "label"]
        v_label = node_lookup.at[v, "label"]

        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(
                    width=float(row["edge_width"]),
                    color=row["edge_color"],
                ),
                hoverinfo="text",
                text=f"{u_label} ↔ {v_label}<br>Co-occurrence: {int(row['weight'])}",
                showlegend=False,
            )
        )

    node_trace = go.Scatter(
        x=nodes["x"],
        y=nodes["y"],
        mode="markers+text",
        text=nodes["label"],
        textposition="middle center",
        textfont=dict(
            size=meta.get("text_font_size", GRAPH_STYLE["text_font_size"]),
            color=meta.get("text_font_color", GRAPH_STYLE["text_font_color"]),
            family=meta.get("text_font_family", GRAPH_STYLE["text_font_family"]),
        ),
        hoverinfo="text",
        hovertext=nodes["hovertext"],
        marker=dict(
            size=nodes["node_size"],
            color=nodes["node_color"],
            colorscale=meta["colorscale"],
            showscale=True,
            colorbar=dict(
                title=meta["colorbar_title"],
                orientation="h",
                y=-0.15,
                x=0.5,
                xanchor="center",
                yanchor="top",
                len=0.6,
                tickmode="array",
                tickvals=list(range(meta["n_communities"])),
                ticktext=[str(i) for i in range(meta["n_communities"])],
            ),
            line=dict(
                width=meta.get("node_outline_width", GRAPH_STYLE["node_outline_width"]),
                color=meta.get("node_outline_color", GRAPH_STYLE["node_outline_color"]),
            ),
            opacity=meta.get("node_opacity", GRAPH_STYLE["node_opacity"]),
            cmin=meta["node_color_min"],
            cmax=meta["node_color_max"],
        ),
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(**_make_graph_layout(meta["title"]))

    return fig


def plot_ingredient_louvain_graph_from_export(
    node_path=BASE_DIR / "plots/ingredient_network_export/ingredient_louvain_nodes.parquet",
    edge_path=BASE_DIR / "plots/ingredient_network_export/ingredient_louvain_edges.parquet",
    meta_path=BASE_DIR / "plots/ingredient_network_export/ingredient_louvain_meta.json",
):
    """
    Replot Louvain community graph using exported lightweight files.
    """
    return _plot_partition_graph_from_export(
        node_path=node_path,
        edge_path=edge_path,
        meta_path=meta_path,
    )


def plot_ingredient_leiden_graph_from_export(
    node_path=BASE_DIR / "plots/ingredient_network_export/ingredient_leiden_nodes.parquet",
    edge_path=BASE_DIR / "plots/ingredient_network_export/ingredient_leiden_edges.parquet",
    meta_path=BASE_DIR / "plots/ingredient_network_export/ingredient_leiden_meta.json",
):
    """
    Replot Leiden community graph using exported lightweight files.
    """
    return _plot_partition_graph_from_export(
        node_path=node_path,
        edge_path=edge_path,
        meta_path=meta_path,
    )

def plot_leiden_community_size_bar_from_export(
    community_path=BASE_DIR / "plots/ingredient_network_export/ingredient_leiden_communities.parquet",
    meta_path=BASE_DIR / "plots/ingredient_network_export/ingredient_leiden_meta.json",
):
    """
    Plot a vertical bar chart showing the number of ingredients
    in each Leiden community, using exported lightweight files.
    """
    community_df = pd.read_parquet(community_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    community_df = community_df.sort_values("community").reset_index(drop=True)
    community_df["community_label"] = community_df["community"].apply(lambda x: f"C{x}")

    n = len(community_df)
    if n > 1:
        positions = [i / (n - 1) for i in range(n)]
    else:
        positions = [0.5]

    base_colorscale = meta.get("base_colorscale", "Turbo")
    bar_colors = px.colors.sample_colorscale(base_colorscale, positions)

    fig = go.Figure(
        data=[
            go.Bar(
                x=community_df["community_label"],
                y=community_df["n_ingredients"],
                marker=dict(
                    color=bar_colors,
                    line=dict(color="rgba(0,0,0,0.18)", width=1),
                ),
                hovertemplate=(
                    "<b>Community %{x}</b><br>"
                    "Ingredients: %{y}<br>"
                    "Total recipe count: %{customdata[0]}<br>"
                    "Total weighted degree: %{customdata[1]:,.0f}"
                    "<extra></extra>"
                ),
                customdata=np.column_stack([
                    community_df["total_recipe_count"],
                    community_df["total_weighted_degree"],
                ]),
            )
        ]
    )

    fig.update_layout(
        title="Ingredients per Leiden Community",
        template="plotly_white",
        height=300,
        margin=dict(l=40, r=20, t=50, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#7090a8", size=12),
        xaxis=dict(
            title="Community",
            showgrid=False,
            zeroline=False,
            linecolor="#d8ecf8",
            tickcolor="#d8ecf8",
        ),
        yaxis=dict(
            title="Number of ingredients",
            showgrid=True,
            gridcolor="#eaf3fb",
            zeroline=False,
            linecolor="#d8ecf8",
        ),
        showlegend=False,
    )

    return fig


def plot_top_ingredient_pairs_from_export(
    pair_path=BASE_DIR / "plots/ingredient_network_export/top_ingredient_pairs.parquet",
    meta_path=BASE_DIR / "plots/ingredient_network_export/top_ingredient_pairs_meta.json",
):
    plot_df = pd.read_parquet(pair_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    fig = px.bar(
        plot_df,
        x="weight",
        y="pair",
        orientation="h",
        text="weight",
        title=meta["title"],
    )

    fig.update_traces(
        textposition=meta["textposition"],
        hovertext=plot_df["hovertext"],
        hovertemplate="%{hovertext}<extra></extra>",
    )

    fig.update_layout(
        template=meta["template"],
        title_x=0.5,
        xaxis_title=meta["xaxis_title"],
        yaxis_title=meta["yaxis_title"],
        height=meta["height"],
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def plot_top_ingredient_frequencies_from_export(
    freq_path=BASE_DIR / "plots/ingredient_network_export/top_ingredient_frequencies.parquet",
    meta_path=BASE_DIR / "plots/ingredient_network_export/top_ingredient_frequencies_meta.json",
):
    plot_df = pd.read_parquet(freq_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    fig = px.bar(
        plot_df,
        x="count",
        y="label",
        orientation="h",
        text="count",
        title=meta["title"],
    )

    fig.update_traces(
        textposition=meta["textposition"],
        marker_color=meta["marker_color"],
        hovertext=plot_df["hovertext"],
        hovertemplate="%{hovertext}<extra></extra>",
    )

    fig.update_layout(
        xaxis_title=meta["xaxis_title"],
        yaxis_title=meta["yaxis_title"],
        title_x=0.5,
        template=meta["template"],
        height=meta["height"],
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def plot_ingredient_cooccurrence_heatmap_from_export(
    heatmap_path=BASE_DIR / "plots/ingredient_network_export/ingredient_cooccurrence_heatmap.parquet",
    meta_path=BASE_DIR / "plots/ingredient_network_export/ingredient_cooccurrence_heatmap_meta.json",
):
    matrix = pd.read_parquet(heatmap_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    fig = px.imshow(
        matrix,
        color_continuous_scale=meta["colorscale"],
        title=meta["title"],
        aspect="auto",
    )

    fig.update_layout(
        template=meta["template"],
        title_x=0.5,
        width=meta["width"],
        height=meta["height"],
        xaxis_title=meta["xaxis_title"],
        yaxis_title=meta["yaxis_title"],
    )
    fig.update_yaxes(scaleanchor=meta["scaleanchor"], scaleratio=meta["scaleratio"])

    return fig


def plot_ingredient_cooccurrence_heatmap_clustered_from_export(
    heatmap_path=BASE_DIR / "plots/ingredient_network_export/ingredient_cooccurrence_heatmap_clustered.parquet",
    meta_path=BASE_DIR / "plots/ingredient_network_export/ingredient_cooccurrence_heatmap_meta.json",
):
    matrix = pd.read_parquet(heatmap_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    fig = px.imshow(
        matrix,
        color_continuous_scale=meta["colorscale"],
        title=meta["clustered_title"],
        aspect="auto",
    )

    fig.update_layout(
        template=meta["template"],
        title_x=0.5,
        width=meta["width"],
        height=meta["height"],
        xaxis_title=meta["xaxis_title"],
        yaxis_title=meta["yaxis_title"],
    )
    fig.update_yaxes(scaleanchor=meta["scaleanchor"], scaleratio=meta["scaleratio"])

    return fig


# ============================================================
# CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Build ingredient co-occurrence network and export lightweight plotting data."
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "data/tables/food_recipe.db"),
        help="Path to SQLite database."
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default="recipes",
        help="SQLite table name."
    )
    parser.add_argument("--recipe-col", type=str, default="Name", help="Recipe title column name.")
    parser.add_argument(
        "--ingredients-col",
        type=str,
        default="RecipeIngredientParts",
        help="Ingredient list column name."
    )
    parser.add_argument("--where", type=str, default=None, help="Optional SQL WHERE clause.")

    parser.add_argument("--top-n-ingredients", type=int, default=GRAPH_DEFAULTS["top_n_ingredients"])
    parser.add_argument("--louvain-top-n-ingredients", type=int, default=GRAPH_DEFAULTS["top_n_ingredients"])
    parser.add_argument("--leiden-top-n-ingredients", type=int, default=GRAPH_DEFAULTS["top_n_ingredients"])
    parser.add_argument("--top-pairs-n", type=int, default=50)
    parser.add_argument("--top-frequencies-n", type=int, default=GRAPH_DEFAULTS["top_n_ingredients"])
    parser.add_argument("--heatmap-top-n", type=int, default=30)

    parser.add_argument("--min-edge-weight", type=int, default=GRAPH_DEFAULTS["min_edge_weight"])
    parser.add_argument("--min-node-degree", type=int, default=GRAPH_DEFAULTS["min_node_degree"])
    parser.add_argument("--max-nodes", type=int, default=GRAPH_DEFAULTS["max_nodes"])
    parser.add_argument("--layout-seed", type=int, default=GRAPH_DEFAULTS["layout_seed"])
    parser.add_argument("--layout-k", type=float, default=GRAPH_DEFAULTS["layout_k"])
    parser.add_argument("--title", type=str, default="Network of Top 100 Most Used Ingredients")

    parser.add_argument("--louvain-min-edge-weight", type=int, default=GRAPH_DEFAULTS["min_edge_weight"])
    parser.add_argument("--louvain-min-node-degree", type=int, default=GRAPH_DEFAULTS["min_node_degree"])
    parser.add_argument("--louvain-max-nodes", type=int, default=GRAPH_DEFAULTS["max_nodes"])
    parser.add_argument("--louvain-layout-k", type=float, default=GRAPH_DEFAULTS["layout_k"])
    parser.add_argument("--louvain-resolution", type=float, default=1.2)

    parser.add_argument("--leiden-min-edge-weight", type=int, default=GRAPH_DEFAULTS["min_edge_weight"])
    parser.add_argument("--leiden-min-node-degree", type=int, default=GRAPH_DEFAULTS["min_node_degree"])
    parser.add_argument("--leiden-max-nodes", type=int, default=GRAPH_DEFAULTS["max_nodes"])
    parser.add_argument("--leiden-layout-k", type=float, default=GRAPH_DEFAULTS["layout_k"])
    parser.add_argument("--leiden-resolution", type=float, default=1.0)

    parser.add_argument("--show-louvain", action="store_true", help="Open Louvain community graph in browser.")
    parser.add_argument("--write-louvain-html", action="store_true", help="Write Louvain community graph HTML preview.")
    parser.add_argument("--write-louvain-image", action="store_true", help="Write Louvain community graph PNG image.")

    parser.add_argument("--show-leiden", action="store_true", help="Open Leiden community graph in browser.")
    parser.add_argument("--write-leiden-html", action="store_true", help="Write Leiden community graph HTML preview.")
    parser.add_argument("--write-leiden-image", action="store_true", help="Write Leiden community graph PNG image.")

    parser.add_argument(
        "--no-capitalize-labels",
        action="store_true",
        help="Do not title-case node labels."
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots" / "ingredient_network_export",
        help="Output directory for exported files."
    )

    parser.add_argument("--show-network", action="store_true", help="Open the Plotly figure in a browser.")
    parser.add_argument("--write-network-html", action="store_true", help="Write network_preview.html to the output directory.")
    parser.add_argument("--write-network-image", action="store_true", help="Write ingredient_network.png to the output directory.")

    parser.add_argument("--write-pairs-html", action="store_true", help="Write top ingredient pairs HTML preview.")
    parser.add_argument("--write-pairs-image", action="store_true", help="Write top ingredient pairs PNG image.")
    parser.add_argument("--show-pairs", action="store_true", help="Open top ingredient pairs plot in browser.")

    parser.add_argument("--write-frequencies-html", action="store_true", help="Write top ingredient frequencies HTML preview.")
    parser.add_argument("--write-frequencies-image", action="store_true", help="Write top ingredient frequencies PNG image.")
    parser.add_argument("--show-frequencies", action="store_true", help="Open top ingredient frequencies plot in browser.")

    parser.add_argument("--write-heatmap-html", action="store_true", help="Write heatmap HTML preview.")
    parser.add_argument("--write-heatmap-image", action="store_true", help="Write heatmap PNG image.")
    parser.add_argument("--show-heatmap", action="store_true", help="Open heatmap plot in browser.")

    parser.add_argument("--show-heatmap-clustered", action="store_true", help="Open clustered heatmap plot in browser.")
    parser.add_argument("--write-heatmap-clustered-html", action="store_true", help="Write clustered heatmap HTML preview.")
    parser.add_argument("--write-heatmap-clustered-image", action="store_true", help="Write clustered heatmap PNG image.")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    df = load_recipes_from_sqlite(
        db_path=args.db_path,
        table_name=args.table_name,
        recipe_col=args.recipe_col,
        ingredients_col=args.ingredients_col,
        where=args.where,
    )

    G, fig, export_edge_df, export_node_df, export_meta = plot_ingredient_network(
        df=df,
        recipe_col=args.recipe_col,
        ingredients_col=args.ingredients_col,
        top_n_ingredients=args.top_n_ingredients,
        min_edge_weight=args.min_edge_weight,
        min_node_degree=args.min_node_degree,
        max_nodes=args.max_nodes,
        layout_seed=args.layout_seed,
        layout_k=args.layout_k,
        title=f"Top {args.top_n_ingredients} Most Common Ingredients",
        capitalize_labels=not args.no_capitalize_labels,
    )

    G_louvain, louvain_fig, louvain_edge_df, louvain_node_df, louvain_community_df, louvain_meta = \
        plot_ingredient_louvain_graph(
            df=df,
            recipe_col=args.recipe_col,
            ingredients_col=args.ingredients_col,
            top_n_ingredients=args.louvain_top_n_ingredients,
            min_edge_weight=args.louvain_min_edge_weight,
            min_node_degree=args.louvain_min_node_degree,
            max_nodes=args.louvain_max_nodes,
            layout_seed=args.layout_seed,
            layout_k=args.louvain_layout_k,
            title="Ingredient Community Graph (Louvain)",
            capitalize_labels=not args.no_capitalize_labels,
            louvain_resolution=args.louvain_resolution,
        )

    G_leiden, leiden_fig, leiden_edge_df, leiden_node_df, leiden_community_df, leiden_meta = \
        plot_ingredient_leiden_graph(
            df=df,
            recipe_col=args.recipe_col,
            ingredients_col=args.ingredients_col,
            top_n_ingredients=args.leiden_top_n_ingredients,
            min_edge_weight=args.leiden_min_edge_weight,
            min_node_degree=args.leiden_min_node_degree,
            max_nodes=args.leiden_max_nodes,
            layout_seed=args.layout_seed,
            layout_k=args.leiden_layout_k,
            title="Ingredient Community Graph (Leiden)",
            capitalize_labels=not args.no_capitalize_labels,
            leiden_resolution=args.leiden_resolution,
        )

    out_dir = Path(args.out_dir)

    node_path, edge_path, meta_path = export_network_data(
        export_node_df=export_node_df,
        export_edge_df=export_edge_df,
        export_meta=export_meta,
        out_dir=out_dir,
    )

    print(f"Exported node data: {node_path}")
    print(f"Exported edge data: {edge_path}")
    print(f"Exported metadata : {meta_path}")
    print(f"Graph nodes: {G.number_of_nodes()}")
    print(f"Graph edges: {G.number_of_edges()}")

    louvain_node_path, louvain_edge_path, louvain_community_path, louvain_meta_path = \
        export_louvain_network_data(
            export_node_df=louvain_node_df,
            export_edge_df=louvain_edge_df,
            export_community_df=louvain_community_df,
            export_meta=louvain_meta,
            out_dir=out_dir,
        )

    print(f"Exported Louvain node data    : {louvain_node_path}")
    print(f"Exported Louvain edge data    : {louvain_edge_path}")
    print(f"Exported Louvain communities  : {louvain_community_path}")
    print(f"Exported Louvain metadata     : {louvain_meta_path}")

    leiden_node_path, leiden_edge_path, leiden_community_path, leiden_meta_path = \
        export_leiden_network_data(
            export_node_df=leiden_node_df,
            export_edge_df=leiden_edge_df,
            export_community_df=leiden_community_df,
            export_meta=leiden_meta,
            out_dir=out_dir,
        )

    print(f"Exported Leiden node data     : {leiden_node_path}")
    print(f"Exported Leiden edge data     : {leiden_edge_path}")
    print(f"Exported Leiden communities   : {leiden_community_path}")
    print(f"Exported Leiden metadata      : {leiden_meta_path}")

    if args.write_network_html:
        html_path = out_dir / "network_preview.html"
        fig.write_html(html_path)
        print(f"Wrote HTML preview: {html_path}")

    if args.write_network_image:
        image_path = out_dir / "ingredient_network.png"
        fig.write_image(image_path, width=1700, height=1400, scale=4)
        print(f"Wrote image: {image_path}")

    if args.show_network:
        fig.show()

    if args.write_louvain_html:
        louvain_html_path = out_dir / "ingredient_louvain_graph.html"
        louvain_fig.write_html(louvain_html_path)
        print(f"Wrote Louvain HTML preview: {louvain_html_path}")

    if args.write_louvain_image:
        louvain_image_path = out_dir / "ingredient_louvain_graph.png"
        louvain_fig.write_image(louvain_image_path, width=1700, height=1400, scale=4)
        print(f"Wrote Louvain image: {louvain_image_path}")

    if args.show_louvain:
        louvain_fig.show()

    if args.write_leiden_html:
        leiden_html_path = out_dir / "ingredient_leiden_graph.html"
        leiden_fig.write_html(leiden_html_path)
        print(f"Wrote Leiden HTML preview: {leiden_html_path}")

    if args.write_leiden_image:
        leiden_image_path = out_dir / "ingredient_leiden_graph.png"
        leiden_fig.write_image(leiden_image_path, width=1700, height=1400, scale=4)
        print(f"Wrote Leiden image: {leiden_image_path}")

    if args.show_leiden:
        leiden_fig.show()


if __name__ == "__main__":
    main()