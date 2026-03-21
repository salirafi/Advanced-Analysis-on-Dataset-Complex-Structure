#!/usr/bin/env python3
"""
Functions to build the ingredient network graph using networkx.
The figure's data and metadata are exported as a standalone JSON.
No processed data being output!
"""

from __future__ import annotations

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
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from scipy.cluster.hierarchy import leaves_list, linkage

try:
    import igraph as ig
    import leidenalg as la
except ImportError:
    ig = None
    la = None

try:
    from .ingredient_standardization import standardize_ingredient
except ImportError:
    from ingredient_standardization import standardize_ingredient

BASE_DIR = Path(__file__).resolve().parents[1]


# ##################
# Config
# ##################

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
    "display_edge_k_min": 0,
    "display_edge_k_max": 10,
    "display_edge_k_power": 0.5,
}

RUN_CONFIG = {
    "db_path": BASE_DIR / "data" / "tables" / "food_recipe.db",
    "table_name": "recipes",
    "recipe_col": "Name",
    "ingredients_col": "RecipeIngredientParts",
    "where": None,
    "output_dir": BASE_DIR / "plots",
    "json_filename": "plot_ingredients.json",
    "capitalize_labels": True,
    "network": {
        "top_n_ingredients": GRAPH_DEFAULTS["top_n_ingredients"],
        "min_edge_weight": GRAPH_DEFAULTS["min_edge_weight"],
        "min_node_degree": GRAPH_DEFAULTS["min_node_degree"],
        "max_nodes": GRAPH_DEFAULTS["max_nodes"],
        "layout_seed": GRAPH_DEFAULTS["layout_seed"],
        "layout_k": GRAPH_DEFAULTS["layout_k"],
        "title": None,
        "colorscale": "Blues",
        "display_edge_k_min": GRAPH_DEFAULTS["display_edge_k_min"],
        "display_edge_k_max": GRAPH_DEFAULTS["display_edge_k_max"],
        "display_edge_k_power": GRAPH_DEFAULTS["display_edge_k_power"],
    },
    "leiden": {
        "enabled": True,
        "top_n_ingredients": GRAPH_DEFAULTS["top_n_ingredients"],
        "min_edge_weight": GRAPH_DEFAULTS["min_edge_weight"],
        "min_node_degree": GRAPH_DEFAULTS["min_node_degree"],
        "max_nodes": GRAPH_DEFAULTS["max_nodes"],
        "layout_seed": GRAPH_DEFAULTS["layout_seed"],
        "layout_k": GRAPH_DEFAULTS["layout_k"],
        "title": None,
        "colorscale": "Turbo",
        "leiden_resolution": 1.4,
        "display_edge_k_min": GRAPH_DEFAULTS["display_edge_k_min"],
        "display_edge_k_max": GRAPH_DEFAULTS["display_edge_k_max"],
        "display_edge_k_power": GRAPH_DEFAULTS["display_edge_k_power"],
    },
    "heatmap": {
        "enabled": True,
        "top_n": 30,
        "colorscale": "Blues",
    },
    "top_pairs": {
        "top_n": 50,
        "title": None,
    },
}

USED_FIG_KEYS = [
    "ingredient_network",
    "ingredient_clustered_heatmap",
    "ingredient_top_pairs",
    "ingredient_leiden_graph",
    "ingredient_leiden_community_sizes",
]

WEBAPP_PANELS = {
    "ingredient_main": "ingredient_network",
    "ingredient_alt": "ingredient_leiden_graph",
    "ingredient_heatmap": "ingredient_clustered_heatmap",
    "ingredient_top_pairs": "ingredient_top_pairs",
    "ingredient_leiden_communities": "ingredient_leiden_community_sizes",
}


# ##################
# Shared helpers
# ##################


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


class _PathAwarePlotlyEncoder(PlotlyJSONEncoder):
    def default(self, x):
        if isinstance(x, Path):
            return str(x)
        return super().default(x)


def safe_write_json(obj, path: str | Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, cls=_PathAwarePlotlyEncoder)


# ##################
# Loading / preprocessing
# ##################


def parse_ingredient_cell(value):
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
    db_path = Path(db_path)
    query = f"SELECT {recipe_col}, {ingredients_col} FROM {table_name}"
    if where:
        query += f" WHERE {where}"
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    return preprocess_ingredient_lists(df, ingredients_col=ingredients_col)


# ##################
# Graph computation
# ##################


def scale_log(values: Iterable[float], out_min: float, out_max: float, multiplier: float = 1.5):
    arr = np.asarray(list(values), dtype=float)
    arr = np.log1p(arr)
    if arr.size == 0:
        return np.array([], dtype=float)
    if np.allclose(arr.min(), arr.max()):
        return np.full_like(arr, (out_min + out_max) / 2, dtype=float)
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return (out_min + arr * (out_max - out_min)) * multiplier



def build_discrete_colorscale(n, base_colors="Turbo"):
    if n <= 0:
        return [(0.0, "rgb(0,0,0)"), (1.0, "rgb(0,0,0)")]
    palette = px.colors.sample_colorscale(
        base_colors,
        [i / (n - 1) if n > 1 else 0 for i in range(n)],
    )
    colorscale = []
    for i, c in enumerate(palette):
        colorscale.append((i / n, c))
        colorscale.append(((i + 1) / n, c))
    return colorscale



def _make_graph_layout(title: str | None) -> dict:
    return dict(
        showlegend=False,
        hovermode="closest",
        margin=GRAPH_STYLE["margin"],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor=GRAPH_STYLE["plot_bgcolor"],
        paper_bgcolor=GRAPH_STYLE["paper_bgcolor"],
        title=title,
    )



def _build_filtered_ingredient_graph(
    df: pd.DataFrame,
    recipe_col: str = "Name",
    ingredients_col: str = "RecipeIngredientParts",
    top_n_ingredients: int | None = 50,
    min_edge_weight: int = 10,
    min_node_degree: int = 1,
    max_nodes: int | None = None,
):
    ingredient_freq = Counter()

    for _, row in df[[recipe_col, ingredients_col]].dropna().iterrows():
        ingredients = row[ingredients_col]
        if not isinstance(ingredients, list):
            continue
        ingredients_unique = sorted(set(i for i in ingredients if isinstance(i, str) and i.strip()))
        ingredient_freq.update(ingredients_unique)

    allowed_ingredients = None
    if top_n_ingredients is not None:
        allowed_ingredients = {ingredient for ingredient, _ in ingredient_freq.most_common(top_n_ingredients)}

    pair_freq = Counter()
    filtered_ingredient_freq = Counter()

    for _, row in df[[recipe_col, ingredients_col]].dropna().iterrows():
        ingredients = row[ingredients_col]
        if not isinstance(ingredients, list):
            continue
        ingredients_unique = sorted(set(i for i in ingredients if isinstance(i, str) and i.strip()))
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
        raise ValueError("No edges remain after filtering. Try lowering min_edge_weight or increasing top_n_ingredients.")

    G = nx.Graph()
    for _, row in edge_df.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["weight"])
    for node in G.nodes():
        G.nodes[node]["recipe_count"] = filtered_ingredient_freq[node]

    low_degree_nodes = [n for n, d in dict(G.degree()).items() if d < min_node_degree]
    G.remove_nodes_from(low_degree_nodes)
    G.remove_nodes_from(list(nx.isolates(G)))

    if G.number_of_nodes() == 0:
        raise ValueError("No nodes remain after filtering. Try lowering min_edge_weight/min_node_degree.")

    if max_nodes is not None and G.number_of_nodes() > max_nodes:
        weighted_degree_tmp = dict(G.degree(weight="weight"))
        top_nodes = sorted(weighted_degree_tmp, key=weighted_degree_tmp.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    edge_rows_filtered = [
        {"source": u, "target": v, "weight": int(data["weight"])}
        for u, v, data in G.edges(data=True)
    ]
    edge_df = pd.DataFrame(edge_rows_filtered)
    weighted_degree = dict(G.degree(weight="weight"))
    simple_degree = dict(G.degree())

    node_df = pd.DataFrame(
        [
            {
                "ingredient": node,
                "degree": int(simple_degree[node]),
                "weighted_degree": float(weighted_degree[node]),
                "recipe_count": int(G.nodes[node]["recipe_count"]),
            }
            for node in G.nodes()
        ]
    ).sort_values("weighted_degree", ascending=False).reset_index(drop=True)

    return G, edge_df, node_df, weighted_degree, simple_degree



def _select_display_edges_per_node(
    edge_df: pd.DataFrame,
    node_df: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 12,
    k_power: float = 0.5,
) -> pd.DataFrame:
    if edge_df.empty or node_df.empty:
        return edge_df.copy()

    node_scale = node_df.set_index("ingredient")["weighted_degree"].astype(float)
    scale_vals = np.log1p(node_scale.to_numpy())
    smin = float(scale_vals.min())
    smax = float(scale_vals.max())

    if np.isclose(smin, smax):
        norm_scale = pd.Series(1.0, index=node_scale.index)
    else:
        norm_scale = pd.Series((np.log1p(node_scale) - smin) / (smax - smin), index=node_scale.index)
    norm_scale = norm_scale.clip(0, 1) ** float(k_power)
    node_k = (k_min + norm_scale * (k_max - k_min)).round().astype(int).clip(lower=k_min, upper=k_max)

    left = edge_df[["source", "target", "weight"]].rename(columns={"source": "node", "target": "nbr"})
    right = edge_df[["source", "target", "weight"]].rename(columns={"target": "node", "source": "nbr"})
    long_df = pd.concat([left, right], ignore_index=True)
    long_df["node_k"] = long_df["node"].map(node_k).fillna(k_min).astype(int)
    long_df = long_df.sort_values(["node", "weight", "nbr"], ascending=[True, False, True])
    long_df["rank_within_node"] = long_df.groupby("node").cumcount() + 1
    long_df = long_df[long_df["rank_within_node"] <= long_df["node_k"]].copy()
    long_df["a"] = long_df[["node", "nbr"]].min(axis=1)
    long_df["b"] = long_df[["node", "nbr"]].max(axis=1)

    return (
        long_df[["a", "b", "weight"]]
        .drop_duplicates(subset=["a", "b"])
        .rename(columns={"a": "source", "b": "target"})
        .sort_values(["weight", "source", "target"], ascending=[False, True, True])
        .reset_index(drop=True)
    )



def _compute_shared_network_geometry(
    G: nx.Graph,
    display_edge_df: pd.DataFrame,
    weighted_degree: dict,
    layout_seed: int = 42,
    layout_k: float = 0.35,
):
    pos = nx.spring_layout(G, seed=layout_seed, k=layout_k, weight="weight")
    node_order = list(G.nodes())

    recipe_counts = [G.nodes[n]["recipe_count"] for n in node_order]
    node_sizes_scaled = scale_log(
        recipe_counts,
        GRAPH_STYLE["node_size_min"],
        GRAPH_STYLE["node_size_max"],
        multiplier=1.0,
    )

    edge_width_map: dict[tuple[str, str], float] = {}
    if not display_edge_df.empty:
        edge_widths_scaled = scale_log(
            display_edge_df["weight"].to_numpy(dtype=float),
            GRAPH_STYLE["edge_width_min"],
            GRAPH_STYLE["edge_width_max"],
            multiplier=0.5,
        )
        for (_, row), width in zip(display_edge_df.iterrows(), edge_widths_scaled):
            edge_width_map[(row["source"], row["target"])] = float(width)
            edge_width_map[(row["target"], row["source"])] = float(width)

    wdeg_values = np.array([weighted_degree[n] for n in node_order], dtype=float)
    wdeg_log = np.log1p(wdeg_values)
    if len(wdeg_log) > 1:
        lo = np.percentile(wdeg_log, 5)
        hi = np.percentile(wdeg_log, 95)
        wdeg_log = np.clip(wdeg_log, lo, hi)

    return pos, node_order, node_sizes_scaled, edge_width_map, wdeg_log



def _build_igraph_from_networkx(G: nx.Graph):
    if ig is None or la is None:
        raise ImportError("Leiden dependencies are missing. Install with: pip install igraph leidenalg")

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



def _detect_leiden_communities(G: nx.Graph, resolution: float = 1.0, seed: int = 42):
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


# ##################
# Lightweight figure inputs
# ##################


def build_ingredient_network_export(
    df: pd.DataFrame,
    recipe_col: str = "Name",
    ingredients_col: str = "RecipeIngredientParts",
    top_n_ingredients: int | None = GRAPH_DEFAULTS["top_n_ingredients"],
    min_edge_weight: int = GRAPH_DEFAULTS["min_edge_weight"],
    min_node_degree: int = GRAPH_DEFAULTS["min_node_degree"],
    max_nodes: int | None = GRAPH_DEFAULTS["max_nodes"],
    layout_seed: int = GRAPH_DEFAULTS["layout_seed"],
    layout_k: float = GRAPH_DEFAULTS["layout_k"],
    title: str | None = "Ingredient Co-occurrence Network",
    capitalize_labels: bool = GRAPH_DEFAULTS["capitalize_labels"],
    colorscale: str = "Blues",
    display_edge_k_min: int = GRAPH_DEFAULTS["display_edge_k_min"],
    display_edge_k_max: int = GRAPH_DEFAULTS["display_edge_k_max"],
    display_edge_k_power: float = GRAPH_DEFAULTS["display_edge_k_power"],
):
    G, edge_df_full, node_df, weighted_degree, simple_degree = _build_filtered_ingredient_graph(
        df=df,
        recipe_col=recipe_col,
        ingredients_col=ingredients_col,
        top_n_ingredients=top_n_ingredients,
        min_edge_weight=min_edge_weight,
        min_node_degree=min_node_degree,
        max_nodes=max_nodes,
    )

    edge_df_display = _select_display_edges_per_node(
        edge_df=edge_df_full,
        node_df=node_df,
        k_min=display_edge_k_min,
        k_max=display_edge_k_max,
        k_power=display_edge_k_power,
    )

    pos, node_order, node_sizes_scaled, edge_width_map, wdeg_log = _compute_shared_network_geometry(
        G=G,
        display_edge_df=edge_df_display,
        weighted_degree=weighted_degree,
        layout_seed=layout_seed,
        layout_k=layout_k,
    )

    nodes = []
    for i, node in enumerate(node_order):
        label = node.title() if capitalize_labels else node
        hovertext = (
            f"<b>{label}</b><br>"
            f"Recipes containing ingredient: {G.nodes[node]['recipe_count']}<br>"
            f"Connected ingredients: {simple_degree[node]}<br>"
            f"Weighted degree: {weighted_degree[node]}"
        )
        nodes.append(
            {
                "ingredient": node,
                "label": label,
                "x": float(pos[node][0]),
                "y": float(pos[node][1]),
                "degree": int(simple_degree[node]),
                "weighted_degree": float(weighted_degree[node]),
                "recipe_count": int(G.nodes[node]["recipe_count"]),
                "node_size": float(node_sizes_scaled[i]),
                "node_color": float(wdeg_log[i]),
                "hovertext": hovertext,
            }
        )
    export_node_df = pd.DataFrame(nodes)

    edges = []
    for _, row in edge_df_display.iterrows():
        u = row["source"]
        v = row["target"]
        edges.append(
            {
                "source": u,
                "target": v,
                "weight": int(row["weight"]),
                "x0": float(pos[u][0]),
                "y0": float(pos[u][1]),
                "x1": float(pos[v][0]),
                "y1": float(pos[v][1]),
                "edge_width": float(edge_width_map[(u, v)]),
                "edge_color": GRAPH_STYLE["edge_color"],
                "hovertext": f"{(u.title() if capitalize_labels else u)} ↔ {(v.title() if capitalize_labels else v)}<br>Co-occurrence: {int(row['weight'])}",
            }
        )
    export_edge_df = pd.DataFrame(edges)

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
        "plot_bgcolor": GRAPH_STYLE["plot_bgcolor"],
        "paper_bgcolor": GRAPH_STYLE["paper_bgcolor"],
        "margin": GRAPH_STYLE["margin"],
        "full_graph_n_nodes": int(G.number_of_nodes()),
        "full_graph_n_edges": int(edge_df_full.shape[0]),
        "display_graph_n_edges": int(edge_df_display.shape[0]),
        "display_edge_k_min": int(display_edge_k_min),
        "display_edge_k_max": int(display_edge_k_max),
        "display_edge_k_power": float(display_edge_k_power),
    }
    return export_node_df, export_edge_df, export_meta



def build_ingredient_leiden_export(
    df: pd.DataFrame,
    recipe_col: str = "Name",
    ingredients_col: str = "RecipeIngredientParts",
    top_n_ingredients: int | None = GRAPH_DEFAULTS["top_n_ingredients"],
    min_edge_weight: int = GRAPH_DEFAULTS["min_edge_weight"],
    min_node_degree: int = GRAPH_DEFAULTS["min_node_degree"],
    max_nodes: int | None = GRAPH_DEFAULTS["max_nodes"],
    layout_seed: int = GRAPH_DEFAULTS["layout_seed"],
    layout_k: float = GRAPH_DEFAULTS["layout_k"],
    title: str | None = "Ingredient Community Graph (Leiden)",
    capitalize_labels: bool = GRAPH_DEFAULTS["capitalize_labels"],
    colorscale: str = "Turbo",
    leiden_resolution: float = 1.0,
    display_edge_k_min: int = GRAPH_DEFAULTS["display_edge_k_min"],
    display_edge_k_max: int = GRAPH_DEFAULTS["display_edge_k_max"],
    display_edge_k_power: float = GRAPH_DEFAULTS["display_edge_k_power"],
):
    G, edge_df_full, node_df, weighted_degree, simple_degree = _build_filtered_ingredient_graph(
        df=df,
        recipe_col=recipe_col,
        ingredients_col=ingredients_col,
        top_n_ingredients=top_n_ingredients,
        min_edge_weight=min_edge_weight,
        min_node_degree=min_node_degree,
        max_nodes=max_nodes,
    )

    edge_df_display = _select_display_edges_per_node(
        edge_df=edge_df_full,
        node_df=node_df,
        k_min=display_edge_k_min,
        k_max=display_edge_k_max,
        k_power=display_edge_k_power,
    )
    _, node_to_community = _detect_leiden_communities(G, resolution=leiden_resolution, seed=layout_seed)
    pos, node_order, node_sizes_scaled, edge_width_map, _ = _compute_shared_network_geometry(
        G=G,
        display_edge_df=edge_df_display,
        weighted_degree=weighted_degree,
        layout_seed=layout_seed,
        layout_k=layout_k,
    )

    n_communities = len(set(node_to_community.values()))
    discrete_colorscale = build_discrete_colorscale(n_communities, colorscale)

    nodes = []
    for i, node in enumerate(node_order):
        cid = int(node_to_community[node])
        label = node.title() if capitalize_labels else node
        hovertext = (
            f"<b>{label}</b><br>"
            f"Community: {cid}<br>"
            f"Recipes containing ingredient: {G.nodes[node]['recipe_count']}<br>"
            f"Connected ingredients: {simple_degree[node]}<br>"
            f"Weighted degree: {weighted_degree[node]}"
        )
        nodes.append(
            {
                "ingredient": node,
                "label": label,
                "x": float(pos[node][0]),
                "y": float(pos[node][1]),
                "degree": int(simple_degree[node]),
                "weighted_degree": float(weighted_degree[node]),
                "recipe_count": int(G.nodes[node]["recipe_count"]),
                "community": cid,
                "node_size": float(node_sizes_scaled[i]),
                "node_color": cid,
                "hovertext": hovertext,
            }
        )
    export_node_df = pd.DataFrame(nodes)

    edges = []
    for _, row in edge_df_display.iterrows():
        u = row["source"]
        v = row["target"]
        same_comm = bool(node_to_community[u] == node_to_community[v])
        edge_color = "rgba(120,120,120,0.30)" if same_comm else "rgba(170,170,170,0.15)"
        edges.append(
            {
                "source": u,
                "target": v,
                "weight": int(row["weight"]),
                "x0": float(pos[u][0]),
                "y0": float(pos[u][1]),
                "x1": float(pos[v][0]),
                "y1": float(pos[v][1]),
                "edge_width": float(edge_width_map[(u, v)]),
                "same_community": same_comm,
                "edge_color": edge_color,
                "hovertext": f"{(u.title() if capitalize_labels else u)} ↔ {(v.title() if capitalize_labels else v)}<br>Co-occurrence: {int(row['weight'])}",
            }
        )
    export_edge_df = pd.DataFrame(edges)

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
        "chart_type": "ingredient_leiden_graph",
        "title": title,
        "method": "leiden",
        "colorscale": discrete_colorscale,
        "base_colorscale": colorscale,
        "colorbar_title": "Community",
        "node_color_min": -0.5,
        "node_color_max": n_communities - 0.5,
        "n_communities": int(n_communities),
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
        "plot_bgcolor": GRAPH_STYLE["plot_bgcolor"],
        "paper_bgcolor": GRAPH_STYLE["paper_bgcolor"],
        "margin": GRAPH_STYLE["margin"],
        "leiden_resolution": float(leiden_resolution),
        "top_n_ingredients": None if top_n_ingredients is None else int(top_n_ingredients),
        "min_edge_weight": int(min_edge_weight),
        "min_node_degree": int(min_node_degree),
        "max_nodes": None if max_nodes is None else int(max_nodes),
        "display_edge_k_min": int(display_edge_k_min),
        "display_edge_k_max": int(display_edge_k_max),
        "display_edge_k_power": float(display_edge_k_power),
        "full_graph_n_nodes": int(G.number_of_nodes()),
        "full_graph_n_edges": int(edge_df_full.shape[0]),
        "display_graph_n_edges": int(edge_df_display.shape[0]),
    }
    return export_node_df, export_edge_df, export_community_df, export_meta



def build_top_pairs_export(
    edge_df: pd.DataFrame,
    top_n: int = 30,
    title: str | None = "Top Ingredient Pair Co-occurrences",
    capitalize_labels: bool = True,
):
    if edge_df.empty:
        raise ValueError("edge_df is empty.")
    plot_df = edge_df.copy()
    if capitalize_labels:
        plot_df["pair"] = [
            f"{min(a, b).title()} + {max(a, b).title()}" for a, b in zip(plot_df["source"], plot_df["target"])
        ]
    else:
        plot_df["pair"] = plot_df["source"] + " + " + plot_df["target"]

    plot_df = plot_df.sort_values("weight", ascending=False).head(top_n).copy()
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    plot_df["hovertext"] = (
        "<b>" + plot_df["pair"] + "</b><br>Recipes containing both ingredients: " + plot_df["weight"].astype(str)
    )

    export_pair_df = plot_df[["source", "target", "pair", "weight", "hovertext"]].copy()
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
    return export_pair_df, export_meta



def build_clustered_heatmap_export(
    df: pd.DataFrame,
    ingredients_col: str = "RecipeIngredientParts",
    top_n: int = 30,
    colorscale: str = "Magma",
    capitalize_labels: bool = True,
):
    ingredient_freq = Counter()
    for ingredients in df[ingredients_col].dropna():
        if isinstance(ingredients, list):
            ingredient_freq.update(set(i for i in ingredients if isinstance(i, str) and i.strip()))

    top_ingredients = [i for i, _ in ingredient_freq.most_common(top_n)]
    top_set = set(top_ingredients)
    pair_freq = Counter()

    for ingredients in df[ingredients_col].dropna():
        if not isinstance(ingredients, list):
            continue
        ingredients_filtered = sorted(set(i for i in ingredients if isinstance(i, str) and i.strip()) & top_set)
        for a, b in itertools.combinations(ingredients_filtered, 2):
            pair_freq[(a, b)] += 1

    co_matrix = pd.DataFrame(0, index=top_ingredients, columns=top_ingredients, dtype=float)
    for (a, b), w in pair_freq.items():
        co_matrix.loc[a, b] = w
        co_matrix.loc[b, a] = w
    for ing in top_ingredients:
        co_matrix.loc[ing, ing] = 0.0

    co_matrix_log = np.log1p(co_matrix)
    order = leaves_list(linkage(co_matrix_log, method="average"))
    co_matrix_log_clustered = co_matrix_log.iloc[order, order]

    if capitalize_labels:
        display = co_matrix_log_clustered.copy()
        display.index = [x.title() for x in display.index]
        display.columns = [x.title() for x in display.columns]
    else:
        display = co_matrix_log_clustered

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
        "showscale": False,
        "xgap": 0,
        "ygap": 0,
    }
    return display, export_meta


# ##################
# Plotly figure builders
# ##################


def build_fig_ingredient_network(nodes: pd.DataFrame, edges: pd.DataFrame, meta: dict) -> go.Figure:
    edge_traces = [
        go.Scatter(
            x=[row["x0"], row["x1"]],
            y=[row["y0"], row["y1"]],
            mode="lines",
            line=dict(width=float(row["edge_width"]), color=row["edge_color"]),
            hoverinfo="text",
            text=row["hovertext"],
            showlegend=False,
        )
        for _, row in edges.iterrows()
    ]

    node_trace = go.Scatter(
        x=nodes["x"],
        y=nodes["y"],
        mode="markers+text",
        text=nodes["label"],
        textposition="middle center",
        textfont=dict(
            size=meta["text_font_size"],
            color=meta["text_font_color"],
            family=meta["text_font_family"],
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
                y=-0.05,
                x=0.5,
                xanchor="center",
                yanchor="top",
                len=0.6,
            ),
            line=dict(width=meta["node_outline_width"], color=meta["node_outline_color"]),
            opacity=meta["node_opacity"],
            cmin=meta["node_color_min"],
            cmax=meta["node_color_max"],
        ),
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(**_make_graph_layout(meta["title"]))
    return fig



def build_fig_ingredient_leiden_graph(nodes: pd.DataFrame, edges: pd.DataFrame, meta: dict) -> go.Figure:
    edge_traces = [
        go.Scatter(
            x=[row["x0"], row["x1"]],
            y=[row["y0"], row["y1"]],
            mode="lines",
            line=dict(width=float(row["edge_width"]), color=row["edge_color"]),
            hoverinfo="text",
            text=row["hovertext"],
            showlegend=False,
        )
        for _, row in edges.iterrows()
    ]

    node_trace = go.Scatter(
        x=nodes["x"],
        y=nodes["y"],
        mode="markers+text",
        text=nodes["label"],
        textposition="middle center",
        textfont=dict(
            size=meta["text_font_size"],
            color=meta["text_font_color"],
            family=meta["text_font_family"],
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
                y=-0.05,
                x=0.5,
                xanchor="center",
                yanchor="top",
                len=0.6,
                tickmode="array",
                tickvals=list(range(meta["n_communities"])),
                ticktext=[str(i) for i in range(meta["n_communities"])],
            ),
            line=dict(width=meta["node_outline_width"], color=meta["node_outline_color"]),
            opacity=meta["node_opacity"],
            cmin=meta["node_color_min"],
            cmax=meta["node_color_max"],
        ),
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(**_make_graph_layout(meta["title"]))
    return fig



def build_fig_leiden_community_sizes(community_df: pd.DataFrame, meta: dict) -> go.Figure:
    community_df = community_df.sort_values("community").reset_index(drop=True)
    community_df["community_label"] = community_df["community"].apply(lambda x: f"C{x}")
    n = len(community_df)
    positions = [i / (n - 1) for i in range(n)] if n > 1 else [0.5]
    bar_colors = px.colors.sample_colorscale(meta.get("base_colorscale", "Turbo"), positions)

    fig = go.Figure(
        data=[
            go.Bar(
                x=community_df["community_label"],
                y=community_df["n_ingredients"],
                marker=dict(color=bar_colors, line=dict(color="rgba(0,0,0,0.18)", width=1)),
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
        title=None,
        template="plotly_white",
        height=300,
        margin=dict(l=40, r=20, t=50, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#7090a8", size=12),
        xaxis=dict(title="Community", showgrid=False, zeroline=False, linecolor="#d8ecf8", tickcolor="#d8ecf8"),
        yaxis=dict(title="Number of ingredients", showgrid=True, gridcolor="#eaf3fb", zeroline=False, linecolor="#d8ecf8"),
        showlegend=False,
    )
    return fig



def build_fig_top_pairs(plot_df: pd.DataFrame, meta: dict) -> go.Figure:
    fig = px.bar(plot_df, x="weight", y="pair", orientation="h", text="weight", title=meta["title"])
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



def build_fig_clustered_heatmap(matrix: pd.DataFrame, meta: dict) -> go.Figure:
    fig = px.imshow(matrix, color_continuous_scale=meta["colorscale"], title=meta["clustered_title"], aspect="equal")
    fig.update_layout(
        template=meta["template"],
        title_x=0.5,
        width=meta["width"],
        height=meta["height"],
        xaxis_title=meta["xaxis_title"],
        yaxis_title=meta["yaxis_title"],
    )
    fig.update_traces(showscale=meta.get("showscale", False), xgap=meta.get("xgap", 0), ygap=meta.get("ygap", 0))
    fig.update_coloraxes(showscale=meta.get("showscale", False))
    return fig


# ##################
# Full Plotly JSON export
# ##################


def figure_to_json_dict(fig: go.Figure) -> dict:
    return fig.to_plotly_json()



def build_all_figures(df: pd.DataFrame, config: dict = RUN_CONFIG) -> dict[str, dict]:
    standalone = {}
    capitalize_labels = bool(config["capitalize_labels"])
    recipe_col = config["recipe_col"]
    ingredients_col = config["ingredients_col"]

    network_cfg = config["network"]
    network_nodes, network_edges, network_meta = build_ingredient_network_export(
        df=df,
        recipe_col=recipe_col,
        ingredients_col=ingredients_col,
        top_n_ingredients=network_cfg["top_n_ingredients"],
        min_edge_weight=network_cfg["min_edge_weight"],
        min_node_degree=network_cfg["min_node_degree"],
        max_nodes=network_cfg["max_nodes"],
        layout_seed=network_cfg["layout_seed"],
        layout_k=network_cfg["layout_k"],
        title=network_cfg["title"],
        capitalize_labels=capitalize_labels,
        colorscale=network_cfg["colorscale"],
        display_edge_k_min=network_cfg["display_edge_k_min"],
        display_edge_k_max=network_cfg["display_edge_k_max"],
        display_edge_k_power=network_cfg["display_edge_k_power"],
    )
    standalone["ingredient_network"] = figure_to_json_dict(
        build_fig_ingredient_network(network_nodes, network_edges, network_meta)
    )

    top_pairs_cfg = config["top_pairs"]
    top_pairs_df, top_pairs_meta = build_top_pairs_export(
        edge_df=network_edges[["source", "target", "weight"]].copy(),
        top_n=top_pairs_cfg["top_n"],
        title=top_pairs_cfg["title"],
        capitalize_labels=capitalize_labels,
    )
    standalone["ingredient_top_pairs"] = figure_to_json_dict(
        build_fig_top_pairs(top_pairs_df, top_pairs_meta)
    )

    if config["leiden"]["enabled"]:
        leiden_cfg = config["leiden"]
        leiden_nodes, leiden_edges, leiden_communities, leiden_meta = build_ingredient_leiden_export(
            df=df,
            recipe_col=recipe_col,
            ingredients_col=ingredients_col,
            top_n_ingredients=leiden_cfg["top_n_ingredients"],
            min_edge_weight=leiden_cfg["min_edge_weight"],
            min_node_degree=leiden_cfg["min_node_degree"],
            max_nodes=leiden_cfg["max_nodes"],
            layout_seed=leiden_cfg["layout_seed"],
            layout_k=leiden_cfg["layout_k"],
            title=leiden_cfg["title"],
            capitalize_labels=capitalize_labels,
            colorscale=leiden_cfg["colorscale"],
            leiden_resolution=leiden_cfg["leiden_resolution"],
            display_edge_k_min=leiden_cfg["display_edge_k_min"],
            display_edge_k_max=leiden_cfg["display_edge_k_max"],
            display_edge_k_power=leiden_cfg["display_edge_k_power"],
        )
        standalone["ingredient_leiden_graph"] = figure_to_json_dict(
            build_fig_ingredient_leiden_graph(leiden_nodes, leiden_edges, leiden_meta)
        )
        standalone["ingredient_leiden_community_sizes"] = figure_to_json_dict(
            build_fig_leiden_community_sizes(leiden_communities, leiden_meta)
        )

    if config["heatmap"]["enabled"]:
        heatmap_cfg = config["heatmap"]
        clustered_matrix, heatmap_meta = build_clustered_heatmap_export(
            df=df,
            ingredients_col=ingredients_col,
            top_n=heatmap_cfg["top_n"],
            colorscale=heatmap_cfg["colorscale"],
            capitalize_labels=capitalize_labels,
        )
        standalone["ingredient_clustered_heatmap"] = figure_to_json_dict(
            build_fig_clustered_heatmap(clustered_matrix, heatmap_meta)
        )

    return standalone



def export_plotly_payload(payload: dict, output_dir: str | Path, filename: str) -> Path:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    out_path = output_dir / filename
    safe_write_json(payload, out_path)
    return out_path


# ##################
# Main
# ##################


def main(config: dict = RUN_CONFIG):
    output_dir = Path(config["output_dir"])
    ensure_dir(output_dir)

    print("\n══════════════════════════════════════════════════════")
    print("  Ingredient Network Pipeline")
    print("══════════════════════════════════════════════════════\n")

    print("🥗 Loading ingredient data...")
    df = load_recipes_from_sqlite(
        db_path=config["db_path"],
        table_name=config["table_name"],
        recipe_col=config["recipe_col"],
        ingredients_col=config["ingredients_col"],
        where=config["where"],
    )
    print(f"✅ Loaded and cleaned {len(df):,} recipes")

    print("🕸️ Computing ingredient graph structures and building figures...")
    standalone_figures = build_all_figures(df, config=config)
    print(f"🎨 Built {len(standalone_figures)} Plotly figures")

    payload = {
        "visualization": "ingredient_network",
        "webapp_panels": WEBAPP_PANELS,
        "standalone_figures": standalone_figures,
        "meta": {
            "recipe_file": str(config["db_path"]),
            "output_dir": str(output_dir),
            "table_name": config["table_name"],
            "recipe_col": config["recipe_col"],
            "ingredients_col": config["ingredients_col"],
            "capitalize_labels": bool(config["capitalize_labels"]),
            "n_recipes_after_cleaning": int(len(df)),
            "used_fig_keys": list(USED_FIG_KEYS),
        },
    }

    print("💾 Exporting Plotly JSON payload...")
    out_path = export_plotly_payload(payload, output_dir, config["json_filename"])

    print("📋 Summary")
    print(f"   • Recipes after cleaning: {len(df):,}")
    print(f"   • Figures exported: {len(standalone_figures)}")
    print(f"   • Output file: {out_path}")

    return {"output_path": str(out_path), "figure_keys": list(standalone_figures.keys())}


if __name__ == "__main__":
    main()
