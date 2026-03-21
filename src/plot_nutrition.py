#!/usr/bin/env python3
"""
Functions to build the nutritional landscape figure using PCA.
The figure's data and metadata are exported as a standalone JSON.
No processed data being output!
"""

from __future__ import annotations

import json
import os
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]


# ##################
# Config
# ##################

RUN_CONFIG = {
    "db_path": BASE_DIR / "data" / "tables" / "food_recipe.db",
    "table_name": "recipes",
    "output_dir": BASE_DIR / "plots",
    "top_n_categories": 8,
    "n_clusters": 7,
    "min_reviews": 2,
    "max_missing_allowed": 2,
    "random_state": 42,
    "json_filename": "plot_nutrition.json",
}

STYLE = {
    "discrete_cluster_colors": px.colors.sample_colorscale(
        "Turbo",
        [
            i / (RUN_CONFIG["n_clusters"] - 1)
            if RUN_CONFIG["n_clusters"] > 1 else 0
            for i in range(RUN_CONFIG["n_clusters"])
        ],
    ),
    "category_colors": px.colors.qualitative.Set2,
    "pca_marker_size": 7.5,
    "pca_marker_opacity": 0.74,
}

NUTRITION_COLS = [
    "Calories",
    "FatContent",
    "SaturatedFatContent",
    "CholesterolContent",
    "SodiumContent",
    "CarbohydrateContent",
    "FiberContent",
    "SugarContent",
    "ProteinContent",
]

META_COLS = [
    "RecipeId",
    "Name",
    "RecipeCategory",
]

USED_FIG_KEYS = [
    "nutrition_pca_landscape",
    "nutrition_cluster_heatmap",
    "nutrition_pca_loadings",
    "nutrition_pca_categories",
    "nutrition_cluster_categories",
]

WEBAPP_PANELS = {
    "nutrition_main": "nutrition_pca_landscape",
    "nutrition_alt": "nutrition_cluster_heatmap",
    "nutrition_loadings": "nutrition_pca_loadings",
    "nutrition_categories": "nutrition_pca_categories",
    "nutrition_cluster_categories": "nutrition_cluster_categories",
}


# ##################
# Shared file helpers
# ##################


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_write_json(obj, path: str | Path):
    """Write JSON using Plotly's encoder so numpy arrays/scalars serialize cleanly."""

    class _PathAwarePlotlyEncoder(PlotlyJSONEncoder):
        def default(self, x):
            if isinstance(x, Path):
                return str(x)
            return super().default(x)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, cls=_PathAwarePlotlyEncoder)


# ##################
# Loading / preprocessing
# ##################


def load_recipes_from_sqlite(db_path: str | Path, table: str = "recipes", min_reviews: int = 2):
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    print("📂  Loading data...")

    query = f"""
        SELECT
            RecipeId,
            Name,
            RecipeCategory,
            Calories,
            FatContent,
            SaturatedFatContent,
            CholesterolContent,
            SodiumContent,
            CarbohydrateContent,
            FiberContent,
            SugarContent,
            ProteinContent
        FROM {table}
        WHERE
            Calories IS NOT NULL
            AND FatContent IS NOT NULL
            AND CarbohydrateContent IS NOT NULL
            AND ProteinContent IS NOT NULL
            AND SugarContent IS NOT NULL
            AND SodiumContent IS NOT NULL
            AND FiberContent IS NOT NULL
            AND CholesterolContent IS NOT NULL
            AND SaturatedFatContent IS NOT NULL
    """

    with sqlite3.connect(db_path) as conn:

        df = pd.read_sql_query(query, conn)
        print(f"    Filtered recipes : {len(df):,}")
        return df


def clean_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def basic_recipe_cleaning(
    df: pd.DataFrame,
    *,
    min_reviews: int,
    max_missing_allowed: int,
) -> pd.DataFrame:
    
    print("🔧  Cleaning recipes...")
    df = df.copy()

    for col in META_COLS:
        if col not in df.columns:
            df[col] = np.nan

    numeric_cols = NUTRITION_COLS
    df = clean_numeric_columns(df, numeric_cols)

    missing_count = df[NUTRITION_COLS].isna().sum(axis=1)
    df = df.loc[missing_count <= max_missing_allowed].copy()

    for col in NUTRITION_COLS:
        df[col] = df[col].fillna(df[col].median())

    df = df[df["Calories"].fillna(0) > 0].copy()
    for col in NUTRITION_COLS:
        df = df[df[col].fillna(0) >= 0].copy()

    df["RecipeCategory"] = df["RecipeCategory"].fillna("Unknown").astype(str).str.strip()
    df.loc[df["RecipeCategory"] == "", "RecipeCategory"] = "Unknown"
    df["Name"] = df["Name"].fillna("Unknown Recipe").astype(str)

    print(f"    Recipes after filtering and cleaning: {len(df):,}")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:

    print("🔢  Computing derived features...")

    df = df.copy()
    eps = 1e-6

    df["ProteinPer100Cal"] = 100 * df["ProteinContent"] / (df["Calories"] + eps)
    df["FiberPer100Cal"] = 100 * df["FiberContent"] / (df["Calories"] + eps)
    df["SugarPer100Cal"] = 100 * df["SugarContent"] / (df["Calories"] + eps)
    df["SodiumPer100Cal"] = 100 * df["SodiumContent"] / (df["Calories"] + eps)
    df["FatPer100Cal"] = 100 * df["FatContent"] / (df["Calories"] + eps)
    df["CarbPer100Cal"] = 100 * df["CarbohydrateContent"] / (df["Calories"] + eps)
    df["CholesterolPer100Cal"] = 100 * df["CholesterolContent"] / (df["Calories"] + eps)

    df["SugarShareOfCarbs"] = df["SugarContent"] / (df["CarbohydrateContent"] + eps)
    df["SatFatShareOfFat"] = df["SaturatedFatContent"] / (df["FatContent"] + eps)

    return df


def select_features_for_landscape(df: pd.DataFrame) -> list[str]:
    features = [
        "Calories",
        "FatContent",
        "SaturatedFatContent",
        "SodiumContent",
        "CarbohydrateContent",
        "FiberContent",
        "SugarContent",
        "ProteinContent",
        "CholesterolContent",
        "ProteinPer100Cal",
        "FiberPer100Cal",
        "SugarPer100Cal",
        "SodiumPer100Cal",
        "CholesterolPer100Cal",
        "CarbPer100Cal",
        "FatPer100Cal",
        "SugarShareOfCarbs",
        "SatFatShareOfFat",
    ]
    return [f for f in features if f in df.columns]


def transform_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()

    skewed_cols = [
        "Calories",
        "FatContent",
        "SaturatedFatContent",
        "SodiumContent",
        "CarbohydrateContent",
        "FiberContent",
        "SugarContent",
        "ProteinContent",
        "CholesterolContent",
        "ProteinPer100Cal",
        "FiberPer100Cal",
        "SugarPer100Cal",
        "SodiumPer100Cal",
        "CholesterolPer100Cal",
        "CarbPer100Cal",
        "FatPer100Cal",
    ]

    for col in skewed_cols:
        if col in X.columns:
            X[col] = np.log1p(X[col])

    for col in ["SugarShareOfCarbs", "SatFatShareOfFat"]:
        if col in X.columns:
            X[col] = X[col].clip(lower=0, upper=X[col].quantile(0.99))

    return X


def compute_pca_and_clusters(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_clusters: int,
    random_state: int,
):

    X_raw = transform_features(df, feature_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    pca = PCA(n_components=2, random_state=random_state)
    pcs = pca.fit_transform(X_scaled)

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=random_state,
        n_init=10,
        max_iter=10,
    )
    clusters = gmm.fit_predict(X_scaled)
    cluster_probs = gmm.predict_proba(X_scaled).max(axis=1)

    out = df.copy()
    out["PC1"] = pcs[:, 0]
    out["PC2"] = pcs[:, 1]
    out["Cluster"] = clusters.astype(str)
    out["ClusterConfidence"] = cluster_probs

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=["PC1_loading", "PC2_loading"],
    ).reset_index(names="Feature")

    explained = pd.DataFrame(
        {
            "Component": ["PC1", "PC2"],
            "ExplainedVarianceRatio": pca.explained_variance_ratio_,
        }
    )

    return out, loadings, explained


def get_top_categories(df: pd.DataFrame, n: int = 8) -> list[str]:
    return df["RecipeCategory"].value_counts().head(n).index.tolist()


# ##################
# Shared styling helpers
# ##################


def build_discrete_colorscale(colors: list[str]) -> list[list[float | str]]:
    n = len(colors)
    if n <= 1:
        return [[0.0, colors[0]], [1.0, colors[0]]]

    stepped = []
    for i, color in enumerate(colors):
        left = i / n
        right = (i + 1) / n
        stepped.append([left, color])
        stepped.append([right, color])
    return stepped


def add_discrete_cluster_colorbar(fig: go.Figure, cluster_order: list[str], colors: list[str]) -> go.Figure:
    n = len(cluster_order)
    fig.update_layout(
        coloraxis=dict(
            colorscale=build_discrete_colorscale(colors[:n]),
            cmin=-0.5,
            cmax=n - 0.5,
            colorbar=dict(
                title=dict(text="Cluster", side="right"),
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.1,
                yanchor="top",
                len=0.62,
                thickness=25,
                tickmode="array",
                tickvals=list(range(n)),
                ticktext=[str(c) for c in cluster_order],
                outlinecolor="black",
                outlinewidth=1,
                ticklen=6,
                tickcolor="rgba(0,0,0,0.65)",
            ),
        )
    )
    return fig


def apply_trace_matched_hoverlabels(fig: go.Figure) -> go.Figure:
    """Match each trace hover-label background to its plotted color when possible."""
    for trace in fig.data:
        trace_type = getattr(trace, "type", None)

        if trace_type in {"scatter", "scattergl"}:
            marker = getattr(trace, "marker", None)
            if marker is not None:
                color = getattr(marker, "color", None)
                if isinstance(color, str):
                    trace.hoverlabel = dict(bgcolor=color)

        elif trace_type == "bar":
            marker = getattr(trace, "marker", None)
            if marker is not None:
                color = getattr(marker, "color", None)
                if isinstance(color, str):
                    trace.hoverlabel = dict(bgcolor=color)

        elif trace_type == "contour":
            line = getattr(trace, "line", None)
            if line is not None:
                color = getattr(line, "color", None)
                if isinstance(color, str):
                    trace.hoverlabel = dict(bgcolor=color)

        elif trace_type == "heatmap":
            # Heatmap cells vary by value, so use a neutral white box for readability.
            trace.hoverlabel = dict(bgcolor="white", font=dict(color="#243447"))

    return fig


def compute_pca_axis_ranges(plot_df: pd.DataFrame, pad_frac: float = 0.06) -> dict:
    x_min, x_max = plot_df["PC1"].min(), plot_df["PC1"].max()
    y_min, y_max = plot_df["PC2"].min(), plot_df["PC2"].max()

    x_pad = (x_max - x_min) * pad_frac if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * pad_frac if y_max > y_min else 1.0

    return {
        "x_range": [float(x_min - x_pad), float(x_max + x_pad)],
        "y_range": [float(y_min - y_pad), float(y_max + y_pad)],
    }


def smooth_histogram2d(hist: np.ndarray, n_passes: int = 3) -> np.ndarray:
    """Apply a small separable smoothing kernel without introducing a SciPy dependency."""
    smoothed = hist.astype(float, copy=True)
    kernel = np.array([1.0, 2.0, 1.0], dtype=float)
    kernel = kernel / kernel.sum()

    for _ in range(n_passes):
        padded = np.pad(smoothed, ((0, 0), (1, 1)), mode="edge")
        smoothed = (
            kernel[0] * padded[:, :-2]
            + kernel[1] * padded[:, 1:-1]
            + kernel[2] * padded[:, 2:]
        )
        padded = np.pad(smoothed, ((1, 1), (0, 0)), mode="edge")
        smoothed = (
            kernel[0] * padded[:-2, :]
            + kernel[1] * padded[1:-1, :]
            + kernel[2] * padded[2:, :]
        )
    return smoothed


THREE_SIGMA_MASS_FRACTION = 0.9973002039367398


def compute_three_sigma_density_region(
    cluster_df: pd.DataFrame,
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    smoothing_passes: int = 3,
) -> dict | None:
    """Return the smoothed 2D field and the single density threshold enclosing ~3-sigma mass."""
    if len(cluster_df) < 5:
        return None

    hist, _, _ = np.histogram2d(
        cluster_df["PC1"].to_numpy(),
        cluster_df["PC2"].to_numpy(),
        bins=[x_edges, y_edges],
    )
    z = smooth_histogram2d(hist.T, n_passes=smoothing_passes)
    if not np.isfinite(z).any() or float(z.max()) <= 0:
        return None

    z = np.where(np.isfinite(z), z, 0.0)
    total = float(z.sum())
    if total <= 0:
        return None

    flat = z.ravel()
    positive = flat[flat > 0]
    if len(positive) == 0:
        return None

    sorted_positive = np.sort(positive)[::-1]
    cumulative = np.cumsum(sorted_positive) / total
    idx = int(np.searchsorted(cumulative, THREE_SIGMA_MASS_FRACTION, side="left"))
    idx = min(max(idx, 0), len(sorted_positive) - 1)
    threshold = float(sorted_positive[idx])

    mask = z >= threshold
    return {
        "z": z,
        "threshold": threshold,
        "mask": mask,
    }


def sample_representative_points(
    cluster_df: pd.DataFrame,
    *,
    max_points: int = 300,
    random_state: int = 42,
    n_bins: int = 14,
    min_confidence: float = 0.99,
) -> pd.DataFrame:
    """Sample at most max_points recipes with ClusterConfidence > min_confidence."""
    if len(cluster_df) == 0:
        return cluster_df.copy()

    work = cluster_df.loc[cluster_df["ClusterConfidence"] > min_confidence].copy()
    if len(work) == 0:
        return work.copy()

    if len(work) <= max_points:
        return work.copy()

    x_edges_local = np.linspace(work["PC1"].min(), work["PC1"].max(), n_bins + 1)
    y_edges_local = np.linspace(work["PC2"].min(), work["PC2"].max(), n_bins + 1)

    if np.unique(x_edges_local).size < 3 or np.unique(y_edges_local).size < 3:
        return work.sample(max_points, random_state=random_state).copy()

    x_bin = np.clip(np.digitize(work["PC1"].to_numpy(), x_edges_local[1:-1], right=False), 0, n_bins - 1)
    y_bin = np.clip(np.digitize(work["PC2"].to_numpy(), y_edges_local[1:-1], right=False), 0, n_bins - 1)
    work["__sample_bin__"] = [f"{xb}_{yb}" for xb, yb in zip(x_bin, y_bin)]

    counts = work["__sample_bin__"].value_counts().sort_values(ascending=False)
    n_nonempty = int(len(counts))
    if n_nonempty == 0:
        return work.sample(max_points, random_state=random_state).copy()

    base_take = max(1, max_points // n_nonempty)
    chosen_parts = []
    used_idx = []

    for i, bin_key in enumerate(counts.index.tolist()):
        bin_df = work[work["__sample_bin__"] == bin_key]
        take = min(len(bin_df), base_take)
        if i < (max_points - base_take * n_nonempty):
            take = min(len(bin_df), take + 1)
        part = bin_df.sample(take, random_state=random_state + i)
        chosen_parts.append(part)
        used_idx.extend(part.index.tolist())

    sampled = pd.concat(chosen_parts, axis=0) if chosen_parts else work.iloc[:0].copy()
    remaining = max_points - len(sampled)
    if remaining > 0:
        leftover = work.drop(index=used_idx, errors="ignore")
        if len(leftover) > 0:
            extra = leftover.sample(min(remaining, len(leftover)), random_state=random_state + 10_000)
            sampled = pd.concat([sampled, extra], axis=0)

    return sampled.drop(columns="__sample_bin__", errors="ignore").copy()


def add_cluster_contours(
    fig: go.Figure,
    plot_df: pd.DataFrame,
    meta: dict,
    *,
    grid_size: int = 85,
    smoothing_passes: int = 3,
) -> tuple[go.Figure, dict[str, dict]]:
    cluster_order = meta["cluster_order"]
    x_range = meta["x_range"]
    y_range = meta["y_range"]
    x_edges = np.linspace(x_range[0], x_range[1], grid_size + 1)
    y_edges = np.linspace(y_range[0], y_range[1], grid_size + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    contour_meta = {}

    for idx, cluster in enumerate(cluster_order):
        cluster_df = plot_df[plot_df["Cluster"].astype(str) == str(cluster)]
        region = compute_three_sigma_density_region(
            cluster_df,
            x_edges=x_edges,
            y_edges=y_edges,
            smoothing_passes=smoothing_passes,
        )
        if region is None:
            continue

        z = region["z"]
        threshold = region["threshold"]
        contour_meta[str(cluster)] = {"threshold": threshold}

        fig.add_trace(
            go.Contour(
                x=x_centers,
                y=y_centers,
                z=z,
                name=f"Cluster {cluster}",
                showscale=False,
                showlegend=False,
                hoverinfo="skip",
                contours=dict(
                    start=threshold,
                    end=threshold,
                    size=1e-9,
                    coloring="none",
                    showlabels=False,
                ),
                line=dict(
                    color=STYLE["discrete_cluster_colors"][idx % len(STYLE["discrete_cluster_colors"])],
                    width=2.2,
                ),
                opacity=0.95,
            )
        )

    return fig, contour_meta


def apply_shared_pca_layout(fig: go.Figure, meta: dict, *, use_colorbar: bool = False) -> go.Figure:
    explained_pc1 = meta["explained_variance"]["PC1"]
    explained_pc2 = meta["explained_variance"]["PC2"]

    # fig.update_traces(
    #     marker=dict(
    #         size=STYLE["pca_marker_size"],
    #         opacity=STYLE["pca_marker_opacity"],
    #         line=dict(width=0.35, color="rgba(255,255,255,0.35)"),
    #     )
    # )

    fig.update_xaxes(
        title=f"PC1 ({explained_pc1 * 100:.1f}% variance explained)",
        side="top",
        range=meta["x_range"],
        showgrid=False,
        zeroline=False,
        showline=True,
        ticks="",
        automargin=True,
        title_standoff=18,
    )

    fig.update_yaxes(
        title=f"PC2 ({explained_pc2 * 100:.1f}% variance explained)",
        range=meta["y_range"],
        showgrid=False,
        zeroline=False,
        showline=True,
        ticks="",
        automargin=True,
        title_standoff=18,
    )

    fig.update_layout(
        showlegend=not use_colorbar,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
        font=dict(size=14, color="#243447"),
        margin=dict(l=85, r=60, t=50, b=50),
    )
    return fig


# ##################
# Figure builders used by the app
# ##################


def build_fig_pca_landscape(plot_df: pd.DataFrame, meta: dict) -> go.Figure:
    cluster_order = meta["cluster_order"]
    cluster_to_num = {cluster: i for i, cluster in enumerate(cluster_order)}
    plot_df = plot_df.copy()
    plot_df["ClusterNum"] = plot_df["Cluster"].astype(str).map(cluster_to_num)

    fig = go.Figure()
    fig, contour_meta = add_cluster_contours(fig, plot_df, meta)

    sampled_parts = []
    for i, cluster in enumerate(cluster_order):
        cluster_df = plot_df[plot_df["Cluster"].astype(str) == str(cluster)].copy()
        cluster_key = str(cluster)
        if len(cluster_df) == 0 or cluster_key not in contour_meta:
            continue
        sampled = sample_representative_points(
            cluster_df,
            max_points=100,
            random_state=RUN_CONFIG["random_state"] + i,
            min_confidence=0.99,
        )
        sampled_parts.append(sampled)

    sample_df = pd.concat(sampled_parts, axis=0, ignore_index=True) if sampled_parts else plot_df.iloc[:0].copy()

    if len(sample_df) > 0:
        fig.add_trace(
            go.Scattergl(
                x=sample_df["PC1"],
                y=sample_df["PC2"],
                mode="markers",
                showlegend=False,
                hovertext=sample_df["Name"],
                customdata=sample_df[
                    [
                        "RecipeCategory",
                        "Calories",
                        "ProteinContent",
                        "CarbohydrateContent",
                        "SugarContent",
                        "PC1",
                        "PC2",
                        "Cluster",
                        "ClusterConfidence",
                    ]
                ].to_numpy(),
                marker=dict(
                    size=STYLE["pca_marker_size"],
                    opacity=STYLE["pca_marker_opacity"],
                    color=sample_df["ClusterNum"],
                    coloraxis="coloraxis",
                    line=dict(width=0.35, color="rgba(255,255,255,0.35)"),
                ),
                hovertemplate=(
                    "<b>%{hovertext}</b><br>"
                    "Category: %{customdata[0]}<br>"
                    "Calories: %{customdata[1]:.1f} kcal<br>"
                    "Protein: %{customdata[2]:.1f} g<br>"
                    "Carbohydrates: %{customdata[3]:.1f} g<br>"
                    "Sugar: %{customdata[4]:.1f} g<br>"
                    "Cluster: %{customdata[7]}<br>"
                    "Confidence: %{customdata[8]:.3f}<br>"
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    fig = apply_shared_pca_layout(fig, meta, use_colorbar=True)
    fig.update_layout(
        margin=dict(l=120, r=70, t=0, b=0),
        coloraxis_colorbar=dict(y=-0.2),
    )

    fig = add_discrete_cluster_colorbar(fig, cluster_order, STYLE["discrete_cluster_colors"])
    return apply_trace_matched_hoverlabels(fig)


def build_fig_pca_categories(plot_df: pd.DataFrame, meta: dict) -> go.Figure:
    category_order = meta["category_order"]
    category_to_num = {category: i for i, category in enumerate(category_order)}
    plot_df = plot_df.copy()
    plot_df["CategoryNum"] = plot_df["RecipeCategory"].map(category_to_num)

    fig = go.Figure()
    fig, _ = add_cluster_contours(fig, plot_df, meta)

    sampled_parts = []
    for i, cluster in enumerate(meta["cluster_order"]):
        cluster_df = plot_df[plot_df["Cluster"].astype(str) == str(cluster)].copy()
        if len(cluster_df) == 0:
            continue
        sampled = sample_representative_points(
            cluster_df,
            max_points=100,
            random_state=RUN_CONFIG["random_state"] + i,
            min_confidence=0.99,
        )
        sampled_parts.append(sampled)

    sample_df = pd.concat(sampled_parts, axis=0, ignore_index=True) if sampled_parts else plot_df.iloc[:0].copy()

    if len(sample_df) > 0:
        fig.add_trace(
            go.Scattergl(
                x=sample_df["PC1"],
                y=sample_df["PC2"],
                mode="markers",
                showlegend=False,
                hovertext=sample_df["Name"],
                customdata=sample_df[
                    [
                        "RecipeCategory",
                        "Calories",
                        "ProteinContent",
                        "CarbohydrateContent",
                        "SugarContent",
                        "PC1",
                        "PC2",
                        "Cluster",
                        "ClusterConfidence",
                    ]
                ].to_numpy(),
                marker=dict(
                    size=STYLE["pca_marker_size"],
                    opacity=STYLE["pca_marker_opacity"],
                    color=sample_df["CategoryNum"],
                    colorscale=build_discrete_colorscale(
                        [
                            STYLE["category_colors"][i % len(STYLE["category_colors"])]
                            for i in range(len(category_order))
                        ]
                    ),
                    cmin=-0.5,
                    cmax=max(len(category_order) - 0.5, 0.5),
                    line=dict(width=0.35, color="rgba(255,255,255,0.35)"),
                ),
                hovertemplate=(
                    "<b>%{hovertext}</b><br>"
                    "Category: %{customdata[0]}<br>"
                    "Calories: %{customdata[1]:.1f} kcal<br>"
                    "Protein: %{customdata[2]:.1f} g<br>"
                    "Carbohydrates: %{customdata[3]:.1f} g<br>"
                    "Sugar: %{customdata[4]:.1f} g<br>"
                    "Cluster: %{customdata[7]}<br>"
                    "Confidence: %{customdata[8]:.3f}<br>"
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    fig = apply_shared_pca_layout(fig, meta, use_colorbar=False)
    fig.update_layout(
        legend=dict(
            title=None,
            orientation="h",
            x=0.5,
            y=1.12,
            xanchor="center",
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            tracegroupgap=8,
        ),
        margin=dict(l=78, r=62, t=110, b=120),
    )

    fig.update_xaxes(
        showline=True,
        linecolor="rgba(0,0,0,0.35)",
        linewidth=1.2,
        ticks="outside",
        side="bottom",
    )
    fig.update_yaxes(
        showline=True,
        linecolor="rgba(0,0,0,0.35)",
        linewidth=1.2,
        ticks="outside",
    )

    for i, category in enumerate(category_order):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=category,
                marker=dict(
                    size=10,
                    color=STYLE["category_colors"][i % len(STYLE["category_colors"])],
                    line=dict(width=0.35, color="rgba(255,255,255,0.35)"),
                ),
                hoverinfo="skip",
                showlegend=True,
            )
        )

    return apply_trace_matched_hoverlabels(fig)


def build_fig_cluster_heatmap(plot_df: pd.DataFrame, meta: dict) -> go.Figure:
    plot_df = plot_df.copy()
    plot_df["Cluster"] = plot_df["Cluster"].astype(str)
    plot_df["Feature"] = plot_df["Feature"].astype(str)

    x_left = meta["content_feature_order"]
    x_right = meta["per100cal_feature_order"]
    y = [str(v) for v in meta["cluster_order"]]

    zmat = plot_df.pivot(index="Cluster", columns="Feature", values="ZMedian").reindex(index=y)
    raw_med = plot_df.pivot(index="Cluster", columns="Feature", values="MedianRaw").reindex(index=y)

    z_left = zmat.reindex(columns=x_left)
    z_right = zmat.reindex(columns=x_right)
    raw_left = raw_med.reindex(columns=x_left)
    raw_right = raw_med.reindex(columns=x_right)

    zabs = np.nanmax(np.abs(zmat.values))
    if not np.isfinite(zabs) or zabs == 0:
        zabs = 1.0

    def make_hover(raw_df, z_df):
        hover = np.empty(z_df.shape, dtype=object)
        for i, cluster in enumerate(z_df.index):
            for j, feat in enumerate(z_df.columns):
                hover[i, j] = (
                    f"Cluster: {cluster}<br>"
                    f"Feature: {feat}<br>"
                    f"Z-scored median: {z_df.iloc[i, j]:.2f}<br>"
                    f"Raw median: {raw_df.iloc[i, j]:.2f}"
                )
        return hover

    hover_left = make_hover(raw_left, z_left)
    hover_right = make_hover(raw_right, z_right)
    text_left = np.vectorize(lambda v: f"{v:.2f}")(z_left.values)
    text_right = np.vectorize(lambda v: f"{v:.2f}")(z_right.values)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z_left.values,
            x=x_left,
            y=y,
            hovertext=hover_left,
            hoverinfo="text",
            text=text_left,
            texttemplate="%{text}",
            textfont=dict(color="white", size=16),
            colorscale="RdBu",
            zmid=0,
            zmin=-zabs,
            zmax=zabs,
            xaxis="x",
            yaxis="y",
            showscale=False,
        )
    )
    fig.add_trace(
        go.Heatmap(
            z=z_right.values,
            x=x_right,
            y=y,
            hovertext=hover_right,
            hoverinfo="text",
            text=text_right,
            texttemplate="%{text}",
            textfont=dict(color="white", size=14),
            colorscale="RdBu",
            zmid=0,
            zmin=-zabs,
            zmax=zabs,
            xaxis="x2",
            yaxis="y2",
            colorbar=dict(
                title=dict(text="Z-score", side="right"),
                orientation="h",
                x=0.5,
                xanchor="center",
                y=1.08,
                yanchor="bottom",
                len=0.62,
                thickness=25,
                tickmode="auto",
                ticks="outside",
                ticklabelposition="outside top",
                outlinecolor="black",
                outlinewidth=1,
                ticklen=6,
                tickcolor="rgba(0,0,0,0.65)",
            ),
        )
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=14, color="#243447"),
        margin=dict(l=30, r=40, t=80, b=80),
        xaxis=dict(
            domain=[0.00, 0.48],
            title="Content",
            tickangle=-20,
            showgrid=False,
            zeroline=False,
            side="bottom",
            showline=False,
            linecolor="black",
            linewidth=1.5,
        ),
        xaxis2=dict(
            domain=[0.50, 1.00],
            title="Per 100 Calories",
            tickangle=-20,
            showgrid=False,
            zeroline=False,
            side="bottom",
            showline=False,
            linecolor="black",
            linewidth=1.5,
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            automargin=False,
            showline=False,
            linecolor="black",
            linewidth=1.5,
        ),
        yaxis2=dict(
            autorange="reversed",
            matches="y",
            showticklabels=False,
            showline=False,
            linecolor="black",
            linewidth=1.5,
        ),
        shapes=[
            dict(type="rect", xref="paper", yref="paper", x0=0.00, x1=0.48, y0=0.0, y1=1.0,
                 line=dict(color="black", width=1.5), layer="above"),
            dict(type="rect", xref="paper", yref="paper", x0=0.50, x1=1.00, y0=0.0, y1=1.0,
                 line=dict(color="black", width=1.5), layer="above"),
        ],
    )

    fig.add_annotation(
        text="Nutrition Cluster",
        xref="paper", yref="paper",
        x=-0.02,
        y=0.5,
        showarrow=False,
        textangle=-90,
        font=dict(size=15, color="#243447"),
        xanchor="center",
        yanchor="middle",
    )
    return apply_trace_matched_hoverlabels(fig)


def build_fig_cluster_categories(plot_df: pd.DataFrame, meta: dict) -> go.Figure:
    plot_df = plot_df.copy()
    plot_df["Cluster"] = plot_df["Cluster"].astype(str)
    plot_df["Category"] = plot_df["Category"].astype(str)

    cluster_order = [str(c) for c in meta["cluster_order"]]
    category_order = meta["category_order"]
    color_map = {
        cat: STYLE["category_colors"][i % len(STYLE["category_colors"])]
        for i, cat in enumerate(category_order)
    }

    fig = go.Figure()
    for cat in category_order:
        sub = plot_df[plot_df["Category"] == cat].copy()
        x_vals, y_vals, text_vals, custom_counts = [], [], [], []
        for cluster in cluster_order:
            row = sub[sub["Cluster"] == cluster]
            if len(row) == 0:
                x_vals.append(0.0)
                y_vals.append(cluster)
                text_vals.append("")
                custom_counts.append(0)
            else:
                prop = float(row["Proportion"].iloc[0])
                cnt = int(row["Count"].iloc[0])
                x_vals.append(prop)
                y_vals.append(cluster)
                text_vals.append(f"{prop:.1%}" if prop >= 0.06 else "")
                custom_counts.append(cnt)

        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=y_vals,
                name=cat,
                orientation="h",
                marker=dict(
                    color=color_map[cat],
                    line=dict(color="white", width=0.8),
                ),
                text=text_vals,
                textposition="inside",
                customdata=np.array(custom_counts).reshape(-1, 1),
                hovertemplate=(
                    "Cluster: %{y}<br>"
                    f"Category: {cat}<br>"
                    "Share within cluster: %{x:.1%}<br>"
                    "Recipe count: %{customdata[0]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=14, color="#243447"),
        margin=dict(l=90, r=40, t=75, b=70),
        legend=dict(
            title=None,
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )

    fig.update_xaxes(
        title="Share of Recipes within Cluster",
        tickformat=".0%",
        range=[0, 1],
        showgrid=False,
        zeroline=False,
        showline=False,
    )
    fig.update_yaxes(
        title="Cluster",
        categoryorder="array",
        categoryarray=cluster_order[::-1],
        showgrid=False,
        zeroline=False,
        showline=False,
    )
    return apply_trace_matched_hoverlabels(fig)


def build_fig_loadings(plot_df: pd.DataFrame, meta: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["Feature"],
            y=plot_df["PC1_loading"],
            name="PC1 loading",
            marker=dict(color="#f4a3a3"),
        )
    )
    fig.add_trace(
        go.Bar(
            x=plot_df["Feature"],
            y=plot_df["PC2_loading"],
            name="PC2 loading",
            marker=dict(color="#9ec5fe"),
        )
    )

    fig.update_layout(
        barmode="group",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
        font=dict(size=14, color="#243447"),
        margin=dict(l=60, r=30, t=30, b=95),
        legend=dict(
            title=None,
            orientation="h",
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )
    fig.update_xaxes(
        title="Feature",
        tickangle=-35,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="rgba(0,0,0,0.20)",
        ticks="outside",
    )
    fig.update_yaxes(
        title="Loading",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="rgba(0,0,0,0.20)",
        ticks="outside",
    )
    return apply_trace_matched_hoverlabels(fig)


# ##################
# App export prep
# ##################


def build_pca_landscape_inputs(df_landscape: pd.DataFrame, explained_df: pd.DataFrame):
    plot_df = df_landscape[
        [
            "RecipeId", "Name", "RecipeCategory",
            # "AggregatedRating", "ReviewCount",
            "Calories", "ProteinContent", "CarbohydrateContent", "SugarContent",
            # "SodiumContent", "CholesterolContent", "FiberContent", "FatContent", "SaturatedFatContent",
            "PC1", "PC2", "Cluster", "ClusterConfidence",
        ]
    ].copy()

    evr = dict(zip(explained_df["Component"], explained_df["ExplainedVarianceRatio"]))
    axis_ranges = compute_pca_axis_ranges(plot_df)
    meta = {
        "explained_variance": evr,
        "cluster_order": sorted(plot_df["Cluster"].astype(str).unique(), key=lambda x: int(x)),
        "x_range": axis_ranges["x_range"],
        "y_range": axis_ranges["y_range"],
    }
    return plot_df, meta


def build_pca_categories_inputs(df_landscape: pd.DataFrame, explained_df: pd.DataFrame, top_categories: list[str]):
    plot_df = df_landscape[df_landscape["RecipeCategory"].isin(top_categories)].copy()
    plot_df = plot_df[
        [
            "RecipeId", "Name", "RecipeCategory",
            # "AggregatedRating", "ReviewCount",
            "Calories", "ProteinContent", "CarbohydrateContent", "SugarContent",
            # "SodiumContent", "CholesterolContent", "FiberContent", "FatContent", "SaturatedFatContent",
            "PC1", "PC2", "Cluster", "ClusterConfidence",
        ]
    ].copy()

    evr = dict(zip(explained_df["Component"], explained_df["ExplainedVarianceRatio"]))
    axis_ranges = compute_pca_axis_ranges(df_landscape)
    meta = {
        "explained_variance": evr,
        "category_order": top_categories,
        "cluster_order": sorted(plot_df["Cluster"].astype(str).unique(), key=lambda x: int(x)),
        "x_range": axis_ranges["x_range"],
        "y_range": axis_ranges["y_range"],
    }
    return plot_df, meta


def build_cluster_heatmap_inputs(df_landscape: pd.DataFrame):
    content_cols = [
        "Calories",
        "FatContent",
        "SugarContent",
        "ProteinContent",
        "FiberContent",
        "SodiumContent",
        "CarbohydrateContent",
        "CholesterolContent",
    ]
    per100cal_cols = [
        "Calories",
        "FatPer100Cal",
        "SugarPer100Cal",
        "ProteinPer100Cal",
        "FiberPer100Cal",
        "SodiumPer100Cal",
        "CarbPer100Cal",
        "CholesterolPer100Cal",
    ]
    all_profile_cols = list(dict.fromkeys(content_cols + per100cal_cols))

    cluster_profile = df_landscape.groupby("Cluster")[all_profile_cols].median().copy()
    cluster_profile = cluster_profile.sort_index(key=lambda x: x.astype(int))

    std = cluster_profile.std(ddof=0).replace(0, 1.0)
    cluster_profile_z = (cluster_profile - cluster_profile.mean()) / std

    long_rows = []
    for cluster in cluster_profile.index:
        for feat in all_profile_cols:
            long_rows.append(
                {
                    "Cluster": str(cluster),
                    "Feature": feat,
                    "MedianRaw": float(cluster_profile.loc[cluster, feat]),
                    "ZMedian": float(cluster_profile_z.loc[cluster, feat]),
                }
            )

    plot_df = pd.DataFrame(long_rows)
    meta = {
        "content_feature_order": content_cols,
        "per100cal_feature_order": per100cal_cols,
        "cluster_order": [str(x) for x in cluster_profile.index.tolist()],
    }
    return plot_df, meta


def build_cluster_categories_inputs(df_landscape: pd.DataFrame):
    df = df_landscape.copy()
    df["Cluster"] = df["Cluster"].astype(str)
    top_n = 5

    counts = df.groupby(["Cluster", "RecipeCategory"]).size().reset_index(name="Count")
    counts["Proportion"] = counts.groupby("Cluster")["Count"].transform(lambda x: x / x.sum())
    counts = (
        counts.sort_values(["Cluster", "Proportion", "RecipeCategory"], ascending=[True, False, True])
        .groupby("Cluster", group_keys=False)
        .head(top_n)
        .copy()
    )

    plot_df = counts.rename(columns={"RecipeCategory": "Category"})
    cluster_order = sorted(plot_df["Cluster"].unique(), key=int)
    category_order = (
        plot_df.groupby("Category")["Proportion"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    meta = {
        "cluster_order": cluster_order,
        "category_order": category_order,
        "top_n_per_cluster": top_n,
    }
    return plot_df, meta


def build_pca_loadings_inputs(loadings_df: pd.DataFrame):
    return loadings_df.copy(), {}


# ##################
# Full Plotly JSON export
# ##################


def figure_to_json_dict(fig: go.Figure) -> dict:
    return fig.to_plotly_json()


def build_all_figures(df_landscape: pd.DataFrame, loadings_df: pd.DataFrame, explained_df: pd.DataFrame, top_categories: list[str]) -> dict[str, dict]:
    standalone = {}

    plot_df, meta = build_pca_landscape_inputs(df_landscape, explained_df)
    standalone["nutrition_pca_landscape"] = figure_to_json_dict(build_fig_pca_landscape(plot_df, meta))

    plot_df, meta = build_cluster_heatmap_inputs(df_landscape)
    standalone["nutrition_cluster_heatmap"] = figure_to_json_dict(build_fig_cluster_heatmap(plot_df, meta))

    plot_df, meta = build_pca_loadings_inputs(loadings_df)
    standalone["nutrition_pca_loadings"] = figure_to_json_dict(build_fig_loadings(plot_df, meta))

    plot_df, meta = build_pca_categories_inputs(df_landscape, explained_df, top_categories)
    standalone["nutrition_pca_categories"] = figure_to_json_dict(build_fig_pca_categories(plot_df, meta))

    plot_df, meta = build_cluster_categories_inputs(df_landscape)
    standalone["nutrition_cluster_categories"] = figure_to_json_dict(build_fig_cluster_categories(plot_df, meta))

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
    print("  Nutritional PCA Landscape Pipeline")
    print("══════════════════════════════════════════════════════\n")

    df = load_recipes_from_sqlite(
        db_path=config["db_path"],
        table=config["table_name"],
        min_reviews=config["min_reviews"],
    )
    df = basic_recipe_cleaning(
        df,
        min_reviews=config["min_reviews"],
        max_missing_allowed=config["max_missing_allowed"],
    )
    df = add_derived_features(df)

    print("🧠 Selecting PCA features...")
    feature_cols = select_features_for_landscape(df)
    print(f"    Feature count      : {len(feature_cols)}")

    print("📉 Computing PCA and clustering...")
    df_landscape, loadings_df, explained_df = compute_pca_and_clusters(
        df=df,
        feature_cols=feature_cols,
        n_clusters=config["n_clusters"],
        random_state=config["random_state"],
    )
    top_categories = get_top_categories(df_landscape, n=config["top_n_categories"])

    print("🎨 Building figures...")
    standalone_figures = build_all_figures(df_landscape, loadings_df, explained_df, top_categories)

    payload = {
        "visualization": "nutritional_landscape",
        "webapp_panels": WEBAPP_PANELS,
        "standalone_figures": standalone_figures,
        "meta": {
            "recipe_file": str(config["db_path"]),
            "output_dir": str(output_dir),
            "top_n_categories": config["top_n_categories"],
            "top_categories": top_categories,
            "n_clusters": config["n_clusters"],
            "min_reviews": config["min_reviews"],
            "max_missing_allowed": config["max_missing_allowed"],
            "feature_cols": feature_cols,
            "n_recipes_after_cleaning": int(len(df_landscape)),
            "used_fig_keys": list(USED_FIG_KEYS),
        },
    }

    print("💾 Exporting Plotly payload...")
    out_path = export_plotly_payload(payload, output_dir, config["json_filename"])

    print("📋 Summary")
    print(f"    Recipes retained   : {len(df_landscape):,}")
    print(f"    Top categories     : {', '.join(top_categories)}")
    print(f"    Figures exported   : {len(standalone_figures)}")
    print(f"    Output file        : {out_path}")
    print("✅ Done.")

    return {"output_path": str(out_path), "figure_keys": list(standalone_figures.keys())}


if __name__ == "__main__":
    main()
