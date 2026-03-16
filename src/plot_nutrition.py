# ============================================================
# Visualization 3 — The Nutritional Landscape
# Export-first pipeline for web-app replotting
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import plotly.express as px
import plotly.graph_objects as go


# -----------------------------
# USER SETTINGS
# -----------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RECIPE_FILE = BASE_DIR / "data" / "tables" / "food_recipe.db"
OUTPUT_DIR = BASE_DIR / "plots" / "nutritional_landscape_outputs"

TOP_N_CATEGORIES = 8
N_CLUSTERS = 5
MIN_REVIEWS = 3
MAX_MISSING_ALLOWED = 2
RANDOM_STATE = 42

# web-app / export settings
EXPORT_PARQUET = False   # True if pyarrow is installed and you want smaller/faster files
WRITE_HTML = True
WRITE_PNG = True         # Requires kaleido installed
WRITE_PLOTLY_JSON = True


# -----------------------------
# EXPECTED COLUMNS
# -----------------------------
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
    "AggregatedRating",
    "ReviewCount",
    "RecipeServings",
]


# -----------------------------
# PATH HELPERS
# -----------------------------
def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def figure_dir(root_output_dir: str | Path, fig_key: str) -> Path:
    path = Path(root_output_dir) / "figures" / fig_key
    ensure_dir(path)
    return path

def safe_write_json(obj, path: str | Path):
    def _json_default(x):
        if isinstance(x, Path):
            return str(x)
        raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)

def safe_read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def export_dataframe(df: pd.DataFrame, path_no_ext: str | Path, use_parquet: bool = False):
    path_no_ext = Path(path_no_ext)

    if use_parquet:
        try:
            out = path_no_ext.with_suffix(".parquet")
            df.to_parquet(out, index=False)
            return str(out)
        except Exception:
            pass

    out = path_no_ext.with_suffix(".csv")
    df.to_csv(out, index=False)
    return str(out)


def load_exported_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


# -----------------------------
# DATA LOADING
# -----------------------------
def load_recipes_from_sqlite(db_path, table="recipes"):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)

    query = f"""
        SELECT
            RecipeId,
            Name,
            RecipeCategory,
            AggregatedRating,
            ReviewCount,
            RecipeServings,
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
            AND ReviewCount >= {MIN_REVIEWS}
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# -----------------------------
# CLEANING / FEATURES
# -----------------------------
def clean_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def basic_recipe_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in META_COLS:
        if col not in df.columns:
            df[col] = np.nan

    numeric_cols = NUTRITION_COLS + ["AggregatedRating", "ReviewCount", "RecipeServings"]
    df = clean_numeric_columns(df, numeric_cols)

    missing_count = df[NUTRITION_COLS].isna().sum(axis=1)
    df = df.loc[missing_count <= MAX_MISSING_ALLOWED].copy()

    for col in NUTRITION_COLS:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    if "ReviewCount" in df.columns:
        df = df[(df["ReviewCount"].fillna(0) >= MIN_REVIEWS)].copy()

    df = df[df["Calories"].fillna(0) > 0].copy()
    for col in NUTRITION_COLS:
        df = df[df[col].fillna(0) >= 0].copy()

    df["RecipeCategory"] = df["RecipeCategory"].fillna("Unknown").astype(str).str.strip()
    df.loc[df["RecipeCategory"] == "", "RecipeCategory"] = "Unknown"

    df["Name"] = df["Name"].fillna("Unknown Recipe").astype(str)

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6

    df["ProteinPer100Cal"] = 100 * df["ProteinContent"] / (df["Calories"] + eps)
    df["FiberPer100Cal"] = 100 * df["FiberContent"] / (df["Calories"] + eps)
    df["SugarPer100Cal"] = 100 * df["SugarContent"] / (df["Calories"] + eps)
    df["SodiumPer100Cal"] = 100 * df["SodiumContent"] / (df["Calories"] + eps)
    df["FatPer100Cal"] = 100 * df["FatContent"] / (df["Calories"] + eps)
    df["CarbPer100Cal"] = 100 * df["CarbohydrateContent"] / (df["Calories"] + eps)

    df["SugarShareOfCarbs"] = df["SugarContent"] / (df["CarbohydrateContent"] + eps)
    df["SatFatShareOfFat"] = df["SaturatedFatContent"] / (df["FatContent"] + eps)

    df["NutritionBalanceScore"] = (
        1.5 * df["ProteinPer100Cal"]
        + 1.5 * df["FiberPer100Cal"]
        - 1.0 * df["SugarPer100Cal"]
        - 0.002 * df["SodiumPer100Cal"]
    )

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
        "ProteinPer100Cal",
        "FiberPer100Cal",
        "SugarPer100Cal",
        "SodiumPer100Cal",
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
        "ProteinPer100Cal",
        "FiberPer100Cal",
        "SugarPer100Cal",
        "SodiumPer100Cal",
    ]

    for col in skewed_cols:
        if col in X.columns:
            X[col] = np.log1p(X[col])

    for col in ["SugarShareOfCarbs", "SatFatShareOfFat"]:
        if col in X.columns:
            X[col] = X[col].clip(lower=0, upper=X[col].quantile(0.99))

    return X


def compute_pca_and_clusters(df: pd.DataFrame, feature_cols: list[str], n_clusters: int = 5):
    X_raw = transform_features(df, feature_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    clusters = kmeans.fit_predict(X_scaled)

    out = df.copy()
    out["PC1"] = pcs[:, 0]
    out["PC2"] = pcs[:, 1]
    out["NutritionCluster"] = clusters.astype(str)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=["PC1_loading", "PC2_loading"]
    ).reset_index(names="Feature")

    explained = pd.DataFrame({
        "Component": ["PC1", "PC2"],
        "ExplainedVarianceRatio": pca.explained_variance_ratio_
    })

    return out, loadings, explained, scaler, pca, kmeans


def get_top_categories(df: pd.DataFrame, n: int = 8) -> list[str]:
    return df["RecipeCategory"].value_counts().head(n).index.tolist()


def summarize_category_spread(df: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    sub = df[df["RecipeCategory"].isin(categories)].copy()

    summary = (
        sub.groupby("RecipeCategory")
        .agg(
            n_recipes=("RecipeId", "count"),
            median_calories=("Calories", "median"),
            median_protein_per_100cal=("ProteinPer100Cal", "median"),
            median_fiber_per_100cal=("FiberPer100Cal", "median"),
            median_sugar_per_100cal=("SugarPer100Cal", "median"),
            median_sodium_per_100cal=("SodiumPer100Cal", "median"),
            calories_iqr=("Calories", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            sugar_iqr=("SugarContent", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            sodium_iqr=("SodiumContent", lambda x: x.quantile(0.75) - x.quantile(0.25)),
        )
        .sort_values("n_recipes", ascending=False)
        .reset_index()
    )
    return summary


# -----------------------------
# STYLING
# -----------------------------
def apply_food_theme(fig: go.Figure, title: str, subtitle: str | None = None) -> go.Figure:
    full_title = title if subtitle is None else f"{title}<br><sup>{subtitle}</sup>"

    fig.update_layout(
        title=dict(text=full_title, x=0.02, xanchor="left"),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
        font=dict(size=14),
        margin=dict(l=60, r=40, t=90, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.7)"
        )
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False)

    return fig


# -----------------------------
# EXPORT / RE-PLOT CORE
# -----------------------------
def export_figure_bundle(
    fig_key: str,
    plot_df: pd.DataFrame,
    metadata: dict,
    fig: go.Figure,
    output_dir: str,
    use_parquet: bool = False,
):
    fdir = figure_dir(output_dir, fig_key)

    data_path = export_dataframe(plot_df, os.path.join(fdir, "plot_data"), use_parquet=use_parquet)
    meta_path = os.path.join(fdir, "metadata.json")
    safe_write_json(metadata, meta_path)

    if WRITE_PLOTLY_JSON:
        fig.write_json(os.path.join(fdir, "figure.plotly.json"))

    if WRITE_HTML:
        fig.write_html(
            os.path.join(fdir, "figure.html"),
            include_plotlyjs="cdn",
            full_html=True
        )

    if WRITE_PNG:
        try:
            fig.write_image(os.path.join(fdir, "figure.png"), scale=2, width=1400, height=900)
        except Exception:
            pass

    return {
        "fig_key": fig_key,
        "data_path": data_path,
        "metadata_path": meta_path,
        "plotly_json_path": os.path.join(fdir, "figure.plotly.json") if WRITE_PLOTLY_JSON else None,
        "html_path": os.path.join(fdir, "figure.html") if WRITE_HTML else None,
        "png_path": os.path.join(fdir, "figure.png") if WRITE_PNG else None,
    }


# -----------------------------
# FIGURE BUILDERS
# -----------------------------
def build_fig_pca_landscape(plot_df: pd.DataFrame, meta: dict) -> go.Figure:
    explained_pc1 = meta["explained_variance"]["PC1"]
    explained_pc2 = meta["explained_variance"]["PC2"]

    cluster_order = meta["cluster_order"]

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="NutritionCluster",
        category_orders={"NutritionCluster": cluster_order},
        hover_name="Name",
        hover_data={
            "RecipeCategory": True,
            "Calories": ":.1f",
            "ProteinContent": ":.1f",
            "SugarContent": ":.1f",
            "SodiumContent": ":.1f",
            "AggregatedRating": ":.2f",
            "ReviewCount": True,
            "PC1": ":.3f",
            "PC2": ":.3f",
        },
        opacity=0.72,
        render_mode="webgl",
    )

    fig.add_hline(y=0, line_dash="dash", line_width=1, opacity=0.45)
    fig.add_vline(x=0, line_dash="dash", line_width=1, opacity=0.45)

    fig.update_traces(marker=dict(size=7, line=dict(width=0)))
    fig.update_xaxes(title=f"PC1 ({explained_pc1 * 100:.1f}% variance explained)")
    fig.update_yaxes(title=f"PC2 ({explained_pc2 * 100:.1f}% variance explained)")

    return apply_food_theme(
        fig,
        "The Nutritional Landscape of Recipes",
        "PCA map of nutritional totals and density features, colored by nutrition cluster"
    )


def build_fig_pca_categories(plot_df: pd.DataFrame, meta: dict) -> go.Figure:
    explained_pc1 = meta["explained_variance"]["PC1"]
    explained_pc2 = meta["explained_variance"]["PC2"]
    category_order = meta["category_order"]

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="RecipeCategory",
        category_orders={"RecipeCategory": category_order},
        hover_name="Name",
        hover_data={
            "Calories": ":.1f",
            "ProteinPer100Cal": ":.2f",
            "SugarPer100Cal": ":.2f",
            "SodiumPer100Cal": ":.2f",
            "NutritionCluster": True,
            "AggregatedRating": ":.2f",
            "ReviewCount": True,
        },
        opacity=0.70,
        render_mode="webgl",
    )

    fig.add_hline(y=0, line_dash="dash", line_width=1, opacity=0.45)
    fig.add_vline(x=0, line_dash="dash", line_width=1, opacity=0.45)

    fig.update_traces(marker=dict(size=7, line=dict(width=0)))
    fig.update_xaxes(title=f"PC1 ({explained_pc1 * 100:.1f}% variance explained)")
    fig.update_yaxes(title=f"PC2 ({explained_pc2 * 100:.1f}% variance explained)")

    return apply_food_theme(
        fig,
        "Category Labels vs Nutritional Reality",
        "Where the major recipe categories actually sit in nutrition space"
    )


def build_fig_category_spread(plot_df: pd.DataFrame, meta: dict) -> go.Figure:
    category_order = meta["category_order"]

    fig = px.violin(
        plot_df,
        x="ProteinPer100Cal",
        y="RecipeCategory",
        color="RecipeCategory",
        category_orders={"RecipeCategory": category_order},
        box=True,
        points="suspectedoutliers",
        hover_data={
            "Name": True,
            "ProteinPer100Cal": ":.2f",
            "Calories": ":.1f",
            "ProteinContent": ":.1f",
            "AggregatedRating": ":.2f",
            "ReviewCount": True,
        },
    )

    fig.update_traces(meanline_visible=True, opacity=0.75)
    fig.update_xaxes(title="Protein per 100 Calories")
    fig.update_yaxes(title="Recipe Category")

    return apply_food_theme(
        fig,
        "Category Nutritional Spread",
        "Protein density varies widely even within familiar recipe labels"
    )


def build_fig_tradeoff(plot_df: pd.DataFrame, meta: dict) -> go.Figure:
    category_order = meta["category_order"]

    fig = px.scatter(
        plot_df,
        x="SodiumPer100Cal",
        y="ProteinPer100Cal",
        color="RecipeCategory",
        size="RatingSize",
        category_orders={"RecipeCategory": category_order},
        size_max=22,
        hover_name="Name",
        hover_data={
            "Calories": ":.1f",
            "ProteinContent": ":.1f",
            "SodiumContent": ":.1f",
            "SugarContent": ":.1f",
            "AggregatedRating": ":.2f",
            "ReviewCount": True,
            "NutritionBalanceScore": ":.2f",
            "NutritionCluster": True,
            "RatingSize": False,
        },
        opacity=0.72,
        render_mode="webgl",
    )

    fig.update_xaxes(type="log", title="Sodium per 100 Calories (log scale)")
    fig.update_yaxes(title="Protein per 100 Calories")

    return apply_food_theme(
        fig,
        "Nutritional Trade-offs",
        "Protein efficiency versus sodium burden, with marker size reflecting rating"
    )


def build_fig_cluster_heatmap(plot_df: pd.DataFrame, meta: dict) -> go.Figure:
    x = meta["feature_order"]
    y = meta["cluster_order"]

    zmat = (
        plot_df.pivot(index="NutritionCluster", columns="Feature", values="ZMedian")
        .reindex(index=y, columns=x)
    )

    raw_med = (
        plot_df.pivot(index="NutritionCluster", columns="Feature", values="MedianRaw")
        .reindex(index=y, columns=x)
    )

    hover_text = np.empty(zmat.shape, dtype=object)
    for i, cluster in enumerate(y):
        for j, feat in enumerate(x):
            hover_text[i, j] = (
                f"Cluster: {cluster}<br>"
                f"Feature: {feat}<br>"
                f"Z-scored median: {zmat.iloc[i, j]:.2f}<br>"
                f"Raw median: {raw_med.iloc[i, j]:.2f}"
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=zmat.values,
            x=x,
            y=y,
            text=hover_text,
            hoverinfo="text",
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Z-score")
        )
    )

    fig.update_xaxes(title="Nutritional Feature", tickangle=-35)
    fig.update_yaxes(title="Nutrition Cluster")

    return apply_food_theme(
        fig,
        "Nutrition Cluster Profiles",
        "Standardized median nutrient signatures by cluster"
    )


def build_fig_loadings(plot_df: pd.DataFrame, meta: dict) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=plot_df["Feature"],
            y=plot_df["PC1_loading"],
            name="PC1 loading"
        )
    )
    fig.add_trace(
        go.Bar(
            x=plot_df["Feature"],
            y=plot_df["PC2_loading"],
            name="PC2 loading"
        )
    )

    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Feature", tickangle=-35)
    fig.update_yaxes(title="Loading")

    return apply_food_theme(
        fig,
        "PCA Loadings",
        "Which nutritional features drive the two main nutrition dimensions"
    )


# -----------------------------
# FIGURE EXPORTERS
# -----------------------------
def export_fig_pca_landscape(df_landscape: pd.DataFrame, explained_df: pd.DataFrame, output_dir: str):
    fig_key = "nutrition_pca_landscape"

    plot_df = df_landscape[
        [
            "RecipeId", "Name", "RecipeCategory",
            "AggregatedRating", "ReviewCount",
            "Calories", "ProteinContent", "SugarContent", "SodiumContent",
            "PC1", "PC2", "NutritionCluster"
        ]
    ].copy()

    evr = dict(zip(explained_df["Component"], explained_df["ExplainedVarianceRatio"]))

    meta = {
        "fig_key": fig_key,
        "fig_type": "scatter",
        "title": "The Nutritional Landscape of Recipes",
        "x_col": "PC1",
        "y_col": "PC2",
        "color_col": "NutritionCluster",
        "hover_name_col": "Name",
        "explained_variance": evr,
        "cluster_order": sorted(plot_df["NutritionCluster"].astype(str).unique(), key=lambda x: int(x)),
    }

    fig = build_fig_pca_landscape(plot_df, meta)
    return export_figure_bundle(fig_key, plot_df, meta, fig, output_dir, EXPORT_PARQUET)


def export_fig_pca_categories(
    df_landscape: pd.DataFrame,
    explained_df: pd.DataFrame,
    top_categories: list[str],
    output_dir: str
):
    fig_key = "nutrition_pca_categories"

    plot_df = df_landscape[df_landscape["RecipeCategory"].isin(top_categories)].copy()
    plot_df = plot_df[
        [
            "RecipeId", "Name", "RecipeCategory",
            "AggregatedRating", "ReviewCount",
            "Calories", "ProteinPer100Cal", "SugarPer100Cal", "SodiumPer100Cal",
            "PC1", "PC2", "NutritionCluster"
        ]
    ]

    evr = dict(zip(explained_df["Component"], explained_df["ExplainedVarianceRatio"]))

    meta = {
        "fig_key": fig_key,
        "fig_type": "scatter",
        "title": "Category Labels vs Nutritional Reality",
        "x_col": "PC1",
        "y_col": "PC2",
        "color_col": "RecipeCategory",
        "hover_name_col": "Name",
        "explained_variance": evr,
        "category_order": top_categories,
        "top_categories": top_categories,
    }

    fig = build_fig_pca_categories(plot_df, meta)
    return export_figure_bundle(fig_key, plot_df, meta, fig, output_dir, EXPORT_PARQUET)


def export_fig_category_spread(df_landscape: pd.DataFrame, top_categories: list[str], output_dir: str):
    fig_key = "nutrition_category_spread"

    plot_df = df_landscape[df_landscape["RecipeCategory"].isin(top_categories)].copy()
    plot_df = plot_df[
        [
            "RecipeId", "Name", "RecipeCategory",
            "AggregatedRating", "ReviewCount",
            "Calories", "ProteinContent", "ProteinPer100Cal"
        ]
    ]

    order = (
        plot_df.groupby("RecipeCategory")["ProteinPer100Cal"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    meta = {
        "fig_key": fig_key,
        "fig_type": "violin",
        "title": "Category Nutritional Spread",
        "x_col": "ProteinPer100Cal",
        "y_col": "RecipeCategory",
        "color_col": "RecipeCategory",
        "category_order": order,
        "metric": "ProteinPer100Cal",
        "metric_label": "Protein per 100 Calories",
    }

    fig = build_fig_category_spread(plot_df, meta)
    return export_figure_bundle(fig_key, plot_df, meta, fig, output_dir, EXPORT_PARQUET)


def export_fig_tradeoff(df_landscape: pd.DataFrame, top_categories: list[str], output_dir: str):
    fig_key = "nutrition_tradeoff"

    plot_df = df_landscape[df_landscape["RecipeCategory"].isin(top_categories)].copy()

    rating = plot_df["AggregatedRating"].fillna(plot_df["AggregatedRating"].median())
    rating_min = rating.min()
    rating_max = rating.max()

    if pd.isna(rating_min) or pd.isna(rating_max) or rating_min == rating_max:
        plot_df["RatingSize"] = 10.0
    else:
        plot_df["RatingSize"] = 8 + 14 * (rating - rating_min) / (rating_max - rating_min)

    plot_df = plot_df[
        [
            "RecipeId", "Name", "RecipeCategory",
            "AggregatedRating", "ReviewCount",
            "Calories", "ProteinContent", "SugarContent", "SodiumContent",
            "ProteinPer100Cal", "SodiumPer100Cal",
            "NutritionBalanceScore", "NutritionCluster", "RatingSize"
        ]
    ].copy()

    meta = {
        "fig_key": fig_key,
        "fig_type": "scatter",
        "title": "Nutritional Trade-offs",
        "x_col": "SodiumPer100Cal",
        "y_col": "ProteinPer100Cal",
        "color_col": "RecipeCategory",
        "size_col": "RatingSize",
        "category_order": top_categories,
        "x_scale": "log",
    }

    fig = build_fig_tradeoff(plot_df, meta)
    return export_figure_bundle(fig_key, plot_df, meta, fig, output_dir, EXPORT_PARQUET)


def export_fig_cluster_heatmap(df_landscape: pd.DataFrame, output_dir: str):
    fig_key = "nutrition_cluster_heatmap"

    profile_cols = [
        "Calories",
        "FatContent",
        "SugarContent",
        "ProteinContent",
        "FiberContent",
        "SodiumContent",
        "ProteinPer100Cal",
        "FiberPer100Cal",
        "SugarPer100Cal",
        "SodiumPer100Cal",
    ]
    profile_cols = [c for c in profile_cols if c in df_landscape.columns]

    cluster_profile = df_landscape.groupby("NutritionCluster")[profile_cols].median().copy()
    cluster_profile = cluster_profile.sort_index(key=lambda x: x.astype(int))

    std = cluster_profile.std(ddof=0).replace(0, 1.0)
    cluster_profile_z = (cluster_profile - cluster_profile.mean()) / std

    long_rows = []
    for cluster in cluster_profile.index:
        for feat in profile_cols:
            long_rows.append({
                "NutritionCluster": str(cluster),
                "Feature": feat,
                "MedianRaw": float(cluster_profile.loc[cluster, feat]),
                "ZMedian": float(cluster_profile_z.loc[cluster, feat]),
            })

    plot_df = pd.DataFrame(long_rows)

    meta = {
        "fig_key": fig_key,
        "fig_type": "heatmap",
        "title": "Nutrition Cluster Profiles",
        "feature_order": profile_cols,
        "cluster_order": [str(x) for x in cluster_profile.index.tolist()],
        "value_col": "ZMedian",
        "raw_value_col": "MedianRaw",
    }

    fig = build_fig_cluster_heatmap(plot_df, meta)
    return export_figure_bundle(fig_key, plot_df, meta, fig, output_dir, EXPORT_PARQUET)


def export_fig_pca_loadings(loadings_df: pd.DataFrame, output_dir: str):
    fig_key = "nutrition_pca_loadings"

    plot_df = loadings_df.copy()
    plot_df["abs_loading_strength"] = (
        plot_df["PC1_loading"].abs() + plot_df["PC2_loading"].abs()
    )
    plot_df = plot_df.sort_values("abs_loading_strength", ascending=False).reset_index(drop=True)

    meta = {
        "fig_key": fig_key,
        "fig_type": "bar",
        "title": "PCA Loadings",
        "x_col": "Feature",
        "y_cols": ["PC1_loading", "PC2_loading"],
    }

    fig = build_fig_loadings(plot_df, meta)
    return export_figure_bundle(fig_key, plot_df, meta, fig, output_dir, EXPORT_PARQUET)


# -----------------------------
# RE-PLOTTING API
# -----------------------------
def replot_exported_nutrition_figure(fig_key: str, output_dir: str) -> go.Figure:
    fdir = figure_dir(output_dir, fig_key)

    metadata = safe_read_json(os.path.join(fdir, "metadata.json"))

    csv_path = os.path.join(fdir, "plot_data.csv")
    parquet_path = os.path.join(fdir, "plot_data.parquet")
    if os.path.exists(parquet_path):
        plot_df = load_exported_dataframe(parquet_path)
    elif os.path.exists(csv_path):
        plot_df = load_exported_dataframe(csv_path)
    else:
        raise FileNotFoundError(f"No exported plot data found for figure '{fig_key}'.")

    if fig_key == "nutrition_pca_landscape":
        return build_fig_pca_landscape(plot_df, metadata)
    if fig_key == "nutrition_pca_categories":
        return build_fig_pca_categories(plot_df, metadata)
    if fig_key == "nutrition_category_spread":
        return build_fig_category_spread(plot_df, metadata)
    if fig_key == "nutrition_tradeoff":
        return build_fig_tradeoff(plot_df, metadata)
    if fig_key == "nutrition_cluster_heatmap":
        return build_fig_cluster_heatmap(plot_df, metadata)
    if fig_key == "nutrition_pca_loadings":
        return build_fig_loadings(plot_df, metadata)

    raise ValueError(f"Unknown figure key: {fig_key}")


# -----------------------------
# REPORTING
# -----------------------------
def print_pca_report(loadings_df: pd.DataFrame, explained_df: pd.DataFrame):
    explained = explained_df.set_index("Component")["ExplainedVarianceRatio"]

    print("\n" + "=" * 70)
    print("PCA EXPLAINED VARIANCE")
    print("=" * 70)
    print(explained.round(4))

    for comp in ["PC1_loading", "PC2_loading"]:
        comp_label = comp.replace("_loading", "")
        print("\n" + "=" * 70)
        print(f"TOP POSITIVE / NEGATIVE LOADINGS FOR {comp_label}")
        print("=" * 70)
        print(loadings_df[["Feature", comp]].sort_values(comp, ascending=False).head(6).round(4))
        print(loadings_df[["Feature", comp]].sort_values(comp, ascending=True).head(6).round(4))


def print_category_summary(summary: pd.DataFrame):
    print("\n" + "=" * 70)
    print("CATEGORY SPREAD SUMMARY")
    print("=" * 70)
    print(summary.round(2).to_string(index=False))


def print_best_tradeoff_examples(df: pd.DataFrame, n: int = 10):
    cols_to_show = [
        "Name",
        "RecipeCategory",
        "Calories",
        "ProteinContent",
        "FiberContent",
        "SugarContent",
        "SodiumContent",
        "ProteinPer100Cal",
        "FiberPer100Cal",
        "SugarPer100Cal",
        "SodiumPer100Cal",
        "NutritionBalanceScore",
        "AggregatedRating",
        "ReviewCount",
    ]
    cols_to_show = [c for c in cols_to_show if c in df.columns]

    best = (
        df.sort_values(
            ["NutritionBalanceScore", "AggregatedRating", "ReviewCount"],
            ascending=[False, False, False]
        )
        .head(n)[cols_to_show]
        .copy()
    )

    worst = (
        df.sort_values(
            ["NutritionBalanceScore", "AggregatedRating", "ReviewCount"],
            ascending=[True, False, False]
        )
        .head(n)[cols_to_show]
        .copy()
    )

    print("\n" + "=" * 70)
    print("EXAMPLE RECIPES WITH STRONGER NUTRITIONAL TRADE-OFFS")
    print("=" * 70)
    print(best.round(2).to_string(index=False))

    print("\n" + "=" * 70)
    print("EXAMPLE RECIPES WITH WEAKER NUTRITIONAL TRADE-OFFS")
    print("=" * 70)
    print(worst.round(2).to_string(index=False))


# -----------------------------
# MANIFEST
# -----------------------------
def write_manifest(output_dir: str, entries: list[dict], global_meta: dict):
    manifest = {
        "visualization": "nutritional_landscape",
        "global_meta": global_meta,
        "figures": entries,
    }
    safe_write_json(manifest, os.path.join(output_dir, "nutrition_figure_manifest.json"))


# -----------------------------
# MAIN
# -----------------------------
def main():
    ensure_dir(OUTPUT_DIR)

    # 1. Load
    df = load_recipes_from_sqlite(RECIPE_FILE)

    # 2. Clean
    df = basic_recipe_cleaning(df)

    # 3. Features
    df = add_derived_features(df)
    feature_cols = select_features_for_landscape(df)

    # 4. PCA + clusters
    df_landscape, loadings_df, explained_df, scaler, pca, kmeans = compute_pca_and_clusters(
        df=df,
        feature_cols=feature_cols,
        n_clusters=N_CLUSTERS
    )

    # 5. Top categories and summaries
    top_categories = get_top_categories(df_landscape, n=TOP_N_CATEGORIES)
    category_summary = summarize_category_spread(df_landscape, top_categories)

    # 6. Reporting
    print_pca_report(loadings_df, explained_df)
    print_category_summary(category_summary)
    print_best_tradeoff_examples(df_landscape, n=10)

    # 7. Save global processed tables
    export_dataframe(df_landscape, os.path.join(OUTPUT_DIR, "recipes_with_nutritional_landscape"), EXPORT_PARQUET)
    export_dataframe(loadings_df, os.path.join(OUTPUT_DIR, "pca_loadings"), EXPORT_PARQUET)
    export_dataframe(category_summary, os.path.join(OUTPUT_DIR, "category_spread_summary"), EXPORT_PARQUET)

    scaler_meta = {
        "feature_cols": feature_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "pca_components": pca.components_.tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "kmeans_cluster_centers": kmeans.cluster_centers_.tolist(),
        "random_state": RANDOM_STATE,
        "n_clusters": N_CLUSTERS,
    }
    safe_write_json(scaler_meta, os.path.join(OUTPUT_DIR, "model_metadata.json"))

    # 8. Export figures individually
    manifest_entries = []
    manifest_entries.append(export_fig_pca_landscape(df_landscape, explained_df, OUTPUT_DIR))
    manifest_entries.append(export_fig_pca_categories(df_landscape, explained_df, top_categories, OUTPUT_DIR))
    manifest_entries.append(export_fig_category_spread(df_landscape, top_categories, OUTPUT_DIR))
    manifest_entries.append(export_fig_tradeoff(df_landscape, top_categories, OUTPUT_DIR))
    manifest_entries.append(export_fig_cluster_heatmap(df_landscape, OUTPUT_DIR))
    manifest_entries.append(export_fig_pca_loadings(loadings_df, OUTPUT_DIR))

    # 9. Write global manifest
    global_meta = {
        "recipe_file": str(RECIPE_FILE),
        "output_dir": str(OUTPUT_DIR),
        "top_n_categories": TOP_N_CATEGORIES,
        "top_categories": top_categories,
        "n_clusters": N_CLUSTERS,
        "min_reviews": MIN_REVIEWS,
        "max_missing_allowed": MAX_MISSING_ALLOWED,
        "feature_cols": feature_cols,
        "n_recipes_after_cleaning": int(len(df_landscape)),
    }
    write_manifest(OUTPUT_DIR, manifest_entries, global_meta)

    print("\nDone.")
    print(f"All outputs saved in: {OUTPUT_DIR}")
    print("Use `replot_exported_nutrition_figure(fig_key, OUTPUT_DIR)` to rebuild any figure later.")


if __name__ == "__main__":
    main()