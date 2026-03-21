#!/usr/bin/env python3
"""
File to clean-up raw data from the Kaggle downloaded dataset.
The pre-processed tables, "recipes" and "reviews", are then stored to /data/tables/ as an SQLite database file named food_recipe.db/
"""

from __future__ import annotations

import ast
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)


RAW_RECIPES_PATH = Path("./data/raw/recipes.csv")
RAW_REVIEWS_PATH = Path("./data/raw/reviews.csv")
OUTPUT_DB_PATH = Path("./data/tables/food_recipe.db")

R_VECTOR_COLUMNS = [
    "Keywords",
    "Images",
    "RecipeInstructions",
    "RecipeIngredientQuantities",
    "RecipeIngredientParts",
]

NUTRITION_COLUMNS = [
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

RECIPE_FLAG_COLUMNS = [
    "TotalTime",
    "RecipeCategory",
    "DatePublished",
    "RecipeIngredientParts",
    "AggregatedRating",
    "ReviewCount",
    "RecipeServings",
    "RecipeYield",
    "RecipeInstructions",
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


def r_vector_to_list(value):
    """Convert an R-style vector string like c("a", "b") into a Python list.

    Also replaces bare NA tokens with None so ast.literal_eval can parse them.
    Non-matching values are returned unchanged.
    """
    if isinstance(value, str) and value.startswith("c("):
        value = value.replace("c(", "[").replace(")", "]")
        value = re.sub(r"\bNA\b", "None", value)
        return ast.literal_eval(value)
    return value


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw recipe and review CSV files."""
    recipes = pd.read_csv(RAW_RECIPES_PATH)
    reviews = pd.read_csv(RAW_REVIEWS_PATH)
    return recipes, reviews


def convert_r_vector_columns(df_recipes: pd.DataFrame) -> pd.DataFrame:
    """Convert R-vector-like columns to JSON strings for SQLite storage."""
    df_recipes = df_recipes.copy()

    for col in R_VECTOR_COLUMNS:
        df_recipes[col] = df_recipes[col].map(r_vector_to_list)
        df_recipes[col] = df_recipes[col].apply(
            lambda x: json.dumps(x) if isinstance(x, list) else x
        )

    return df_recipes


def standardize_invalid_values(df_recipes: pd.DataFrame) -> pd.DataFrame:
    """Replace non-physical nutrition and serving values with NaN."""
    df_recipes = df_recipes.copy()
    df_recipes[NUTRITION_COLUMNS] = df_recipes[NUTRITION_COLUMNS].mask(
        df_recipes[NUTRITION_COLUMNS] <= 0
    )
    df_recipes["RecipeServings"] = df_recipes["RecipeServings"].mask(
        df_recipes["RecipeServings"] <= 0
    )
    return df_recipes


def add_per_serving_nutrition(df_recipes: pd.DataFrame) -> pd.DataFrame:
    """Insert per-serving nutrition columns next to their original columns."""
    df_recipes = df_recipes.copy()

    for col in NUTRITION_COLUMNS:
        new_col = f"{col}PerServing"
        values = df_recipes[col] / df_recipes["RecipeServings"]
        pos = df_recipes.columns.get_loc(col) + 1
        df_recipes.insert(pos, new_col, values)

    return df_recipes


def parse_time_and_date_columns(
    df_recipes: pd.DataFrame,
    df_reviews: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert time/date columns to useful formats and add consistency checks."""
    df_recipes = df_recipes.copy()
    df_reviews = df_reviews.copy()

    time_cols = ["CookTime", "PrepTime", "TotalTime"]
    for col in time_cols:
        df_recipes[col] = pd.to_timedelta(df_recipes[col], errors="coerce")
        df_recipes[col] = df_recipes[col].dt.total_seconds()

    df_recipes["DatePublished"] = pd.to_datetime(
        df_recipes["DatePublished"], errors="coerce"
    )

    review_date_cols = ["DateSubmitted", "DateModified"]
    for col in review_date_cols:
        df_reviews[col] = pd.to_datetime(df_reviews[col], errors="coerce")

    diff = (df_recipes["CookTime"] + df_recipes["PrepTime"]) - df_recipes["TotalTime"]
    consistent_time = np.where(
        diff <= 60.0,
        "consistent",
        np.where((diff > 60.0) & (diff < 900.0), "partially missing", "inconsistent"),
    )
    df_recipes.insert(7, "ConsistentTime", consistent_time)

    # Store datetimes as strings so SQLite receives simple serializable values.
    df_recipes["DatePublished"] = df_recipes["DatePublished"].astype(str)
    df_reviews[review_date_cols] = df_reviews[review_date_cols].astype(str)

    return df_recipes, df_reviews


def add_total_time_bucket(df_recipes: pd.DataFrame) -> pd.DataFrame:
    """Add a categorical total-time bucket column."""
    df_recipes = df_recipes.copy()

    bins = [
        0.0,
        15 * 60.0,
        30 * 60.0,
        60 * 60.0,
        120 * 60.0,
        240 * 60.0,
        999999999 * 60.0,
    ]
    labels = [
        "<15 mins",
        "15-30 mins",
        "30-60 mins",
        "1-2 hrs",
        "2-4 hrs",
        ">4 hrs",
    ]

    df_recipes.insert(
        8,
        "TotalTimeBucket",
        pd.cut(df_recipes["TotalTime"], bins=bins, labels=labels),
    )

    return df_recipes


def add_missing_value_flags(df_recipes: pd.DataFrame) -> pd.DataFrame:
    """Append boolean availability flags for selected columns."""
    df_recipes = df_recipes.copy()
    flags = df_recipes[RECIPE_FLAG_COLUMNS].notna().add_prefix("Has")
    return pd.concat([df_recipes, flags], axis=1)


def export_to_sqlite(
    df_recipes: pd.DataFrame,
    df_reviews: pd.DataFrame,
    db_path: Path = OUTPUT_DB_PATH,
) -> None:
    """Write processed recipe and review tables to a SQLite database."""
    with sqlite3.connect(db_path) as conn:
        df_recipes.to_sql("recipes", conn, if_exists="replace", index=False)
        df_reviews.to_sql("reviews", conn, if_exists="replace", index=False)


def main() -> None:
    """Run the full preprocessing pipeline."""
    df_recipes, df_reviews = load_raw_data()

    print("Total number of rows:", len(df_recipes))
    print("Total number of unique recipe IDs:", df_recipes["RecipeId"].nunique())

    df_recipes = convert_r_vector_columns(df_recipes)
    df_recipes = standardize_invalid_values(df_recipes)
    df_recipes = add_per_serving_nutrition(df_recipes)
    df_recipes, df_reviews = parse_time_and_date_columns(df_recipes, df_reviews)
    df_recipes = add_total_time_bucket(df_recipes)
    df_recipes = add_missing_value_flags(df_recipes)

    export_to_sqlite(df_recipes, df_reviews)
    print(f"Saved processed database to {OUTPUT_DB_PATH}")


if __name__ == "__main__":
    main()
