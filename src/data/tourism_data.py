"""Tourism data loading and cleaning for Part 2.

Covers brief task 1: import the datasets, check missing values and duplicates,
remove anomalies.

The loaders are deliberately defensive about column names. The problem
statement calls the places file ``tourism_with_id.csv`` while the supplied file
is an ``.xlsx``, and the public version of this dataset ships with two trailing
unnamed columns. Rather than assume, we normalise and validate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Canonical column names -> accepted aliases (lower-cased, stripped).
PLACE_ALIASES = {
    "place_id": {"place_id", "placeid", "id"},
    "place_name": {"place_name", "placename", "name"},
    "description": {"description", "desc"},
    "category": {"category", "categories"},
    "city": {"city", "location"},
    "price": {"price"},
    "rating": {"rating", "avg_rating"},
    "time_minutes": {"time_minutes", "time_minute", "time_min", "timeminutes"},
    "coordinate": {"coordinate", "coordinates"},
    "lat": {"lat", "latitude"},
    "long": {"long", "lon", "longitude"},
}
RATING_ALIASES = {
    "user_id": {"user_id", "userid"},
    "place_id": {"place_id", "placeid"},
    "place_ratings": {"place_ratings", "place_rating", "rating", "ratings"},
}
USER_ALIASES = {
    "user_id": {"user_id", "userid"},
    "location": {"location", "city"},
    "age": {"age"},
}


def _normalise_columns(df: pd.DataFrame, aliases: dict[str, set[str]],
                       label: str) -> pd.DataFrame:
    """Rename columns to canonical names; drop junk; fail loudly if key ones are missing."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Drop pandas' unnamed placeholder columns and any all-null column.
    junk = [c for c in df.columns if c.lower().startswith("unnamed") or df[c].isna().all()]
    if junk:
        df = df.drop(columns=junk)
        print(f"[{label}] dropped empty/unnamed columns: {junk}")

    rename = {}
    for col in df.columns:
        key = col.lower().replace(" ", "_")
        for canonical, accepted in aliases.items():
            if key in accepted:
                rename[col] = canonical
                break
    df = df.rename(columns=rename)

    missing = set(aliases) - set(df.columns)
    critical = missing & {"place_id", "user_id", "place_name", "place_ratings"}
    if critical:
        raise ValueError(
            f"[{label}] required column(s) {sorted(critical)} not found. "
            f"Got: {list(df.columns)}. Update the alias table in tourism_data.py."
        )
    if missing:
        print(f"[{label}] note - optional columns absent: {sorted(missing)}")
    return df


def load_raw(cfg) -> dict[str, pd.DataFrame]:
    """Read the three source files, whatever format they arrived in."""
    root = Path(cfg.tourism_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Set HERITAGE_DATA_ROOT or copy the Part 2 files there."
        )

    places_path = root / cfg.places_file
    if not places_path.exists():  # tolerate .csv/.xlsx swap
        alternatives = list(root.glob("tourism_with_id.*"))
        if not alternatives:
            raise FileNotFoundError(f"No tourism_with_id.* file inside {root}")
        places_path = alternatives[0]
        print(f"Using {places_path.name} for the places table.")

    places = (
        pd.read_excel(places_path)
        if places_path.suffix.lower() in {".xlsx", ".xls"}
        else pd.read_csv(places_path)
    )
    ratings = pd.read_csv(root / cfg.ratings_file)
    users = pd.read_csv(root / cfg.users_file)

    return {
        "places": _normalise_columns(places, PLACE_ALIASES, "places"),
        "ratings": _normalise_columns(ratings, RATING_ALIASES, "ratings"),
        "users": _normalise_columns(users, USER_ALIASES, "users"),
    }


def inspect(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Brief task 1.I - per-column dtype, nulls, uniques, sample value."""
    report = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_missing": df.isna().sum(),
            "pct_missing": (100 * df.isna().mean()).round(2),
            "n_unique": df.nunique(),
            "example": [df[c].dropna().iloc[0] if df[c].notna().any() else None
                        for c in df.columns],
        }
    )
    print(f"\n=== {name} === shape={df.shape}  duplicated_rows={df.duplicated().sum()}")
    return report


def clean(data: dict[str, pd.DataFrame], cfg, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Brief task 1.II - remove duplicates and anomalies.

    Anomalies handled, in order of how much they would distort the model:
      1. duplicate (user_id, place_id) ratings   -> keep the last
      2. ratings outside the valid 1-5 range     -> drop
      3. impossible ages (<= 0 or > 100)         -> drop
      4. orphan ratings referencing unknown ids  -> drop
      5. Time_Minutes is ~50% null in this source -> left as NaN, never imputed
         with the mean (that would invent tour durations that do not exist)
    """
    places = data["places"].copy()
    ratings = data["ratings"].copy()
    users = data["users"].copy()
    log: list[str] = []

    before = len(ratings)
    ratings = ratings.drop_duplicates(subset=["user_id", "place_id"], keep="last")
    log.append(f"ratings: removed {before - len(ratings)} duplicate (user, place) pairs")

    before = len(ratings)
    ratings = ratings[ratings["place_ratings"].between(1, 5)]
    log.append(f"ratings: removed {before - len(ratings)} out-of-range ratings")

    if "age" in users.columns:
        before = len(users)
        users = users[users["age"].between(1, 100)]
        log.append(f"users: removed {before - len(users)} rows with impossible ages")

    before = len(places)
    places = places.drop_duplicates(subset=["place_id"], keep="first")
    log.append(f"places: removed {before - len(places)} duplicate place_ids")

    known_places = set(places["place_id"])
    known_users = set(users["user_id"])
    before = len(ratings)
    ratings = ratings[
        ratings["place_id"].isin(known_places) & ratings["user_id"].isin(known_users)
    ]
    log.append(f"ratings: removed {before - len(ratings)} orphan rows")

    # Split "City, Province" style location strings into a usable province field.
    if "location" in users.columns:
        users["province"] = (
            users["location"].astype(str).str.split(",").str[-1].str.strip()
        )
        users["home_city"] = (
            users["location"].astype(str).str.split(",").str[0].str.strip()
        )

    if verbose:
        print("Cleaning summary")
        for line in log:
            print("  -", line)
        print(
            f"\nFinal shapes  places={places.shape}  ratings={ratings.shape}  "
            f"users={users.shape}"
        )
        density = len(ratings) / (users["user_id"].nunique() * places["place_id"].nunique())
        print(f"Rating matrix density: {density:.2%} (sparsity {1 - density:.2%})")

    return {
        "places": places.reset_index(drop=True),
        "ratings": ratings.reset_index(drop=True),
        "users": users.reset_index(drop=True),
        "_log": log,
    }


def build_merged(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Brief task 4 - one table of places joined to their user ratings."""
    merged = (
        data["ratings"]
        .merge(data["places"], on="place_id", how="left", suffixes=("", "_place"))
        .merge(data["users"], on="user_id", how="left", suffixes=("", "_user"))
    )
    return merged


def place_popularity(merged: pd.DataFrame, min_ratings: int = 5) -> pd.DataFrame:
    """Most-loved spots. Brief task 4.I.

    A place with a single 5-star rating is not "the best place in Indonesia".
    We therefore report the count alongside the mean and apply a Bayesian
    (shrunken) score that pulls low-volume places toward the global average.
    """
    grouped = (
        merged.groupby(["place_id", "place_name", "city", "category"], dropna=False)
        .agg(n_ratings=("place_ratings", "size"), mean_rating=("place_ratings", "mean"))
        .reset_index()
    )
    prior_mean = merged["place_ratings"].mean()
    m = min_ratings
    grouped["bayesian_score"] = (
        (grouped["n_ratings"] * grouped["mean_rating"] + m * prior_mean)
        / (grouped["n_ratings"] + m)
    )
    grouped["mean_rating"] = grouped["mean_rating"].round(3)
    grouped["bayesian_score"] = grouped["bayesian_score"].round(3)
    return grouped.sort_values("bayesian_score", ascending=False).reset_index(drop=True)


def train_test_split_ratings(ratings: pd.DataFrame, cfg):
    """Random hold-out split on the rating events.

    Note for the report: a fully rigorous evaluation would split per user by
    timestamp, but this dataset carries no timestamps, so a stratified random
    split on user_id is the closest honest approximation.
    """
    from sklearn.model_selection import train_test_split

    # Users with a single rating cannot be stratified; keep them in train.
    counts = ratings["user_id"].value_counts()
    multi = ratings[ratings["user_id"].isin(counts[counts >= 2].index)]
    singles = ratings[~ratings["user_id"].isin(counts[counts >= 2].index)]

    train, test = train_test_split(
        multi,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=multi["user_id"],
    )
    train = pd.concat([train, singles], ignore_index=True)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_rating_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """users x places pivot table (NaN where unrated)."""
    return ratings.pivot_table(
        index="user_id", columns="place_id", values="place_ratings", aggfunc="mean"
    )
