"""Descriptive statistics, outlier detection and normalisation for Part 2.

The brief asks for preliminary inspection and removal of anomalies. This module
covers the wider standard: distribution statistics, outlier detection by three
methods, a test of *why* data is missing, and normalisation guidance.

The distinction that runs through this module: **an outlier is not the same as
an error.** Kepulauan Seribu is 70 km off the Jakarta coast and looks like a
geographic outlier, but it genuinely belongs to Jakarta. Dropping it would be
data destruction. Every function here flags candidates for a human decision
rather than silently deleting rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Descriptive statistics
# --------------------------------------------------------------------------
def numeric_summary(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Full descriptive statistics, including the shape measures `describe()` omits.

    Skewness and kurtosis are the point of this table: they tell you whether a
    mean is meaningful and whether a transform is needed before the feature is
    used in anything distance-based.
    """
    from scipy import stats

    columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for col in columns:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        rows.append(
            {
                "column": col,
                "n": int(s.size),
                "n_missing": int(df[col].isna().sum()),
                "pct_missing": round(100 * df[col].isna().mean(), 2),
                "n_zero": int((s == 0).sum()),
                "mean": round(float(s.mean()), 3),
                "std": round(float(s.std()), 3),
                "min": round(float(s.min()), 3),
                "q1": round(float(q1), 3),
                "median": round(float(s.median()), 3),
                "q3": round(float(q3), 3),
                "max": round(float(s.max()), 3),
                "iqr": round(float(q3 - q1), 3),
                # Coefficient of variation - comparable across differently-scaled columns.
                "cv": round(float(s.std() / s.mean()), 3) if s.mean() else np.nan,
                "skew": round(float(stats.skew(s)), 3),
                "kurtosis": round(float(stats.kurtosis(s)), 3),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def categorical_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Level counts, mode dominance and normalised entropy.

    Normalised entropy near 1 means levels are evenly used; near 0 means one
    level dominates and the column carries little information.
    """
    from scipy import stats

    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        probs = counts / counts.sum()
        max_entropy = np.log(len(counts)) if len(counts) > 1 else 1.0
        rows.append(
            {
                "column": col,
                "n_levels": int(len(counts)),
                "n_missing": int(df[col].isna().sum()),
                "mode": counts.index[0],
                "mode_count": int(counts.iloc[0]),
                "mode_share_%": round(100 * float(probs.iloc[0]), 1),
                "normalised_entropy": round(float(stats.entropy(probs) / max_entropy), 3),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def correlation_matrix(df: pd.DataFrame, columns: list[str] | None = None,
                       method: str = "spearman") -> pd.DataFrame:
    """Correlations between numeric features.

    Spearman by default: several columns here are heavily skewed, and Pearson on
    a skewed variable mostly measures the influence of its extreme values.
    """
    columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    return df[columns].corr(method=method).round(3)


# --------------------------------------------------------------------------
# Outlier detection
# --------------------------------------------------------------------------
def detect_outliers(series: pd.Series, method: str = "iqr",
                    threshold: float | None = None) -> pd.Series:
    """Boolean mask of outliers.

    Methods:
      * ``iqr``    - outside Q1 - k*IQR .. Q3 + k*IQR (k=1.5). Distribution-free.
      * ``zscore`` - |z| > 3. Assumes roughly normal; breaks on skewed data
                     because the extreme values inflate the standard deviation
                     they are being measured against.
      * ``mad``    - modified z-score using the median absolute deviation.
                     Robust: the statistics it uses are not themselves distorted
                     by the outliers. Preferred for skewed columns.
    """
    s = series.dropna()
    if s.empty:
        return pd.Series(False, index=series.index)

    if method == "iqr":
        k = threshold if threshold is not None else 1.5
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (series < q1 - k * iqr) | (series > q3 + k * iqr)

    elif method == "zscore":
        k = threshold if threshold is not None else 3.0
        std = s.std()
        if std == 0:
            return pd.Series(False, index=series.index)
        mask = ((series - s.mean()) / std).abs() > k

    elif method == "mad":
        k = threshold if threshold is not None else 3.5
        median = s.median()
        mad = (s - median).abs().median()
        if mad == 0:
            # More than half the values are identical, so the robust z-score is
            # undefined (0/0). Returning "no outliers" here would be a silent
            # blind spot: if 95% of images are exactly 128x128, the one 900x40
            # image is the most obvious outlier in the dataset. When the spread
            # is degenerate, any departure from the median is the anomaly.
            mask = series.notna() & (series != median)
        else:
            mask = (0.6745 * (series - median) / mad).abs() > k

    else:
        raise ValueError(f"Unknown method '{method}'. Use iqr, zscore or mad.")

    return mask.fillna(False)


def outlier_report(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Compare all three detectors per column.

    Where the methods disagree sharply, the column is skewed and the robust
    (MAD) verdict is the one to trust.
    """
    columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for col in columns:
        s = df[col]
        entry = {"column": col, "n_valid": int(s.notna().sum())}
        for method in ("iqr", "zscore", "mad"):
            mask = detect_outliers(s, method=method)
            entry[f"n_{method}"] = int(mask.sum())
            entry[f"pct_{method}"] = round(100 * mask.sum() / max(s.notna().sum(), 1), 2)
        entry["max"] = round(float(s.max()), 2) if s.notna().any() else np.nan
        rows.append(entry)
    return pd.DataFrame(rows).set_index("column")


def show_outliers(df: pd.DataFrame, column: str, method: str = "mad",
                  display_cols: list[str] | None = None, n: int = 10) -> pd.DataFrame:
    """The actual flagged rows, so you can judge whether they are errors."""
    mask = detect_outliers(df[column], method=method)
    cols = display_cols or [c for c in df.columns if c != "description"]
    return df.loc[mask, cols].sort_values(column, ascending=False).head(n)


def geographic_outliers(places: pd.DataFrame, group_col: str = "city",
                        threshold: float = 5.0) -> pd.DataFrame:
    """Places far from their own city's centroid, by robust (MAD) distance.

    A city's attractions should cluster geographically. A place many MADs away
    is either a coordinate error or a genuinely remote site administratively
    attached to that city - and the two need different responses.
    """
    rows = []
    for city, group in places.groupby(group_col):
        for axis in ("lat", "long"):
            if axis not in group.columns:
                continue
            median = group[axis].median()
            mad = (group[axis] - median).abs().median()
            if mad == 0:
                continue
            z = 0.6745 * (group[axis] - median) / mad
            for idx in group.index[z.abs() > threshold]:
                rows.append(
                    {
                        "place_id": places.at[idx, "place_id"],
                        "place_name": places.at[idx, "place_name"],
                        group_col: city,
                        "axis": axis,
                        "value": round(float(places.at[idx, axis]), 4),
                        f"{group_col}_median": round(float(median), 4),
                        "robust_z": round(float(z[idx]), 1),
                        "km_from_centroid": round(
                            abs(float(places.at[idx, axis]) - float(median))
                            * (111.0 if axis == "lat" else 111.0 * np.cos(np.radians(median))),
                            1,
                        ),
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=["place_id", "place_name", group_col, "axis", "value", "robust_z"]
        )
    return (
        pd.DataFrame(rows)
        .sort_values("robust_z", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )


def coordinate_consistency(places: pd.DataFrame) -> dict:
    """Do the `coordinate` string and the `lat`/`long` columns agree?

    A cheap, decisive integrity check: two representations of the same fact
    should match, and if they do not, one of them has been corrupted.
    """
    import ast

    if "coordinate" not in places.columns:
        return {"available": False}

    parsed_lat, parsed_lng, failures = [], [], 0
    for value in places["coordinate"]:
        try:
            d = ast.literal_eval(str(value))
            parsed_lat.append(d.get("lat"))
            parsed_lng.append(d.get("lng"))
        except (ValueError, SyntaxError):
            parsed_lat.append(np.nan)
            parsed_lng.append(np.nan)
            failures += 1

    lat_ok = np.allclose(np.array(parsed_lat, dtype=float), places["lat"].to_numpy(),
                         atol=1e-6, equal_nan=True)
    lng_ok = np.allclose(np.array(parsed_lng, dtype=float), places["long"].to_numpy(),
                         atol=1e-6, equal_nan=True)
    return {
        "available": True,
        "parse_failures": failures,
        "lat_matches": bool(lat_ok),
        "long_matches": bool(lng_ok),
        "consistent": bool(lat_ok and lng_ok and failures == 0),
    }


def country_bounds_check(places: pd.DataFrame,
                         lat_range: tuple[float, float] = (-11.0, 6.0),
                         long_range: tuple[float, float] = (95.0, 141.0)) -> pd.DataFrame:
    """Coordinates outside Indonesia's bounding box - unambiguous errors."""
    mask = (
        ~places["lat"].between(*lat_range) | ~places["long"].between(*long_range)
    )
    return places.loc[mask, ["place_id", "place_name", "city", "lat", "long"]]


# --------------------------------------------------------------------------
# Missing data
# --------------------------------------------------------------------------
def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missingness, worst first."""
    report = pd.DataFrame(
        {
            "n_missing": df.isna().sum(),
            "pct_missing": (100 * df.isna().mean()).round(2),
            "dtype": df.dtypes.astype(str),
        }
    )
    return report[report["n_missing"] > 0].sort_values("pct_missing", ascending=False)


def missingness_mechanism(df: pd.DataFrame, target: str,
                          group_cols: list[str]) -> pd.DataFrame:
    """Is `target` missing at random, or does it depend on other columns?

    This determines what you are allowed to do about it:

    * **MCAR** (missing completely at random) - dropping or mean-imputing is
      defensible.
    * **MAR** (depends on observed variables) - mean imputation introduces
      systematic bias, because the missing values are not a random sample of the
      column. Either model the missingness or leave the values as NaN.

    A significant chi-square here is evidence against MCAR.
    """
    from scipy import stats

    if target not in df.columns:
        raise KeyError(f"'{target}' not in dataframe")

    flag = df[target].isna()
    rows = []
    for col in group_cols:
        if col not in df.columns:
            continue
        table = pd.crosstab(df[col], flag)
        if table.shape[1] < 2 or table.shape[0] < 2:
            continue
        chi2, p, dof, _ = stats.chi2_contingency(table)
        rates = (df.groupby(col)[target].apply(lambda s: 100 * s.isna().mean())).round(1)
        rows.append(
            {
                "grouped_by": col,
                "chi2": round(float(chi2), 2),
                "dof": int(dof),
                "p_value": round(float(p), 5),
                "missing_rate_range": f"{rates.min():.0f}%-{rates.max():.0f}%",
                "verdict": "MAR - depends on this column" if p < 0.05
                           else "consistent with MCAR",
            }
        )
    return pd.DataFrame(rows)


def missing_rate_by_group(df: pd.DataFrame, target: str, group_col: str) -> pd.Series:
    """Missing rate of `target` within each level of `group_col`."""
    return (
        df.groupby(group_col)[target]
        .apply(lambda s: round(100 * s.isna().mean(), 1))
        .sort_values(ascending=False)
    )


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
def suggest_transform(series: pd.Series) -> dict:
    """Recommend a transform based on the distribution's actual shape."""
    from scipy import stats

    s = series.dropna()
    if s.empty:
        return {"column": series.name, "recommendation": "no data"}

    skew = float(stats.skew(s))
    has_zeros = bool((s == 0).any())
    has_negatives = bool((s < 0).any())

    if abs(skew) < 0.5:
        rec, why = "standardise (z-score)", "roughly symmetric"
    elif skew > 2 and not has_negatives:
        rec = "log1p, then standardise" if has_zeros else "log, then standardise"
        why = f"strong right skew ({skew:.2f}); a log pulls the tail in"
    elif skew > 0.5 and not has_negatives:
        rec, why = "sqrt or robust scaling", f"moderate right skew ({skew:.2f})"
    else:
        rec, why = "robust scaling (median/IQR)", f"skew {skew:.2f}; use robust statistics"

    return {
        "column": series.name,
        "skew": round(skew, 3),
        "has_zeros": has_zeros,
        "recommendation": rec,
        "reason": why,
    }


def transform_report(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    return pd.DataFrame([suggest_transform(df[c]) for c in columns]).set_index("column")


def log1p_transform(series: pd.Series) -> pd.Series:
    """log(1+x) - defined at zero, which matters here because many places are free."""
    if (series.dropna() < 0).any():
        raise ValueError(f"'{series.name}' has negative values; log1p is undefined.")
    return np.log1p(series)


def robust_scale(series: pd.Series) -> pd.Series:
    """(x - median) / IQR. Unlike z-scoring, outliers do not distort the scale."""
    s = series.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return series - s.median()
    return (series - s.median()) / iqr


def minmax_scale(series: pd.Series) -> pd.Series:
    """Scale to [0, 1]. Used for the ratings fed to the sigmoid-output Keras model."""
    s = series.dropna()
    span = s.max() - s.min()
    if span == 0:
        return series * 0.0
    return (series - s.min()) / span


def compare_scalings(series: pd.Series) -> pd.DataFrame:
    """Side-by-side effect of each transform on the distribution's shape."""
    from scipy import stats

    variants = {"raw": series}
    if not (series.dropna() < 0).any():
        variants["log1p"] = log1p_transform(series)
    variants["robust"] = robust_scale(series)
    variants["minmax"] = minmax_scale(series)

    rows = []
    for name, values in variants.items():
        v = values.dropna()
        rows.append(
            {
                "transform": name,
                "mean": round(float(v.mean()), 3),
                "std": round(float(v.std()), 3),
                "min": round(float(v.min()), 3),
                "max": round(float(v.max()), 3),
                "skew": round(float(stats.skew(v)), 3),
                "kurtosis": round(float(stats.kurtosis(v)), 3),
            }
        )
    return pd.DataFrame(rows).set_index("transform")
