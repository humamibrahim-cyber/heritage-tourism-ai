"""Does the rating data actually contain preference signal?

This module exists because of a finding on the supplied data: the
`Place_Ratings` column in `tourism_rating.csv` is statistically
indistinguishable from random numbers. Before reporting that a recommender
"works" or "fails", it is worth establishing whether there is anything in the
data for it to learn.

Five independent checks, each of which a genuine ratings dataset should pass:

1. **Shape** - human ratings are J-shaped (most 4s and 5s, mean ~4.0-4.3).
   Near-uniform 1-5 with mean ~3.0 is what a random generator produces.
2. **Place effects** - if some places are better than others, an ANOVA across
   places should be significant.
3. **Split-half reliability** - split the ratings in half at random; a place's
   mean in one half should predict its mean in the other. This is the single
   most direct test, and it is compared against a shuffled null.
4. **External validity** - the places table carries an independent listed
   rating (Google-style). Real user ratings should correlate with it.
5. **Random floor** - any recommender should beat randomly picking places.

Run `diagnose()` for all five plus a verdict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Typical mean of a real 1-5 star ratings corpus (MovieLens, Amazon, Yelp all
# fall in this band). Used only for interpretation, never as a hard threshold.
HUMAN_MEAN_RANGE = (3.8, 4.3)


def rating_shape_test(ratings: pd.DataFrame, col: str = "place_ratings") -> dict:
    """Check 1 - is the rating distribution shaped like human behaviour?"""
    from scipy import stats

    counts = ratings[col].value_counts().sort_index()
    chi2, p = stats.chisquare(counts.values)
    mean = float(ratings[col].mean())

    # Skew is the informative statistic: human ratings skew negative
    # (a long tail of low scores under a mass of 4s and 5s).
    skew = float(stats.skew(ratings[col]))

    return {
        "check": "rating shape",
        "mean": round(mean, 3),
        "skew": round(skew, 3),
        "chi2_vs_uniform": round(float(chi2), 2),
        "p_value": round(float(p), 4),
        "human_like": bool(HUMAN_MEAN_RANGE[0] <= mean <= HUMAN_MEAN_RANGE[1] and skew < -0.2),
    }


def place_effect_anova(ratings: pd.DataFrame, min_ratings: int = 5) -> dict:
    """Check 2 - do places genuinely differ in how they are rated?"""
    from scipy import stats

    groups = [
        g["place_ratings"].values
        for _, g in ratings.groupby("place_id")
        if len(g) >= min_ratings
    ]
    if len(groups) < 2:
        raise ValueError("Not enough places with sufficient ratings for an ANOVA.")

    f_stat, p = stats.f_oneway(*groups)

    grand = ratings["place_ratings"].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = float(((ratings["place_ratings"] - grand) ** 2).sum())
    eta_sq = ss_between / ss_total if ss_total else 0.0

    return {
        "check": "place effects",
        "n_places": len(groups),
        "F": round(float(f_stat), 3),
        "p_value": round(float(p), 4),
        "eta_squared": round(float(eta_sq), 5),
        "variance_explained_%": round(100 * float(eta_sq), 3),
        "significant": bool(p < 0.05),
    }


def group_effect_anova(merged: pd.DataFrame, by: str) -> dict:
    """Generalised check 2 - do ratings differ by category, city, age band…?"""
    from scipy import stats

    groups = [g["place_ratings"].values for _, g in merged.groupby(by) if len(g) > 1]
    f_stat, p = stats.f_oneway(*groups)

    grand = merged["place_ratings"].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = float(((merged["place_ratings"] - grand) ** 2).sum())
    eta_sq = ss_between / ss_total if ss_total else 0.0

    return {
        "check": f"{by} effects",
        "n_groups": len(groups),
        "F": round(float(f_stat), 3),
        "p_value": round(float(p), 4),
        "variance_explained_%": round(100 * float(eta_sq), 3),
        "significant": bool(p < 0.05),
    }


def split_half_reliability(
    ratings: pd.DataFrame, n_shuffles: int = 50, seed: int = 42
) -> dict:
    """Check 3 - is a place's average rating reproducible?

    Split the ratings randomly in two and correlate each place's mean across
    the halves. Real preference data gives a clearly positive correlation.
    The shuffled null tells us what pure noise looks like at this sample size,
    which is the comparison that actually matters.
    """
    rng = np.random.default_rng(seed)

    def _corr(df: pd.DataFrame, random_state: int) -> float:
        half_a = df.sample(frac=0.5, random_state=random_state)
        half_b = df.drop(half_a.index)
        mean_a = half_a.groupby("place_id")["place_ratings"].mean()
        mean_b = half_b.groupby("place_id")["place_ratings"].mean()
        common = mean_a.index.intersection(mean_b.index)
        if len(common) < 3:
            return np.nan
        return float(np.corrcoef(mean_a[common], mean_b[common])[0, 1])

    observed = _corr(ratings, random_state=1)

    null = []
    for _ in range(n_shuffles):
        shuffled = ratings.copy()
        shuffled["place_ratings"] = rng.permutation(shuffled["place_ratings"].values)
        value = _corr(shuffled, random_state=1)
        if not np.isnan(value):
            null.append(value)

    null_mean, null_sd = float(np.mean(null)), float(np.std(null))
    z = (observed - null_mean) / null_sd if null_sd else 0.0

    return {
        "check": "split-half reliability",
        "observed_r": round(observed, 4),
        "shuffled_r_mean": round(null_mean, 4),
        "shuffled_r_sd": round(null_sd, 4),
        "z_score": round(float(z), 2),
        # z > 2 means the observed correlation sits outside the noise band.
        "exceeds_noise": bool(z > 2),
    }


def external_validity(merged: pd.DataFrame, min_ratings: int = 10) -> dict:
    """Check 4 - do user ratings agree with the independent listed rating?

    The places table carries a `rating` column sourced independently of the
    user ratings. If the user ratings describe the same world, the two should
    correlate. This is the hardest check to explain away.
    """
    from scipy import stats

    if "rating" not in merged.columns:
        return {"check": "external validity", "available": False}

    per_place = merged.groupby("place_id").agg(
        user_mean=("place_ratings", "mean"),
        listed=("rating", "first"),
        n=("place_ratings", "size"),
    )
    per_place = per_place[per_place["n"] >= min_ratings].dropna()
    if len(per_place) < 10:
        return {"check": "external validity", "available": False}

    r, p = stats.pearsonr(per_place["listed"], per_place["user_mean"])
    return {
        "check": "external validity",
        "available": True,
        "n_places": int(len(per_place)),
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 4),
        "agrees": bool(p < 0.05 and r > 0.1),
    }


def make_random_recommend_fn(place_ids, seed: int = 42):
    """Check 5 - a recommender that picks at random. The true performance floor.

    A popularity baseline is not the floor; random is. If a trained model cannot
    beat random, it has learned nothing.
    """
    rng = np.random.default_rng(seed)
    pool = np.asarray(place_ids)

    def fn(user_id, k, seen):
        candidates = pool[~np.isin(pool, list(seen))]
        if len(candidates) == 0:
            return []
        size = min(k, len(candidates))
        return list(rng.choice(candidates, size=size, replace=False))

    return fn


def diagnose(ratings: pd.DataFrame, merged: pd.DataFrame | None = None,
             verbose: bool = True) -> tuple[pd.DataFrame, str]:
    """Run every check and return (results table, plain-English verdict)."""
    results = [
        rating_shape_test(ratings),
        place_effect_anova(ratings),
        split_half_reliability(ratings),
    ]
    if merged is not None:
        results.append(external_validity(merged))
        for column in ("category", "city"):
            if column in merged.columns:
                results.append(group_effect_anova(merged, column))

    # Long format: each check reports different fields, so a wide table would
    # be mostly NaN and unreadable.
    rows = []
    for entry in results:
        name = entry["check"]
        for key, value in entry.items():
            if key == "check":
                continue
            rows.append({"check": name, "metric": key, "value": value})
    table = pd.DataFrame(rows)

    # Weigh the three checks that speak most directly to preference signal.
    shape = results[0]
    places = results[1]
    reliability = results[2]
    external = next((r for r in results if r["check"] == "external validity"), None)

    failed = []
    if not shape["human_like"]:
        failed.append(
            f"the distribution is not human-shaped (mean {shape['mean']}, "
            f"skew {shape['skew']}; real corpora sit at 3.8-4.3 with negative skew)"
        )
    if not places["significant"]:
        failed.append(
            f"places do not differ significantly (ANOVA p={places['p_value']}, "
            f"only {places['variance_explained_%']}% of variance explained)"
        )
    if not reliability["exceeds_noise"]:
        failed.append(
            f"place means are not reproducible (split-half r="
            f"{reliability['observed_r']}, shuffled r={reliability['shuffled_r_mean']})"
        )
    if external and external.get("available") and not external["agrees"]:
        failed.append(
            f"user ratings do not correlate with the independent listed rating "
            f"(r={external['pearson_r']}, p={external['p_value']})"
        )

    if len(failed) >= 3:
        verdict = (
            "NO USABLE PREFERENCE SIGNAL. " + str(len(failed)) + " independent checks failed: "
            + "; ".join(failed) + ". These ratings behave like randomly generated "
            "numbers. No collaborative filtering model can outperform a random "
            "recommender on this data, because there is no structure to learn. "
            "Report this as the finding - it is a real result, not a modelling failure."
        )
    elif failed:
        verdict = (
            "WEAK SIGNAL. Some checks failed: " + "; ".join(failed) + ". Treat any "
            "model improvement as provisional and compare against the random floor."
        )
    else:
        verdict = (
            "SIGNAL PRESENT. The ratings show human-like shape, significant "
            "between-place differences and reproducible place means. "
            "Collaborative filtering is worth pursuing here."
        )

    if verbose:
        print(table.to_string(index=False))
        print("\n" + "=" * 72)
        print("VERDICT")
        print("=" * 72)
        for line in _wrap(verdict, 72):
            print(line)

    return table, verdict


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines
