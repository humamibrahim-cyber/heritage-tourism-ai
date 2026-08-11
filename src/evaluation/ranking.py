"""Recommendation metrics for Part 2.

Two families, because they answer different questions:

  * Rating-prediction error (RMSE / MAE) - "how close is the predicted score?"
  * Top-K ranking quality (Precision, Recall, NDCG, coverage) - "is the list we
    actually show the tourist any good?"

A model can win on RMSE and still produce a useless top-10, which is exactly
why both belong in the report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Rating prediction
# --------------------------------------------------------------------------
def rating_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "n": int(len(y_true)),
    }


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------
def _dcg(relevances: np.ndarray) -> float:
    return float(np.sum(relevances / np.log2(np.arange(2, len(relevances) + 2))))


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    rec = list(recommended)[:k]
    gains = np.array([1.0 if item in relevant else 0.0 for item in rec])
    ideal = np.ones(min(len(relevant), k))
    denominator = _dcg(ideal)
    return _dcg(gains) / denominator if denominator > 0 else 0.0


def evaluate_topk(
    recommend_fn,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
    n_places_total: int | None = None,
    verbose: bool = True,
) -> dict:
    """Score a recommender's top-K lists against held-out ratings.

    ``recommend_fn(user_id, k, seen)`` must return an ordered list of place_ids.

    Only users who have at least one *relevant* (rating >= threshold) held-out
    item are scored - for anyone else precision/recall are undefined, and
    including them as zeros would quietly deflate every model equally but
    misleadingly.
    """
    seen_by_user = train_ratings.groupby("user_id")["place_id"].apply(set).to_dict()
    relevant_by_user = (
        test_ratings[test_ratings["place_ratings"] >= relevance_threshold]
        .groupby("user_id")["place_id"]
        .apply(set)
        .to_dict()
    )

    precisions, recalls, ndcgs, hits = [], [], [], []
    recommended_pool: set = set()
    skipped = 0

    for user_id, relevant in relevant_by_user.items():
        if not relevant:
            continue
        try:
            recs = list(recommend_fn(user_id, k, seen_by_user.get(user_id, set())))
        except (KeyError, ValueError):
            skipped += 1  # cold-start user the model cannot serve
            continue
        if not recs:
            skipped += 1
            continue

        recommended_pool.update(recs)
        n_hit = len(set(recs[:k]) & relevant)
        precisions.append(n_hit / k)
        recalls.append(n_hit / len(relevant))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
        hits.append(1.0 if n_hit > 0 else 0.0)

    if not precisions:
        raise ValueError("No users could be evaluated - check the split and the model.")

    result = {
        f"precision@{k}": round(float(np.mean(precisions)), 4),
        f"recall@{k}": round(float(np.mean(recalls)), 4),
        f"ndcg@{k}": round(float(np.mean(ndcgs)), 4),
        f"hit_rate@{k}": round(float(np.mean(hits)), 4),
        "users_evaluated": len(precisions),
        "users_skipped": skipped,
    }
    if n_places_total:
        # Catalogue coverage: what share of the catalogue ever gets recommended.
        # A model that only ever suggests the same 20 famous places is not
        # doing its job, however good its precision looks.
        result["catalogue_coverage"] = round(len(recommended_pool) / n_places_total, 4)

    if verbose:
        print(pd.Series(result).to_string())
    return result


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Tidy side-by-side comparison table for the report."""
    return pd.DataFrame(results).T


# --------------------------------------------------------------------------
# recommend_fn adapters
# --------------------------------------------------------------------------
def make_keras_recommend_fn(recommender):
    def fn(user_id, k, seen):
        df = recommender.recommend_for_user(user_id, n=k, exclude_seen=seen)
        return df["place_id"].tolist()

    return fn


def make_itemcf_recommend_fn(model):
    def fn(user_id, k, seen):
        df = model.recommend_for_user(user_id, n=k)
        return [p for p in df["place_id"].tolist() if p not in seen][:k]

    return fn


def make_popularity_recommend_fn(baseline):
    def fn(user_id, k, seen):
        return baseline.recommend_for_user(user_id, n=k, exclude=seen)

    return fn
