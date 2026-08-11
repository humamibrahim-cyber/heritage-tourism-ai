"""Item-based collaborative filtering for Part 2.

This is the model that answers the brief's exact wording: "recommend other
places to visit using the current tourist location (place name)".

Method
------
1. Build the users x places rating matrix.
2. Mean-centre each user's row. Without this, a generous user who rates
   everything 5 and a harsh user who rates everything 3 look like they have
   opposite tastes when in fact they may agree on rank order.
3. Cosine similarity between place columns.
4. Recommend the nearest places to a query place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ItemBasedCF:
    """Cosine item-item collaborative filtering with a place-name interface."""

    def __init__(self, min_ratings_per_place: int = 3, shrinkage: float = 10.0):
        self.min_ratings_per_place = min_ratings_per_place
        # Shrinkage damps similarities computed from very few co-ratings, which
        # are otherwise wildly overconfident (two co-raters can give cos = 1.0).
        self.shrinkage = shrinkage
        self.similarity_: pd.DataFrame | None = None
        self.place_lookup_: pd.DataFrame | None = None
        self.matrix_: pd.DataFrame | None = None
        self.co_counts_: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, ratings: pd.DataFrame, places: pd.DataFrame) -> "ItemBasedCF":
        matrix = ratings.pivot_table(
            index="user_id", columns="place_id", values="place_ratings", aggfunc="mean"
        )

        keep = matrix.notna().sum(axis=0) >= self.min_ratings_per_place
        dropped = int((~keep).sum())
        if dropped:
            print(
                f"ItemBasedCF: excluded {dropped} places with fewer than "
                f"{self.min_ratings_per_place} ratings (too sparse to be reliable "
                f"neighbours). {int(keep.sum())} places retained."
            )
        matrix = matrix.loc[:, keep]

        # Mean-centre per user, then treat unrated as neutral (0 after centring).
        centred = matrix.sub(matrix.mean(axis=1), axis=0).fillna(0.0)
        values = centred.to_numpy(dtype=np.float32)

        norms = np.linalg.norm(values, axis=0, keepdims=True)
        norms[norms == 0] = 1e-9
        similarity = (values.T @ values) / (norms.T @ norms)

        # Shrink by number of users who rated both places.
        rated = matrix.notna().to_numpy(dtype=np.float32)
        co_counts = rated.T @ rated
        similarity *= co_counts / (co_counts + self.shrinkage)

        np.fill_diagonal(similarity, 0.0)  # never recommend the query itself

        self.matrix_ = matrix
        self.co_counts_ = co_counts
        self.similarity_ = pd.DataFrame(
            similarity, index=matrix.columns, columns=matrix.columns
        )
        self.place_lookup_ = places.set_index("place_id")
        return self

    # ------------------------------------------------------------------
    def _resolve_place(self, place_name: str) -> int:
        """Map a (possibly partial, case-insensitive) place name to its id."""
        if self.place_lookup_ is None:
            raise RuntimeError("Call fit() first.")
        names = self.place_lookup_["place_name"].astype(str)

        exact = names[names.str.lower() == str(place_name).lower()]
        if len(exact):
            return int(exact.index[0])

        partial = names[names.str.contains(str(place_name), case=False, regex=False)]
        if len(partial) == 1:
            return int(partial.index[0])
        if len(partial) > 1:
            options = ", ".join(partial.head(5).tolist())
            raise ValueError(f"'{place_name}' is ambiguous. Did you mean: {options}?")
        raise KeyError(f"'{place_name}' not found in the places table.")

    # ------------------------------------------------------------------
    def recommend_similar(self, place_name: str, n: int = 10) -> pd.DataFrame:
        """Top-n places most similar to ``place_name``. The brief's core ask."""
        if self.similarity_ is None:
            raise RuntimeError("Call fit() first.")
        pid = self._resolve_place(place_name)
        if pid not in self.similarity_.index:
            raise KeyError(
                f"'{place_name}' has fewer than {self.min_ratings_per_place} ratings, "
                f"so it was excluded from the similarity matrix (cold start)."
            )

        scores = self.similarity_.loc[pid].sort_values(ascending=False).head(n)
        out = self.place_lookup_.loc[scores.index]
        out.index.name = "place_id"
        out = out.reset_index()
        cols = [c for c in ["place_id", "place_name", "category", "city", "rating", "price"]
                if c in out.columns]
        out = out[cols]
        out.insert(1, "similarity", scores.to_numpy().round(4))
        out.insert(0, "rank", range(1, len(out) + 1))
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    def recommend_for_user(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """Weighted-sum personalised recommendations from the user's history."""
        if self.matrix_ is None:
            raise RuntimeError("Call fit() first.")
        if user_id not in self.matrix_.index:
            raise KeyError(f"user_id {user_id} unseen during fit (cold start).")

        user_row = self.matrix_.loc[user_id]
        rated = user_row.dropna()
        if rated.empty:
            raise ValueError(f"user {user_id} has no ratings.")

        sim = self.similarity_.loc[rated.index]           # rated x all
        centred = rated - rated.mean()
        numerator = sim.mul(centred, axis=0).sum(axis=0)
        denominator = sim.abs().sum(axis=0).replace(0, np.nan)
        scores = (numerator / denominator).drop(index=rated.index, errors="ignore")
        top = scores.sort_values(ascending=False).head(n).dropna()

        out = self.place_lookup_.loc[top.index]
        out.index.name = "place_id"
        out = out.reset_index()
        cols = [c for c in ["place_id", "place_name", "category", "city", "rating"]
                if c in out.columns]
        out = out[cols]
        out.insert(1, "score", top.to_numpy().round(4))
        out.insert(0, "rank", range(1, len(out) + 1))
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    def predict(self, user_ids, place_ids) -> np.ndarray:
        """Predicted ratings, for RMSE comparison against the Keras model."""
        if self.matrix_ is None:
            raise RuntimeError("Call fit() first.")
        global_mean = float(np.nanmean(self.matrix_.to_numpy()))
        user_means = self.matrix_.mean(axis=1)
        preds = []

        for uid, pid in zip(user_ids, place_ids):
            base = float(user_means.get(uid, global_mean))
            if uid not in self.matrix_.index or pid not in self.similarity_.index:
                preds.append(base)
                continue
            rated = self.matrix_.loc[uid].dropna()
            rated = rated[rated.index != pid]
            if rated.empty:
                preds.append(base)
                continue
            sims = self.similarity_.loc[pid, rated.index]
            denom = sims.abs().sum()
            if denom < 1e-8:
                preds.append(base)
                continue
            preds.append(base + float((sims * (rated - base)).sum() / denom))

        return np.clip(np.array(preds), 1.0, 5.0)


class PopularityBaseline:
    """The bar every recommender must clear.

    Recommending the globally most-popular places to everyone is a shockingly
    strong baseline on small datasets. If the trained models cannot beat it,
    that is the finding, and it belongs in the report.
    """

    def __init__(self, min_ratings: int = 5):
        self.min_ratings = min_ratings
        self.global_mean_: float = 0.0
        self.place_scores_: pd.Series | None = None
        self.ranking_: list[int] = []

    def fit(self, ratings: pd.DataFrame) -> "PopularityBaseline":
        stats = ratings.groupby("place_id")["place_ratings"].agg(["count", "mean"])
        self.global_mean_ = float(ratings["place_ratings"].mean())
        m = self.min_ratings
        stats["score"] = (
            (stats["count"] * stats["mean"] + m * self.global_mean_) / (stats["count"] + m)
        )
        self.place_scores_ = stats["score"]
        self.ranking_ = stats.sort_values("score", ascending=False).index.tolist()
        return self

    def predict(self, user_ids, place_ids) -> np.ndarray:
        return np.array(
            [float(self.place_scores_.get(p, self.global_mean_)) for p in place_ids]
        )

    def recommend_for_user(self, user_id, n: int = 10, exclude=None) -> list[int]:
        exclude = set(exclude or [])
        return [p for p in self.ranking_ if p not in exclude][:n]
