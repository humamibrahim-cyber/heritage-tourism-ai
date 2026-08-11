"""Keras embedding matrix factorisation for Part 2.

Model
-----
    score(u, i) = sigmoid( <user_emb[u], place_emb[i]> + b_u + b_i )

Ratings are min-max scaled to [0, 1] so the sigmoid output is directly
comparable, then mapped back to the 1-5 scale for reporting.

Why deliberately small
----------------------
This dataset has ~300 users, ~437 places and ~10k ratings - a ~92% sparse
matrix. A high-capacity neural recommender memorises it within a handful of
epochs. Embedding dim 32 with L2 regularisation and early stopping is the
honest configuration, and it is what the report should defend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class RatingEncoder:
    """Maps raw user/place ids to contiguous embedding indices."""

    def __init__(self):
        self.user_to_idx: dict = {}
        self.place_to_idx: dict = {}
        self.idx_to_user: dict = {}
        self.idx_to_place: dict = {}
        self.min_rating: float = 1.0
        self.max_rating: float = 5.0

    def fit(self, ratings: pd.DataFrame) -> "RatingEncoder":
        users = sorted(ratings["user_id"].unique())
        places = sorted(ratings["place_id"].unique())
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.place_to_idx = {p: i for i, p in enumerate(places)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.idx_to_place = {i: p for p, i in self.place_to_idx.items()}
        self.min_rating = float(ratings["place_ratings"].min())
        self.max_rating = float(ratings["place_ratings"].max())
        return self

    @property
    def n_users(self) -> int:
        return len(self.user_to_idx)

    @property
    def n_places(self) -> int:
        return len(self.place_to_idx)

    def transform(self, ratings: pd.DataFrame, scale_y: bool = True):
        """Drop unseen ids (cold start) and return (X, y)."""
        df = ratings[
            ratings["user_id"].isin(self.user_to_idx)
            & ratings["place_id"].isin(self.place_to_idx)
        ]
        dropped = len(ratings) - len(df)
        if dropped:
            print(f"RatingEncoder: dropped {dropped} rows with unseen user/place ids.")

        x = np.column_stack(
            [
                df["user_id"].map(self.user_to_idx).to_numpy(),
                df["place_id"].map(self.place_to_idx).to_numpy(),
            ]
        ).astype("int32")

        y = df["place_ratings"].to_numpy(dtype="float32")
        if scale_y:
            span = max(self.max_rating - self.min_rating, 1e-6)
            y = (y - self.min_rating) / span
        return x, y

    def inverse_scale(self, y_scaled):
        span = self.max_rating - self.min_rating
        return np.asarray(y_scaled) * span + self.min_rating


def build_recommender_net(n_users: int, n_places: int, cfg):
    """Biased matrix factorisation as a Keras functional model."""
    import tensorflow as tf

    reg = tf.keras.regularizers.l2(cfg.l2_reg)

    user_in = tf.keras.Input(shape=(1,), dtype="int32", name="user")
    place_in = tf.keras.Input(shape=(1,), dtype="int32", name="place")

    user_emb = tf.keras.layers.Embedding(
        n_users, cfg.embedding_dim,
        embeddings_initializer="he_normal", embeddings_regularizer=reg,
        name="user_embedding",
    )(user_in)
    place_emb = tf.keras.layers.Embedding(
        n_places, cfg.embedding_dim,
        embeddings_initializer="he_normal", embeddings_regularizer=reg,
        name="place_embedding",
    )(place_in)

    user_bias = tf.keras.layers.Embedding(n_users, 1, name="user_bias")(user_in)
    place_bias = tf.keras.layers.Embedding(n_places, 1, name="place_bias")(place_in)

    dot = tf.keras.layers.Dot(axes=2, name="interaction")([user_emb, place_emb])
    total = tf.keras.layers.Add(name="add_biases")([dot, user_bias, place_bias])
    flat = tf.keras.layers.Flatten()(total)
    out = tf.keras.layers.Activation("sigmoid", name="rating")(flat)

    model = tf.keras.Model([user_in, place_in], out, name="RecommenderNet")
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.mf_lr),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def train_recommender(model, x_train, y_train, x_val, y_val, cfg, verbose: int = 1):
    """Fit with early stopping - overfitting here is a matter of a few epochs."""
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_rmse", mode="min",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_rmse", mode="min", factor=0.5, patience=4,
            min_lr=1e-6, verbose=0,
        ),
    ]
    return model.fit(
        x=[x_train[:, 0], x_train[:, 1]],
        y=y_train,
        validation_data=([x_val[:, 0], x_val[:, 1]], y_val),
        epochs=cfg.mf_epochs,
        batch_size=cfg.mf_batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )


class KerasRecommender:
    """Convenience wrapper giving the Keras model the same API as ItemBasedCF."""

    def __init__(self, model, encoder: RatingEncoder, places: pd.DataFrame):
        self.model = model
        self.encoder = encoder
        self.places = places.set_index("place_id")

    # ------------------------------------------------------------------
    def predict(self, user_ids, place_ids) -> np.ndarray:
        u = np.array([self.encoder.user_to_idx.get(x, -1) for x in user_ids])
        p = np.array([self.encoder.place_to_idx.get(x, -1) for x in place_ids])
        valid = (u >= 0) & (p >= 0)

        preds = np.full(len(u), (self.encoder.min_rating + self.encoder.max_rating) / 2)
        if valid.any():
            scaled = self.model.predict(
                [u[valid], p[valid]], verbose=0
            ).ravel()
            preds[valid] = self.encoder.inverse_scale(scaled)
        return np.clip(preds, self.encoder.min_rating, self.encoder.max_rating)

    # ------------------------------------------------------------------
    def recommend_for_user(self, user_id, n: int = 10, exclude_seen=None) -> pd.DataFrame:
        """Score every place for one user and return the top-n unseen."""
        if user_id not in self.encoder.user_to_idx:
            raise KeyError(f"user_id {user_id} unseen during training (cold start).")

        place_ids = np.array(list(self.encoder.place_to_idx.keys()))
        exclude = set(exclude_seen or [])
        mask = ~np.isin(place_ids, list(exclude))
        candidates = place_ids[mask]

        u_idx = np.full(len(candidates), self.encoder.user_to_idx[user_id])
        p_idx = np.array([self.encoder.place_to_idx[p] for p in candidates])
        scaled = self.model.predict([u_idx, p_idx], verbose=0).ravel()
        preds = self.encoder.inverse_scale(scaled)

        order = np.argsort(-preds)[:n]
        out = self.places.loc[candidates[order]]
        out.index.name = "place_id"
        out = out.reset_index()
        cols = [c for c in ["place_id", "place_name", "category", "city", "price"]
                if c in out.columns]
        out = out[cols]
        out.insert(1, "predicted_rating", preds[order].round(3))
        out.insert(0, "rank", range(1, len(out) + 1))
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    def place_embedding_matrix(self) -> pd.DataFrame:
        """Learned place vectors - the second route to place-to-place similarity."""
        weights = self.model.get_layer("place_embedding").get_weights()[0]
        index = pd.Index(
            [self.encoder.idx_to_place[i] for i in range(len(weights))],
            name="place_id",  # must be named, or downstream reset_index() loses it
        )
        return pd.DataFrame(weights, index=index)

    def similar_places(self, place_name: str, n: int = 10) -> pd.DataFrame:
        """Cosine similarity in the learned embedding space.

        Same question as ItemBasedCF.recommend_similar, answered by the neural
        model instead - a useful cross-check in the report.
        """
        names = self.places["place_name"].astype(str)
        match = names[names.str.lower() == str(place_name).lower()]
        if match.empty:
            match = names[names.str.contains(str(place_name), case=False, regex=False)]
        if match.empty:
            raise KeyError(f"'{place_name}' not found.")
        pid = int(match.index[0])

        emb = self.place_embedding_matrix()
        if pid not in emb.index:
            raise KeyError(f"'{place_name}' has no learned embedding (no ratings).")

        values = emb.to_numpy()
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        unit = values / norms
        sims = unit @ unit[emb.index.get_loc(pid)]

        series = pd.Series(sims, index=emb.index).drop(index=pid)
        top = series.sort_values(ascending=False).head(n)

        out = self.places.loc[top.index]
        out.index.name = "place_id"   # belt and braces - see place_embedding_matrix
        out = out.reset_index()
        cols = [c for c in ["place_id", "place_name", "category", "city"]
                if c in out.columns]
        out = out[cols]
        out.insert(1, "similarity", top.to_numpy().round(4))
        out.insert(0, "rank", range(1, len(out) + 1))
        return out.reset_index(drop=True)
