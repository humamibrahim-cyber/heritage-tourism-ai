"""Smoke tests on synthetic data.

These do not prove the model is accurate - they prove the code runs, the
shapes line up, and the metrics behave sanely. Run before every commit:

    pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLASS_NAMES, ImageConfig, RecoConfig  # noqa: E402
from src.evaluation import ranking  # noqa: E402
from src.models.item_cf import ItemBasedCF, PopularityBaseline  # noqa: E402

tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_tourism():
    """40 users x 30 places with a planted two-cluster taste structure.

    Users in cluster A prefer places 0-14, cluster B prefer 15-29. A working
    item-CF must recover that block structure.
    """
    rng = np.random.default_rng(0)
    rows = []
    for user_id in range(40):
        cluster = user_id % 2
        for place_id in range(30):
            if rng.random() < 0.35:
                preferred = (place_id < 15) == (cluster == 0)
                base = 4.6 if preferred else 2.4
                rows.append(
                    {
                        "user_id": user_id,
                        "place_id": place_id,
                        "place_ratings": float(
                            np.clip(round(base + rng.normal(0, 0.5)), 1, 5)
                        ),
                    }
                )
    ratings = pd.DataFrame(rows)
    places = pd.DataFrame(
        {
            "place_id": range(30),
            "place_name": [f"Place {i:02d}" for i in range(30)],
            "category": ["Cluster A" if i < 15 else "Cluster B" for i in range(30)],
            "city": ["Jakarta" if i % 2 else "Bandung" for i in range(30)],
            "rating": rng.uniform(3, 5, 30).round(1),
        }
    )
    return ratings, places


# --------------------------------------------------------------------------
# Part 2 - item-based collaborative filtering
# --------------------------------------------------------------------------
def test_item_cf_recovers_cluster_structure(synthetic_tourism):
    ratings, places = synthetic_tourism
    model = ItemBasedCF(min_ratings_per_place=3).fit(ratings, places)

    recs = model.recommend_similar("Place 03", n=5)
    assert len(recs) == 5
    assert "Place 03" not in recs["place_name"].tolist(), "must not recommend itself"
    # Place 03 is in cluster A, so its neighbours should be too.
    same_cluster = (recs["category"] == "Cluster A").sum()
    assert same_cluster >= 4, f"only {same_cluster}/5 neighbours from the same cluster"


def test_item_cf_partial_name_lookup(synthetic_tourism):
    ratings, places = synthetic_tourism
    model = ItemBasedCF(min_ratings_per_place=3).fit(ratings, places)
    assert model._resolve_place("place 07") == 7  # case-insensitive
    with pytest.raises(KeyError):
        model.recommend_similar("Nonexistent Temple")


def test_item_cf_similarity_is_symmetric_and_bounded(synthetic_tourism):
    ratings, places = synthetic_tourism
    model = ItemBasedCF(min_ratings_per_place=3).fit(ratings, places)
    sim = model.similarity_.to_numpy()
    assert np.allclose(sim, sim.T, atol=1e-5), "similarity matrix must be symmetric"
    assert sim.max() <= 1.0001 and sim.min() >= -1.0001
    assert np.allclose(np.diag(sim), 0.0), "self-similarity must be zeroed"


def test_item_cf_predictions_in_valid_range(synthetic_tourism):
    ratings, places = synthetic_tourism
    model = ItemBasedCF(min_ratings_per_place=3).fit(ratings, places)
    preds = model.predict(ratings["user_id"].head(50), ratings["place_id"].head(50))
    assert preds.shape == (50,)
    assert preds.min() >= 1.0 and preds.max() <= 5.0
    assert not np.isnan(preds).any()


def test_popularity_baseline(synthetic_tourism):
    ratings, _ = synthetic_tourism
    base = PopularityBaseline(min_ratings=5).fit(ratings)
    recs = base.recommend_for_user(0, n=10, exclude={0, 1, 2})
    assert len(recs) == 10
    assert not ({0, 1, 2} & set(recs)), "excluded places leaked into recommendations"


# --------------------------------------------------------------------------
# Ranking metrics
# --------------------------------------------------------------------------
def test_ndcg_perfect_and_empty():
    assert ranking.ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)
    assert ranking.ndcg_at_k([7, 8, 9], {1, 2, 3}, k=3) == 0.0


def test_ndcg_rewards_higher_placement():
    top = ranking.ndcg_at_k([1, 9, 9, 9, 9], {1}, k=5)
    bottom = ranking.ndcg_at_k([9, 9, 9, 9, 1], {1}, k=5)
    assert top > bottom, "a hit at rank 1 must score above a hit at rank 5"


def test_rating_metrics_zero_error():
    m = ranking.rating_metrics([1, 2, 3], [1, 2, 3])
    assert m["rmse"] == 0.0 and m["mae"] == 0.0 and m["n"] == 3


def test_evaluate_topk_runs(synthetic_tourism):
    ratings, places = synthetic_tourism
    train = ratings.sample(frac=0.8, random_state=1)
    test = ratings.drop(train.index)

    model = ItemBasedCF(min_ratings_per_place=2).fit(train, places)
    result = ranking.evaluate_topk(
        ranking.make_itemcf_recommend_fn(model),
        train, test, k=5, relevance_threshold=4.0,
        n_places_total=30, verbose=False,
    )
    assert 0.0 <= result["precision@5"] <= 1.0
    assert 0.0 <= result["ndcg@5"] <= 1.0
    assert result["users_evaluated"] > 0


# --------------------------------------------------------------------------
# Part 2 - Keras matrix factorisation
# --------------------------------------------------------------------------
def test_recommender_net_trains_and_beats_random(synthetic_tourism):
    from src.models.recommender_net import (
        KerasRecommender,
        RatingEncoder,
        build_recommender_net,
        train_recommender,
    )

    ratings, places = synthetic_tourism
    cfg = RecoConfig(embedding_dim=8, mf_epochs=25, early_stopping_patience=25)

    encoder = RatingEncoder().fit(ratings)
    x, y = encoder.transform(ratings)
    split = int(0.8 * len(x))

    model = build_recommender_net(encoder.n_users, encoder.n_places, cfg)
    history = train_recommender(
        model, x[:split], y[:split], x[split:], y[split:], cfg, verbose=0
    )

    first, last = history.history["loss"][0], history.history["loss"][-1]
    assert last < first, f"loss did not improve ({first:.4f} -> {last:.4f})"

    rec = KerasRecommender(model, encoder, places)
    top = rec.recommend_for_user(0, n=5, exclude_seen=set())
    assert len(top) == 5
    assert top["predicted_rating"].between(1, 5).all()

    preds = rec.predict([0, 1], [0, 1])
    assert preds.shape == (2,) and np.isfinite(preds).all()

    similar = rec.similar_places("Place 03", n=5)
    assert len(similar) == 5 and "Place 03" not in similar["place_name"].tolist()


REQUIRED_COLS = {"rank", "place_id", "place_name"}


def test_all_recommenders_return_place_id(synthetic_tourism):
    """Regression test.

    Every recommender returns a DataFrame built via ``.loc[...].reset_index()``.
    Indexing with an *unnamed* pandas Index silently discards the original index
    name, so reset_index() produced a column called 'index' and the column
    filter then dropped place_id without raising. Downstream code that looked up
    ["place_id"] blew up far away from the cause. Assert the contract directly.
    """
    from src.models.recommender_net import (
        KerasRecommender, RatingEncoder, build_recommender_net, train_recommender,
    )

    ratings, places = synthetic_tourism
    cfg = RecoConfig(embedding_dim=8, mf_epochs=3, early_stopping_patience=3)

    item_cf = ItemBasedCF(min_ratings_per_place=3).fit(ratings, places)
    encoder = RatingEncoder().fit(ratings)
    x, y = encoder.transform(ratings)
    mf = build_recommender_net(encoder.n_users, encoder.n_places, cfg)
    train_recommender(mf, x[:300], y[:300], x[300:], y[300:], cfg, verbose=0)
    keras_rec = KerasRecommender(mf, encoder, places)

    frames = {
        "itemcf.recommend_similar": item_cf.recommend_similar("Place 03", n=5),
        "itemcf.recommend_for_user": item_cf.recommend_for_user(0, n=5),
        "keras.recommend_for_user": keras_rec.recommend_for_user(0, n=5, exclude_seen=set()),
        "keras.similar_places": keras_rec.similar_places("Place 03", n=5),
    }
    for label, df in frames.items():
        missing = REQUIRED_COLS - set(df.columns)
        assert not missing, f"{label} is missing {missing}; got {list(df.columns)}"
        assert df["place_id"].notna().all(), f"{label} returned null place_ids"


def test_encoder_handles_unseen_ids(synthetic_tourism):
    from src.models.recommender_net import RatingEncoder

    ratings, _ = synthetic_tourism
    encoder = RatingEncoder().fit(ratings)
    unseen = pd.DataFrame(
        {"user_id": [999], "place_id": [999], "place_ratings": [5.0]}
    )
    x, y = encoder.transform(unseen)
    assert len(x) == 0, "unseen ids must be dropped, not crash"


def test_rating_scaling_roundtrip(synthetic_tourism):
    from src.models.recommender_net import RatingEncoder

    ratings, _ = synthetic_tourism
    encoder = RatingEncoder().fit(ratings)
    _, y = encoder.transform(ratings, scale_y=True)
    assert 0.0 <= y.min() and y.max() <= 1.0
    restored = encoder.inverse_scale(y)
    assert np.allclose(restored, ratings["place_ratings"].to_numpy(), atol=1e-4)


# --------------------------------------------------------------------------
# Part 1 - image classifier graph
# --------------------------------------------------------------------------
@pytest.mark.parametrize("backbone", ["efficientnetv2b0", "resnet50v2", "mobilenetv2"])
def test_classifier_builds_and_freezes(backbone):
    """Random weights keep this offline and fast; the graph is what we check."""
    from src.models.backbones import build_classifier, compile_model, get_base

    cfg = ImageConfig(image_size=(96, 96), dense_units=32)
    model = build_classifier(cfg, backbone=backbone, weights=None)
    compile_model(model, lr=1e-3)

    assert model.output_shape == (None, len(CLASS_NAMES))
    base = get_base(model)
    assert base.trainable is False, "convolutional base must start frozen (task 3)"

    out = model.predict(np.random.rand(2, 96, 96, 3).astype("float32"), verbose=0)
    assert out.shape == (2, len(CLASS_NAMES))
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4), "softmax must sum to 1"


def test_unfreeze_keeps_batchnorm_frozen():
    from src.models.backbones import build_classifier, get_base, unfreeze_top

    cfg = ImageConfig(image_size=(96, 96), dense_units=32)
    model = build_classifier(cfg, backbone="mobilenetv2", weights=None)
    unfreeze_top(model, fraction=0.3, verbose=False)

    base = get_base(model)
    bn_layers = [
        l for l in base.layers if isinstance(l, tf.keras.layers.BatchNormalization)
    ]
    assert bn_layers, "expected BatchNorm layers in MobileNetV2"
    assert all(not l.trainable for l in bn_layers), "BatchNorm must stay frozen"
    assert any(l.trainable for l in base.layers), "nothing was unfrozen"


def test_accuracy_threshold_callback_stops_training():
    from src.training.callbacks import make_accuracy_threshold_callback

    cb = make_accuracy_threshold_callback(target=0.90)

    class Dummy:
        stop_training = False

    # Keras 3 exposes `model` as a read-only property; set_model is the hook.
    cb.set_model(Dummy())
    cb.on_epoch_end(0, {"val_accuracy": 0.85})
    assert cb.model.stop_training is False

    cb.on_epoch_end(1, {"val_accuracy": 0.94})
    assert cb.model.stop_training is True
    assert cb.stopped_epoch == 2


def test_augmentation_changes_images_but_not_shape():
    from src.data.image_data import build_augmentation

    aug = build_augmentation()
    batch = np.random.rand(4, 96, 96, 3).astype("float32") * 255
    out = aug(batch, training=True).numpy()
    assert out.shape == batch.shape
    assert not np.allclose(out, batch), "augmentation had no effect"

    passthrough = aug(batch, training=False).numpy()
    assert np.allclose(passthrough, batch, atol=1e-3), "must be inactive at inference"


def test_class_weights_favour_rare_classes():
    from src.config import EXPECTED_TRAIN_COUNTS

    total = sum(EXPECTED_TRAIN_COUNTS.values())
    n = len(EXPECTED_TRAIN_COUNTS)
    weights = {c: total / (n * v) for c, v in EXPECTED_TRAIN_COUNTS.items()}
    assert weights["flying_buttress"] > weights["column"], (
        "the rarest class must carry more weight than the most common"
    )
    assert weights["flying_buttress"] / weights["column"] == pytest.approx(
        EXPECTED_TRAIN_COUNTS["column"] / EXPECTED_TRAIN_COUNTS["flying_buttress"], rel=1e-6
    )


# --------------------------------------------------------------------------
# Signal diagnostics
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def random_ratings():
    """Ratings drawn uniformly at random - no preference structure at all.

    This mirrors the supplied tourism_rating.csv, which failed four independent
    signal checks. The diagnostic must catch it.
    """
    rng = np.random.default_rng(7)
    n_users, n_places = 120, 200
    rows = [
        {
            "user_id": int(u),
            "place_id": int(rng.integers(0, n_places)),
            "place_ratings": float(rng.integers(1, 6)),
        }
        for u in range(n_users)
        for _ in range(40)
    ]
    return pd.DataFrame(rows).drop_duplicates(subset=["user_id", "place_id"])


@pytest.fixture(scope="module")
def signalful_ratings():
    """Ratings with a strong, real place-quality effect. Must pass the checks."""
    rng = np.random.default_rng(7)
    n_users, n_places = 120, 200
    quality = rng.uniform(2.0, 5.0, n_places)   # each place has a true quality
    rows = [
        {
            "user_id": int(u),
            "place_id": int(p),
            "place_ratings": float(np.clip(round(quality[p] + rng.normal(0, 0.4)), 1, 5)),
        }
        for u in range(n_users)
        for p in rng.choice(n_places, size=40, replace=False)
    ]
    return pd.DataFrame(rows).drop_duplicates(subset=["user_id", "place_id"])


def test_diagnose_flags_random_ratings(random_ratings):
    from src.evaluation import signal

    table, verdict = signal.diagnose(random_ratings, verbose=False)
    assert "NO USABLE PREFERENCE SIGNAL" in verdict
    assert not table.empty
    assert set(table.columns) == {"check", "metric", "value"}


def test_diagnose_passes_real_signal(signalful_ratings):
    from src.evaluation import signal

    _, verdict = signal.diagnose(signalful_ratings, verbose=False)
    assert "NO USABLE PREFERENCE SIGNAL" not in verdict, (
        "diagnostic must not cry wolf on data that genuinely has structure"
    )


def test_split_half_separates_signal_from_noise(random_ratings, signalful_ratings):
    from src.evaluation import signal

    noise = signal.split_half_reliability(random_ratings, n_shuffles=20)
    real = signal.split_half_reliability(signalful_ratings, n_shuffles=20)

    assert not noise["exceeds_noise"], "random ratings must not look reliable"
    assert real["exceeds_noise"], "structured ratings must exceed the shuffled null"
    assert real["observed_r"] > noise["observed_r"]


def test_place_anova_detects_place_effects(random_ratings, signalful_ratings):
    from src.evaluation import signal

    assert not signal.place_effect_anova(random_ratings)["significant"]
    assert signal.place_effect_anova(signalful_ratings)["significant"]


def test_random_recommender_fn_excludes_seen():
    from src.evaluation.signal import make_random_recommend_fn

    fn = make_random_recommend_fn(range(50), seed=1)
    seen = {0, 1, 2, 3, 4}
    recs = fn(user_id=0, k=10, seen=seen)
    assert len(recs) == 10
    assert not (set(recs) & seen), "random recommender leaked already-seen places"
    assert len(set(recs)) == 10, "must not repeat within one list"


# --------------------------------------------------------------------------
# Config integrity
# --------------------------------------------------------------------------
def test_config_class_names_match_expected_counts():
    from src.config import EXPECTED_TEST_COUNTS, EXPECTED_TRAIN_COUNTS

    assert set(CLASS_NAMES) == set(EXPECTED_TRAIN_COUNTS)
    assert set(CLASS_NAMES) == set(EXPECTED_TEST_COUNTS)
    assert len(CLASS_NAMES) == 10
    assert list(CLASS_NAMES) == sorted(CLASS_NAMES), (
        "must match image_dataset_from_directory's alphabetical ordering"
    )
    assert sum(EXPECTED_TRAIN_COUNTS.values()) == 10245
    assert sum(EXPECTED_TEST_COUNTS.values()) == 1487
