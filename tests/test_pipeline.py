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
# Descriptive statistics, outliers, missingness, normalisation
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def messy_table():
    """A table with a planted outlier, a skewed column and MAR missingness."""
    rng = np.random.default_rng(3)
    n = 200
    category = rng.choice(["A", "B", "C"], size=n)
    price = rng.exponential(5000, n).round()          # strong right skew
    price[0] = 900_000                                # extreme outlier
    price[1:40] = 0                                   # many free entries
    duration = rng.normal(60, 15, n).round()
    # MAR: category C almost never records a duration
    duration[(category == "C") & (rng.random(n) < 0.9)] = np.nan
    return pd.DataFrame(
        {"category": category, "price": price, "duration": duration,
         "rating": rng.normal(4.4, 0.2, n).round(1)}
    )


def test_numeric_summary_reports_shape(messy_table):
    from src.data import eda

    summary = eda.numeric_summary(messy_table, ["price", "duration"])
    assert summary.loc["price", "skew"] > 2, "should detect the strong right skew"
    assert summary.loc["price", "n_zero"] >= 39
    assert summary.loc["duration", "n_missing"] > 0
    for col in ("mean", "median", "iqr", "cv", "kurtosis", "pct_missing"):
        assert col in summary.columns


@pytest.mark.parametrize("method", ["iqr", "zscore", "mad"])
def test_outlier_methods_find_planted_extreme(messy_table, method):
    from src.data import eda

    mask = eda.detect_outliers(messy_table["price"], method=method)
    assert mask.iloc[0], f"{method} missed the 900,000 outlier"


def test_mad_handles_degenerate_spread():
    """Regression: >50% identical values makes the MAD zero.

    The robust z-score is then 0/0 and the naive implementation reported "no
    outliers" - silently missing the single most obvious anomaly.
    """
    from src.data import eda

    series = pd.Series([1.0] * 50 + [7.5])
    mask = eda.detect_outliers(series, method="mad")
    assert mask.iloc[-1], "degenerate-MAD case must still flag the odd value out"
    assert mask.sum() == 1, "must not flag the identical majority"


def test_outlier_report_covers_all_methods(messy_table):
    from src.data import eda

    report = eda.outlier_report(messy_table, ["price", "rating"])
    for method in ("iqr", "zscore", "mad"):
        assert f"n_{method}" in report.columns
    assert report.loc["price", "n_iqr"] > report.loc["rating", "n_iqr"]


def test_missingness_mechanism_detects_mar(messy_table):
    from src.data import eda

    result = eda.missingness_mechanism(messy_table, "duration", ["category"])
    row = result.iloc[0]
    assert row["p_value"] < 0.05
    assert "MAR" in row["verdict"], "planted category-dependent missingness must be MAR"


def test_suggest_transform_recommends_log_for_skew(messy_table):
    from src.data import eda

    rec = eda.suggest_transform(messy_table["price"])
    assert "log" in rec["recommendation"].lower()
    assert rec["has_zeros"] is True, "log1p (not log) is required when zeros exist"


def test_log1p_reduces_skew(messy_table):
    from scipy import stats

    from src.data import eda

    before = abs(stats.skew(messy_table["price"]))
    after = abs(stats.skew(eda.log1p_transform(messy_table["price"])))
    assert after < before, f"log1p should reduce skew ({before:.2f} -> {after:.2f})"


def test_log1p_rejects_negatives():
    from src.data import eda

    with pytest.raises(ValueError, match="negative"):
        eda.log1p_transform(pd.Series([-1.0, 2.0, 3.0]))


def test_scalings_preserve_ordering(messy_table):
    from src.data import eda

    raw = messy_table["price"]
    for scaled in (eda.robust_scale(raw), eda.minmax_scale(raw), eda.log1p_transform(raw)):
        assert scaled.notna().all()
        # Monotone transforms must not reorder the data.
        assert (raw.rank() == scaled.rank()).all()
    assert eda.minmax_scale(raw).between(0, 1).all()


def test_geographic_outliers_flags_wrong_coordinate():
    """A place whose coordinates sit far from its own city's centroid."""
    from src.data import eda

    places = pd.DataFrame({
        "place_id": range(1, 13),
        "place_name": [f"P{i}" for i in range(1, 13)],
        "city": ["Jakarta"] * 12,
        "lat": [-6.1, -6.2, -6.15, -6.18, -6.12, -6.19,
                -6.14, -6.16, -6.13, -6.17, -6.11, 1.08],   # last one is wrong
        "long": [106.8] * 11 + [103.9],
    })
    flagged = eda.geographic_outliers(places, threshold=5.0)
    assert 12 in flagged["place_id"].values
    assert flagged["km_from_centroid"].abs().max() > 100


def test_country_bounds_check():
    from src.data import eda

    places = pd.DataFrame({
        "place_id": [1, 2], "place_name": ["ok", "bad"], "city": ["Jakarta"] * 2,
        "lat": [-6.2, 48.85], "long": [106.8, 2.35],   # second is Paris
    })
    assert len(eda.country_bounds_check(places)) == 1


# --------------------------------------------------------------------------
# Image data audit (Part 1)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def audited_images(tmp_path_factory):
    """An image tree with deliberately planted quality problems."""
    import shutil

    import cv2

    root = tmp_path_factory.mktemp("imgaudit")
    train, test = root / "train", root / "test"
    rng = np.random.default_rng(0)
    classes = ["altar", "apse", "vault"]

    for directory, n in [(train, 8), (test, 3)]:
        for cls in classes:
            (directory / cls).mkdir(parents=True, exist_ok=True)
            for i in range(n):
                cv2.imwrite(
                    str(directory / cls / f"{i}.jpg"),
                    rng.integers(0, 255, (128, 128, 3), dtype=np.uint8),
                )

    shutil.copy(train / "altar/0.jpg", train / "altar/dup.jpg")        # duplicate
    shutil.copy(train / "altar/1.jpg", train / "apse/crossclass.jpg")  # label noise
    shutil.copy(train / "vault/2.jpg", test / "vault/leak.jpg")        # leakage
    cv2.imwrite(str(train / "apse/blank.jpg"), np.zeros((128, 128, 3), np.uint8))
    cv2.imwrite(str(train / "vault/wide.jpg"),
                rng.integers(0, 255, (40, 900, 3), dtype=np.uint8))
    (train / "altar/broken.jpg").write_bytes(b"definitely not an image")

    from src.data import image_audit

    return {
        "train": image_audit.audit_images(train, verbose=False),
        "test": image_audit.audit_images(test, verbose=False),
    }


def test_audit_detects_corrupt_file(audited_images):
    from src.data.image_audit import quality_report

    counts = quality_report(audited_images["train"]).set_index("issue")["count"]
    assert counts["corrupt / unreadable files"] == 1


def test_audit_detects_duplicates_and_label_noise(audited_images):
    from src.data.image_audit import find_duplicates

    dupes = find_duplicates(audited_images["train"])
    assert len(dupes) == 2
    assert dupes["cross_class"].any(), (
        "the same image in two classes is label noise and must be surfaced"
    )


def test_audit_detects_blank_and_aspect_outliers(audited_images):
    from src.data.image_audit import quality_report

    counts = quality_report(audited_images["train"]).set_index("issue")["count"]
    assert counts["near-constant (blank) images"] == 1
    assert counts["aspect-ratio outliers"] == 1


def test_audit_detects_train_test_leakage(audited_images):
    """The single most important image check: a clean test score depends on it."""
    from src.data.image_audit import find_leakage

    leakage = find_leakage(audited_images["train"], audited_images["test"])
    assert len(leakage) == 1
    assert leakage.iloc[0]["test_file"] == "leak.jpg"


def test_per_class_stats_reports_duplicate_rate(audited_images):
    from src.data.image_audit import per_class_stats

    stats = per_class_stats(audited_images["train"])
    assert stats.loc["altar", "duplicate_rate_%"] > 0
    for col in ("n_images", "mean_brightness", "mean_contrast", "n_unique_images"):
        assert col in stats.columns


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
