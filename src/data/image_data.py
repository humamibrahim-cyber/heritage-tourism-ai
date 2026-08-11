"""Image dataset pipeline for Part 1.

Covers project tasks 1 and 7 of the brief:
  * task 1  - plot 8-10 sample images per class
  * task 7  - set up train/test directories and report sample counts per class
"""
from __future__ import annotations

import collections
import math
from pathlib import Path
from typing import Iterable

import numpy as np

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


# --------------------------------------------------------------------------
# Inventory helpers (pure python - no TensorFlow needed)
# --------------------------------------------------------------------------
def count_images_per_class(directory: str | Path) -> dict[str, int]:
    """Return {class_name: n_images} for a directory of class sub-folders.

    Hidden files and macOS metadata (``.DS_Store``, ``__MACOSX``) are ignored,
    which matters here because the supplied archive is riddled with them.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(
            f"{directory} does not exist. Check DATA_ROOT / that the archive was "
            f"extracted. Expected class sub-folders directly inside this path."
        )

    counts: dict[str, int] = {}
    for child in sorted(directory.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "__")):
            continue
        counts[child.name] = sum(
            1
            for f in child.iterdir()
            if f.is_file()
            and f.suffix.lower() in IMG_EXTENSIONS
            and not f.name.startswith(".")
        )
    return counts


def dataset_summary(train_dir, test_dir) -> "pd.DataFrame":  # noqa: F821
    """Side-by-side train/test counts with imbalance ratio. Brief task 7."""
    import pandas as pd

    train = count_images_per_class(train_dir)
    test = count_images_per_class(test_dir)
    classes = sorted(set(train) | set(test))

    df = pd.DataFrame(
        {
            "class": classes,
            "train": [train.get(c, 0) for c in classes],
            "test": [test.get(c, 0) for c in classes],
        }
    )
    df["total"] = df["train"] + df["test"]
    df["train_share_%"] = (100 * df["train"] / df["train"].sum()).round(2)
    # How over/under-represented a class is versus a perfectly balanced split.
    df["imbalance_x"] = (df["train"] / (df["train"].sum() / len(df))).round(2)
    return df.sort_values("train", ascending=False).reset_index(drop=True)


def list_class_images(directory, class_name: str, limit: int | None = None) -> list[Path]:
    """Deterministic list of image paths for one class."""
    folder = Path(directory) / class_name
    files = sorted(
        f
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS and not f.name.startswith(".")
    )
    return files[:limit] if limit else files


# --------------------------------------------------------------------------
# Task 1 - sample image grid
# --------------------------------------------------------------------------
def plot_class_samples(
    directory,
    class_name: str,
    n: int = 10,
    cols: int = 5,
    figsize_scale: float = 2.2,
    random_state: int | None = 42,
):
    """Plot ``n`` sample images for one class using OpenCV, as the brief hints.

    OpenCV loads BGR; we convert to RGB so matplotlib shows true colours -
    forgetting this is the classic reason sample grids look blue-tinted.
    """
    import cv2
    import matplotlib.pyplot as plt

    files = list_class_images(directory, class_name)
    if not files:
        raise ValueError(f"No images found for class '{class_name}' in {directory}")

    rng = np.random.default_rng(random_state)
    picks = rng.choice(len(files), size=min(n, len(files)), replace=False)
    picks = sorted(picks)

    rows = math.ceil(len(picks) / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * figsize_scale, rows * figsize_scale)
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, picks):
        bgr = cv2.imread(str(files[idx]))
        if bgr is None:  # unreadable/corrupt file
            ax.text(0.5, 0.5, "unreadable", ha="center", va="center")
            ax.axis("off")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(f"{rgb.shape[1]}x{rgb.shape[0]}", fontsize=8)
        ax.axis("off")

    for ax in axes[len(picks):]:
        ax.axis("off")

    fig.suptitle(f"{class_name}  -  {len(files)} images available", fontsize=13)
    fig.tight_layout()
    return fig


def plot_all_class_samples(directory, class_names: Iterable[str], n: int = 8):
    """Convenience wrapper: one sample grid per class."""
    return [plot_class_samples(directory, c, n=n) for c in class_names]


def image_size_report(directory, class_names: Iterable[str], per_class: int = 50):
    """Sample image dimensions so we can justify the model input size."""
    import cv2
    import pandas as pd

    rows = []
    for cls in class_names:
        for path in list_class_images(directory, cls, limit=per_class):
            img = cv2.imread(str(path))
            if img is not None:
                rows.append({"class": cls, "height": img.shape[0], "width": img.shape[1]})
    df = pd.DataFrame(rows)
    return df.describe()[["height", "width"]], collections.Counter(
        zip(df["height"], df["width"])
    ).most_common(5)


# --------------------------------------------------------------------------
# tf.data pipelines
# --------------------------------------------------------------------------
def build_datasets(cfg, augment: bool = False, verbose: bool = True):
    """Build (train, val, test) ``tf.data.Dataset`` objects.

    The validation split is carved out of the *training* directory only; the
    1,487 supplied test images are never seen during training so the final
    number is honest.

    Augmentation is applied to the training set only - never to val/test,
    which would make the comparison between the augmented and non-augmented
    runs meaningless.
    """
    import tensorflow as tf

    common = dict(
        labels="inferred",
        label_mode="int",
        class_names=list(cfg_class_names(cfg)),
        image_size=cfg.image_size,
        interpolation="bilinear",
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.train_dir,
        validation_split=cfg.validation_split,
        subset="training",
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        shuffle=True,
        **common,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.train_dir,
        validation_split=cfg.validation_split,
        subset="validation",
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        shuffle=False,
        **common,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.test_dir,
        batch_size=cfg.batch_size,
        shuffle=False,
        **common,
    )

    if verbose:
        print(f"classes      : {train_ds.class_names}")
        print(f"train batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
        print(f"val batches  : {tf.data.experimental.cardinality(val_ds).numpy()}")
        print(f"test batches : {tf.data.experimental.cardinality(test_ds).numpy()}")

    autotune = tf.data.AUTOTUNE
    if augment:
        aug = build_augmentation()
        train_ds = train_ds.map(
            lambda x, y: (aug(x, training=True), y), num_parallel_calls=autotune
        )

    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds


def cfg_class_names(cfg):
    from ..config import CLASS_NAMES

    return getattr(cfg, "class_names", CLASS_NAMES)


def build_augmentation():
    """Augmentation pipeline for brief task 10.

    Kept geometrically mild on purpose: these are architectural elements, so a
    hard vertical flip or large rotation would produce images that never occur
    in reality and hurt more than they help.
    """
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.10),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(0.10, 0.10),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def compute_class_weights(cfg):
    """Inverse-frequency class weights to offset the 4.7x imbalance."""
    counts = count_images_per_class(cfg.train_dir)
    names = list(cfg_class_names(cfg))
    total = sum(counts.get(c, 0) for c in names)
    n_classes = len(names)
    return {
        i: total / (n_classes * max(counts.get(c, 0), 1)) for i, c in enumerate(names)
    }
