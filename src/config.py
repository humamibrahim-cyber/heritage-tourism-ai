"""Central configuration for the Preserving Heritage capstone.

Every tunable lives here so the notebooks stay readable and the two parts of
the project cannot drift out of sync. Paths auto-detect whether the code is
running in Google Colab (dataset on Drive) or locally.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------
# Environment detection
# --------------------------------------------------------------------------
def in_colab() -> bool:
    """True when executing inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def default_data_root() -> Path:
    """Where the dataset is expected to live.

    Override with the HERITAGE_DATA_ROOT environment variable, e.g.
        os.environ["HERITAGE_DATA_ROOT"] = "/content/data"
    """
    env = os.environ.get("HERITAGE_DATA_ROOT")
    if env:
        return Path(env)
    if in_colab():
        return Path("/content/data")
    return Path(__file__).resolve().parents[1] / "data"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = default_data_root()
ARTIFACT_ROOT = Path(os.environ.get("HERITAGE_ARTIFACTS", PROJECT_ROOT / "artifacts"))


# --------------------------------------------------------------------------
# Part 1 - image classification
# --------------------------------------------------------------------------
# The 10 classes present in the archive. NOTE: a `portal` folder appears in the
# zip's __MACOSX metadata but contains no real images, so this is a 10-class
# problem. Sorted order matches keras image_dataset_from_directory.
CLASS_NAMES: tuple[str, ...] = (
    "altar",
    "apse",
    "bell_tower",
    "column",
    "dome(inner)",
    "dome(outer)",
    "flying_buttress",
    "gargoyle",
    "stained_glass",
    "vault",
)

# Image counts observed in the supplied archive, used for sanity assertions.
EXPECTED_TRAIN_COUNTS = {
    "altar": 830,
    "apse": 515,
    "bell_tower": 1060,
    "column": 1920,
    "dome(inner)": 617,
    "dome(outer)": 1178,
    "flying_buttress": 408,
    "gargoyle": 1572,
    "stained_glass": 1034,
    "vault": 1111,
}
EXPECTED_TEST_COUNTS = {
    "altar": 141,
    "apse": 58,
    "bell_tower": 172,
    "column": 211,
    "dome(inner)": 87,
    "dome(outer)": 169,
    "flying_buttress": 79,
    "gargoyle": 241,
    "stained_glass": 164,
    "vault": 165,
}


@dataclass
class ImageConfig:
    """Hyperparameters for Part 1."""

    # Data ---------------------------------------------------------------
    train_dir: Path = field(
        default_factory=lambda: DATA_ROOT
        / "dataset_hist_structures"
        / "Stuctures_Dataset"  # sic - the typo is in the supplied archive
    )
    test_dir: Path = field(
        default_factory=lambda: DATA_ROOT
        / "dataset_hist_structures"
        / "Dataset_test"
        / "Dataset_test_original_1478"
    )
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 64          # Colab Pro (A100/L4/V100). Drop to 32 on free T4.
    validation_split: float = 0.15
    seed: int = 42

    # Model --------------------------------------------------------------
    backbone: str = "efficientnetv2b0"
    dense_units: int = 256
    dropout_rate: float = 0.4
    # 0.0 keeps the sparse (integer-label) loss path. Raising this switches to
    # one-hot CategoricalCrossentropy, which also requires rebuilding the
    # datasets with label_mode="categorical" - see models.backbones.compile_model.
    label_smoothing: float = 0.0

    # Training -----------------------------------------------------------
    head_epochs: int = 20          # stage A: frozen convolutional base
    finetune_epochs: int = 15      # stage B: partial unfreeze
    head_lr: float = 1e-3
    finetune_lr: float = 1e-5
    finetune_fraction: float = 0.30   # unfreeze top 30% of backbone layers
    target_val_accuracy: float = 0.93  # custom callback stops training here
    use_class_weights: bool = True
    mixed_precision: bool = True

    # Benchmark ----------------------------------------------------------
    benchmark_backbones: tuple[str, ...] = (
        "efficientnetv2b0",
        "resnet50v2",
        "mobilenetv2",
    )
    benchmark_epochs: int = 8

    @property
    def num_classes(self) -> int:
        return len(CLASS_NAMES)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (*self.image_size, 3)


# --------------------------------------------------------------------------
# Part 2 - recommendation engine
# --------------------------------------------------------------------------
@dataclass
class RecoConfig:
    """Hyperparameters for Part 2."""

    # Data ---------------------------------------------------------------
    tourism_dir: Path = field(default_factory=lambda: DATA_ROOT / "tourism")
    places_file: str = "tourism_with_id.xlsx"
    ratings_file: str = "tourism_rating.csv"
    users_file: str = "user.csv"

    # Split --------------------------------------------------------------
    test_size: float = 0.2
    seed: int = 42

    # Keras matrix factorisation -----------------------------------------
    # Deliberately small: 300 users x 437 places x ~10k ratings is a tiny,
    # ~92% sparse matrix. A large embedding overfits within a few epochs.
    embedding_dim: int = 32
    l2_reg: float = 1e-6
    mf_epochs: int = 60
    mf_batch_size: int = 128
    mf_lr: float = 1e-3
    early_stopping_patience: int = 8

    # Item-item collaborative filtering ----------------------------------
    min_ratings_per_place: int = 3   # places below this are unreliable neighbours
    similarity: str = "cosine"

    # Evaluation ---------------------------------------------------------
    top_k: int = 10
    relevance_threshold: float = 4.0  # a rating >= 4 counts as "relevant"


IMAGE_CFG = ImageConfig()
RECO_CFG = RecoConfig()


def describe() -> str:
    """Human-readable summary, handy as the first cell of a notebook."""
    return (
        f"Project root : {PROJECT_ROOT}\n"
        f"Data root    : {DATA_ROOT}\n"
        f"Artifacts    : {ARTIFACT_ROOT}\n"
        f"Colab        : {in_colab()}\n"
        f"Classes      : {len(CLASS_NAMES)} -> {', '.join(CLASS_NAMES)}\n"
    )
