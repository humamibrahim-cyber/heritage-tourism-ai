"""Training orchestration for Part 1.

Implements the experiment matrix the brief asks for:
  * task 8  - train WITHOUT augmentation, monitoring validation accuracy
  * task 10 - train WITH augmentation, monitoring validation accuracy
  * plus a backbone benchmark so the architecture choice is evidence-based
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..models.backbones import (
    build_classifier,
    compile_model,
    get_spec,
    unfreeze_top,
)
from .callbacks import merge_histories, standard_callbacks


def _artifact_dir(cfg=None) -> Path:
    from ..config import ARTIFACT_ROOT

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_ROOT


def benchmark_backbones(cfg, train_ds, val_ds, backbones=None, epochs=None):
    """Short frozen-base bake-off to justify the architecture choice (task 2).

    Each candidate trains for a handful of epochs with an identical head and
    identical data, so the comparison is apples-to-apples. Returns a DataFrame
    sorted by best validation accuracy.
    """
    import pandas as pd
    import tensorflow as tf

    backbones = backbones or cfg.benchmark_backbones
    epochs = epochs or cfg.benchmark_epochs
    rows = []

    for key in backbones:
        spec = get_spec(key)
        print(f"\n{'=' * 70}\nBenchmarking {spec.name}\n{'=' * 70}")
        tf.keras.backend.clear_session()

        model = build_classifier(cfg, backbone=key)
        compile_model(model, lr=cfg.head_lr)

        start = time.time()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=2,
        )
        elapsed = time.time() - start

        best_idx = int(max(
            range(len(history.history["val_accuracy"])),
            key=lambda i: history.history["val_accuracy"][i],
        ))
        rows.append(
            {
                "backbone": spec.name,
                "key": key,
                "params_M": spec.params_millions,
                "best_val_acc": round(history.history["val_accuracy"][best_idx], 4),
                "best_val_loss": round(history.history["val_loss"][best_idx], 4),
                "best_epoch": best_idx + 1,
                "seconds": round(elapsed, 1),
                "sec_per_epoch": round(elapsed / epochs, 1),
            }
        )
        del model

    df = pd.DataFrame(rows).sort_values("best_val_acc", ascending=False)
    out = _artifact_dir(cfg) / "backbone_benchmark.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved benchmark to {out}")
    return df.reset_index(drop=True)


def train_two_stage(
    cfg,
    train_ds,
    val_ds,
    backbone: str | None = None,
    augmentation=None,
    class_weight: dict | None = None,
    run_name: str = "run",
    verbose: int = 1,
):
    """Full transfer-learning schedule: frozen head, then partial fine-tune.

    Returns ``(model, merged_history, stage_histories)``.

    Stage A satisfies the brief's "freeze all convolutional layers"; Stage B is
    the fine-tuning refinement that lifts accuracy several points beyond what a
    purely frozen base can reach.
    """
    import tensorflow as tf

    key = backbone or cfg.backbone
    art = _artifact_dir(cfg)
    ckpt = art / f"{run_name}_best.keras"

    tf.keras.backend.clear_session()
    model = build_classifier(cfg, backbone=key, augmentation=augmentation)
    compile_model(model, lr=cfg.head_lr, label_smoothing=cfg.label_smoothing)

    print(f"\n{'=' * 70}\nSTAGE A - frozen convolutional base ({key})\n{'=' * 70}")
    hist_a = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.head_epochs,
        callbacks=standard_callbacks(cfg, ckpt),
        class_weight=class_weight,
        verbose=verbose,
    )

    print(f"\n{'=' * 70}\nSTAGE B - fine-tuning top {cfg.finetune_fraction:.0%}\n{'=' * 70}")
    unfreeze_top(model, cfg.finetune_fraction)
    # Recompile is mandatory: changing `trainable` has no effect until you do.
    compile_model(model, lr=cfg.finetune_lr, label_smoothing=cfg.label_smoothing)

    hist_b = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.head_epochs + cfg.finetune_epochs,
        initial_epoch=len(hist_a.history["loss"]),
        callbacks=standard_callbacks(cfg, ckpt),
        class_weight=class_weight,
        verbose=verbose,
    )

    merged = merge_histories(hist_a, hist_b)
    with open(art / f"{run_name}_history.json", "w") as fh:
        json.dump(merged, fh, indent=2)

    return model, merged, {"stage_a": hist_a.history, "stage_b": hist_b.history}


def run_augmentation_experiment(cfg, datasets_fn, class_weight=None, backbone=None):
    """Brief tasks 8 + 10: identical schedule with and without augmentation.

    ``datasets_fn(augment: bool)`` must return ``(train_ds, val_ds, test_ds)``.
    """
    from ..data.image_data import build_augmentation

    results = {}

    print("\n" + "#" * 70 + "\n# EXPERIMENT 1/2 - NO augmentation (task 8)\n" + "#" * 70)
    train_ds, val_ds, _ = datasets_fn(augment=False)
    model_plain, hist_plain, _ = train_two_stage(
        cfg, train_ds, val_ds, backbone=backbone,
        class_weight=class_weight, run_name="no_aug",
    )
    results["no_augmentation"] = {"model": model_plain, "history": hist_plain}

    print("\n" + "#" * 70 + "\n# EXPERIMENT 2/2 - WITH augmentation (task 10)\n" + "#" * 70)
    train_ds, val_ds, _ = datasets_fn(augment=False)  # augmentation lives in-graph
    model_aug, hist_aug, _ = train_two_stage(
        cfg, train_ds, val_ds, backbone=backbone,
        augmentation=build_augmentation(),
        class_weight=class_weight, run_name="with_aug",
    )
    results["with_augmentation"] = {"model": model_aug, "history": hist_aug}

    return results
