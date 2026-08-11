"""Training callbacks for Part 1.

Brief task 6: "Define your callback class to stop the training once validation
accuracy reaches a certain number of your choice."
"""
from __future__ import annotations

from pathlib import Path


def make_accuracy_threshold_callback(target: float = 0.93, monitor: str = "val_accuracy"):
    """Return a Callback that halts training at a target validation accuracy.

    Written as a factory so the class is only defined once TensorFlow is
    importable - keeps the module importable in a plain-python test run.
    """
    import tensorflow as tf

    class StopAtAccuracy(tf.keras.callbacks.Callback):
        """Brief task 6 - custom early stop on a validation accuracy target."""

        def __init__(self, target: float, monitor: str):
            super().__init__()
            self.target = target
            self.monitor = monitor
            self.stopped_epoch: int | None = None

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            current = logs.get(self.monitor)
            if current is None:
                # Fail loudly rather than silently never triggering.
                available = ", ".join(sorted(logs))
                print(
                    f"[StopAtAccuracy] '{self.monitor}' not in logs "
                    f"(available: {available}) - callback inactive this epoch."
                )
                return
            if current >= self.target:
                self.stopped_epoch = epoch + 1
                self.model.stop_training = True
                print(
                    f"\n[StopAtAccuracy] {self.monitor}={current:.4f} reached the "
                    f"{self.target:.2%} target at epoch {epoch + 1} - stopping."
                )

    return StopAtAccuracy(target, monitor)


def standard_callbacks(
    cfg,
    checkpoint_path: str | Path,
    include_threshold: bool = True,
    patience: int = 6,
):
    """Threshold-stop + checkpoint + LR schedule + safety early stop."""
    import tensorflow as tf

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    if include_threshold:
        callbacks.insert(
            0, make_accuracy_threshold_callback(cfg.target_val_accuracy)
        )
    return callbacks


def merge_histories(*histories):
    """Concatenate Keras History objects so stage A + stage B plot as one curve."""
    merged: dict[str, list] = {}
    for h in histories:
        hist = h.history if hasattr(h, "history") else h
        for key, values in hist.items():
            merged.setdefault(key, []).extend(values)
    return merged
