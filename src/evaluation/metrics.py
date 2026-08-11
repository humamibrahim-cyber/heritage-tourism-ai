"""Evaluation for Part 1 - beyond plain accuracy.

With a 4.7x class imbalance (column: 1,920 vs flying_buttress: 408), raw
accuracy flatters the model. Macro-averaged F1 and a per-class breakdown are
what actually show whether the rare classes are being learned.
"""
from __future__ import annotations

import numpy as np


def predict_dataset(model, dataset):
    """Return (y_true, y_pred, y_proba) for an unshuffled tf.data.Dataset."""
    import tensorflow as tf

    y_true, y_proba = [], []
    for batch_x, batch_y in dataset:
        probs = model.predict(batch_x, verbose=0)
        y_proba.append(probs)
        y_true.append(batch_y.numpy() if hasattr(batch_y, "numpy") else batch_y)

    y_proba = np.concatenate(y_proba, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    if y_true.ndim > 1 and y_true.shape[1] > 1:  # one-hot -> index
        y_true = y_true.argmax(axis=1)
    y_pred = y_proba.argmax(axis=1)
    return y_true, y_pred, y_proba


def classification_report_df(y_true, y_pred, class_names):
    """Per-class precision / recall / F1 as a tidy DataFrame."""
    import pandas as pd
    from sklearn.metrics import classification_report

    report = classification_report(
        y_true, y_pred, target_names=list(class_names), output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report).T
    df["support"] = df["support"].astype(int)
    return df.round(4)


def headline_metrics(y_true, y_pred, y_proba=None, class_names=None) -> dict:
    """The numbers to quote in the report."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
        top_k_accuracy_score,
    )

    out = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(
            f1_score(y_true, y_pred, average="weighted", zero_division=0), 4
        ),
        "cohen_kappa": round(cohen_kappa_score(y_true, y_pred), 4),
    }
    if y_proba is not None:
        n_classes = y_proba.shape[1]
        out["top_3_accuracy"] = round(
            top_k_accuracy_score(y_true, y_proba, k=3, labels=np.arange(n_classes)), 4
        )
    return out


def most_confused_pairs(y_true, y_pred, class_names, top_n: int = 8):
    """Which classes the model mixes up - the interesting part of the analysis."""
    import pandas as pd
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    rows = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                rows.append(
                    {
                        "true": true_name,
                        "predicted": pred_name,
                        "count": int(cm[i, j]),
                        "pct_of_true_class": round(100 * cm[i, j] / max(cm[i].sum(), 1), 1),
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values("count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def find_misclassified(y_true, y_pred, y_proba, top_n: int = 12):
    """Indices of the most confidently wrong predictions - useful for a figure."""
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        return np.array([], dtype=int)
    confidence = y_proba[wrong].max(axis=1)
    return wrong[np.argsort(-confidence)][:top_n]
