"""Shared plotting helpers.

One consistent visual language across both parts of the project: a single
categorical palette, no chart junk, direct labelling where it helps.
"""
from __future__ import annotations

import numpy as np

# Colour-blind-safe categorical palette (Okabe-Ito derived).
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]
ACCENT = "#4C72B0"
CONTRAST = "#DD8452"


def use_house_style():
    """Apply consistent matplotlib defaults. Call once per notebook."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.frameon": False,
            "font.size": 10,
        }
    )
    plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=PALETTE)


# --------------------------------------------------------------------------
# Part 1 figures
# --------------------------------------------------------------------------
def plot_training_curves(history, title="Training history", target: float | None = None):
    """Brief task 12 - accuracy and loss per epoch, to expose overfitting.

    The gap between the train and validation curves is the whole point of this
    figure, so it is shaded rather than left for the reader to eyeball.
    """
    import matplotlib.pyplot as plt

    hist = history.history if hasattr(history, "history") else history
    epochs = np.arange(1, len(hist["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    axes[0].plot(epochs, hist["accuracy"], color=ACCENT, lw=2, label="train")
    axes[0].plot(epochs, hist["val_accuracy"], color=CONTRAST, lw=2, label="validation")
    axes[0].fill_between(
        epochs, hist["accuracy"], hist["val_accuracy"],
        color="grey", alpha=0.12, label="generalisation gap",
    )
    if target:
        axes[0].axhline(target, ls="--", lw=1, color="#555")
        axes[0].text(
            epochs[-1], target, f" target {target:.0%}",
            va="bottom", ha="right", fontsize=9, color="#555",
        )
    best = int(np.argmax(hist["val_accuracy"]))
    axes[0].scatter([epochs[best]], [hist["val_accuracy"][best]], color=CONTRAST, zorder=5)
    axes[0].annotate(
        f"best {hist['val_accuracy'][best]:.3f} (ep {best + 1})",
        (epochs[best], hist["val_accuracy"][best]),
        textcoords="offset points", xytext=(6, -12), fontsize=9, color=CONTRAST,
    )
    axes[0].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    axes[0].legend(loc="lower right")

    axes[1].plot(epochs, hist["loss"], color=ACCENT, lw=2, label="train")
    axes[1].plot(epochs, hist["val_loss"], color=CONTRAST, lw=2, label="validation")
    axes[1].set(xlabel="epoch", ylabel="loss", title="Loss")
    axes[1].legend(loc="upper right")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def compare_histories(histories: dict, metric: str = "val_accuracy", title=None):
    """Overlay several runs - e.g. augmented vs non-augmented (tasks 8 vs 10)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for (label, hist), colour in zip(histories.items(), PALETTE):
        values = hist.history[metric] if hasattr(hist, "history") else hist[metric]
        epochs = np.arange(1, len(values) + 1)
        ax.plot(epochs, values, lw=2, color=colour, label=label)
        ax.scatter([epochs[int(np.argmax(values))]], [max(values)], color=colour, zorder=5)
        ax.annotate(
            f"{max(values):.3f}",
            (epochs[int(np.argmax(values))], max(values)),
            textcoords="offset points", xytext=(6, 4), fontsize=9, color=colour,
        )
    ax.set(xlabel="epoch", ylabel=metric.replace("_", " "),
           title=title or f"{metric.replace('_', ' ')} by run")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names, normalise=True, title=None):
    """Row-normalised confusion matrix - reads as per-class recall."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    display = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalise else cm

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(display, cmap="Blues", vmin=0, vmax=display.max())
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set(xlabel="predicted", ylabel="actual",
           title=title or ("Confusion matrix (row-normalised = recall)" if normalise
                           else "Confusion matrix"))
    ax.grid(False)

    threshold = display.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = display[i, j]
            if value > 0.005:
                ax.text(
                    j, i, f"{value:.2f}" if normalise else f"{int(value)}",
                    ha="center", va="center", fontsize=8,
                    color="white" if value > threshold else "#333",
                )
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_class_distribution(summary_df, title="Images per class"):
    """Horizontal bars - class names are long, so vertical bars would collide."""
    import matplotlib.pyplot as plt

    df = summary_df.sort_values("train")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df["class"], df["train"], color=ACCENT, label="train")
    ax.barh(df["class"], df["test"], left=df["train"], color=CONTRAST, label="test")
    for y, (tr, te) in enumerate(zip(df["train"], df["test"])):
        ax.text(tr + te + 25, y, f"{tr + te:,}", va="center", fontsize=9, color="#444")
    ax.set(xlabel="images", title=title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Part 2 figures
# --------------------------------------------------------------------------
def plot_bar(labels, values, title="", xlabel="", ylabel="", horizontal=True,
             value_fmt="{:.2f}", color=None, figsize=(9, 5)):
    """General labelled bar chart used throughout the EDA."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    colour = color or ACCENT
    if horizontal:
        ax.barh(labels, values, color=colour)
        for y, v in enumerate(values):
            ax.text(v, y, " " + value_fmt.format(v), va="center", fontsize=9, color="#444")
        ax.set(xlabel=ylabel or xlabel, title=title)
    else:
        ax.bar(labels, values, color=colour)
        for x, v in enumerate(values):
            ax.text(x, v, value_fmt.format(v), ha="center", va="bottom",
                    fontsize=9, color="#444")
        ax.set(ylabel=ylabel, xlabel=xlabel, title=title)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig
