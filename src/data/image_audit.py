"""Image data quality audit for Part 1 — run before training, not after.

Image datasets have their own failure modes, and none of them show up in a
`describe()` table:

* **corrupt / truncated files** - crash training mid-epoch, often hours in
* **exact duplicates** - the same photograph in both train and test leaks the
  test set and inflates your reported accuracy
* **near-constant images** - all-black or all-white files carry no signal
* **dimension outliers** - a wildly different aspect ratio distorts badly when
  resized to a square input
* **greyscale files in an RGB pipeline** - silently broadcast, changing colour
  statistics for that class

The duplicate check across splits is the one that matters most: it is the
difference between a trustworthy test score and a meaningless one.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def _iter_images(directory: Path):
    for class_dir in sorted(directory.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith((".", "__")):
            continue
        for path in sorted(class_dir.iterdir()):
            if (
                path.is_file()
                and path.suffix.lower() in IMG_EXTENSIONS
                and not path.name.startswith(".")
            ):
                yield class_dir.name, path


def file_hash(path: Path, chunk_size: int = 65536) -> str:
    """MD5 of the raw bytes. Detects byte-identical duplicates cheaply."""
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_images(directory, sample_pixels: bool = True,
                 verbose: bool = True) -> pd.DataFrame:
    """Per-image metadata for a directory of class sub-folders.

    Dimensions come from PIL's lazy header read, so this stays fast over
    thousands of files. Pixel statistics require a decode and are sampled at a
    reduced size.
    """
    from PIL import Image

    directory = Path(directory)
    records = []
    for class_name, path in _iter_images(directory):
        record = {
            "class": class_name,
            "path": str(path),
            "filename": path.name,
            "bytes": path.stat().st_size,
            "corrupt": False,
        }
        try:
            with Image.open(path) as img:
                record["width"], record["height"] = img.size
                record["mode"] = img.mode
                record["format"] = img.format
                if sample_pixels:
                    small = img.convert("RGB").resize((32, 32))
                    arr = np.asarray(small, dtype=np.float32)
                    record["mean_pixel"] = float(arr.mean())
                    record["std_pixel"] = float(arr.std())
            record["hash"] = file_hash(path)
        except Exception as exc:  # unreadable or truncated
            record["corrupt"] = True
            record["error"] = str(exc)[:120]
        records.append(record)

    df = pd.DataFrame(records)
    if not df.empty and "width" in df.columns:
        df["aspect_ratio"] = (df["width"] / df["height"]).round(3)
        df["pixels"] = df["width"] * df["height"]

    if verbose:
        print(f"Audited {len(df)} images across {df['class'].nunique()} classes "
              f"in {directory.name}")
    return df


def quality_report(audit: pd.DataFrame) -> pd.DataFrame:
    """Roll the per-image audit into a table of issues to act on."""
    issues = []

    n_corrupt = int(audit["corrupt"].sum())
    issues.append({
        "issue": "corrupt / unreadable files",
        "count": n_corrupt,
        "severity": "CRITICAL" if n_corrupt else "ok",
        "action": "delete before training - they crash fit() mid-epoch" if n_corrupt else "none",
    })

    valid = audit[~audit["corrupt"]]
    if valid.empty:
        return pd.DataFrame(issues)

    dup_groups = valid.groupby("hash").size()
    n_dup_files = int((dup_groups[dup_groups > 1]).sum() - (dup_groups > 1).sum())
    issues.append({
        "issue": "exact duplicate images (within split)",
        "count": n_dup_files,
        "severity": "WARNING" if n_dup_files else "ok",
        "action": "inflates effective class counts; consider de-duplicating"
                  if n_dup_files else "none",
    })

    if "std_pixel" in valid.columns:
        n_blank = int((valid["std_pixel"] < 1.0).sum())
        issues.append({
            "issue": "near-constant (blank) images",
            "count": n_blank,
            "severity": "WARNING" if n_blank else "ok",
            "action": "carry no signal - inspect and remove" if n_blank else "none",
        })

    if "mode" in valid.columns:
        n_non_rgb = int((valid["mode"] != "RGB").sum())
        issues.append({
            "issue": "non-RGB images",
            "count": n_non_rgb,
            "severity": "INFO" if n_non_rgb else "ok",
            "action": "converted to RGB on load; check colour stats per class"
                      if n_non_rgb else "none",
        })

    if "aspect_ratio" in valid.columns:
        from .eda import detect_outliers

        # detect_outliers handles the degenerate case where most images share
        # one exact aspect ratio, which makes the MAD zero.
        n_ar = int(detect_outliers(valid["aspect_ratio"], method="mad", threshold=5).sum())
        issues.append({
            "issue": "aspect-ratio outliers",
            "count": n_ar,
            "severity": "INFO" if n_ar else "ok",
            "action": "distort badly when resized to a square input" if n_ar else "none",
        })

        n_sizes = int(valid[["width", "height"]].drop_duplicates().shape[0])
        issues.append({
            "issue": "distinct image dimensions",
            "count": n_sizes,
            "severity": "INFO",
            "action": "all resized to the model input size on load",
        })

    return pd.DataFrame(issues)


def find_duplicates(audit: pd.DataFrame, min_group: int = 2) -> pd.DataFrame:
    """Groups of byte-identical images, with the classes each copy sits in.

    Duplicates spanning two *different* classes are label noise: the same
    picture cannot be both an altar and an apse.
    """
    valid = audit[~audit["corrupt"]]
    groups = valid.groupby("hash")

    rows = []
    for digest, group in groups:
        if len(group) < min_group:
            continue
        classes = sorted(group["class"].unique())
        rows.append({
            "hash": digest[:12],
            "n_copies": len(group),
            "classes": ", ".join(classes),
            "cross_class": len(classes) > 1,
            "files": ", ".join(group["filename"].head(3)),
        })
    if not rows:
        return pd.DataFrame(columns=["hash", "n_copies", "classes", "cross_class", "files"])
    return (
        pd.DataFrame(rows)
        .sort_values(["cross_class", "n_copies"], ascending=[False, False])
        .reset_index(drop=True)
    )


def find_leakage(train_audit: pd.DataFrame, test_audit: pd.DataFrame) -> pd.DataFrame:
    """Images present in BOTH train and test — the check that matters most.

    Any overlap means the reported test accuracy is partly memorisation, and the
    number cannot be quoted as an estimate of field performance.
    """
    train_hashes = set(train_audit.loc[~train_audit["corrupt"], "hash"])
    test_valid = test_audit[~test_audit["corrupt"]]
    overlap = test_valid[test_valid["hash"].isin(train_hashes)]

    if overlap.empty:
        return pd.DataFrame(columns=["hash", "test_file", "test_class", "train_class"])

    train_lookup = (
        train_audit[~train_audit["corrupt"]]
        .drop_duplicates("hash")
        .set_index("hash")["class"]
    )
    return pd.DataFrame({
        "hash": overlap["hash"].str[:12].values,
        "test_file": overlap["filename"].values,
        "test_class": overlap["class"].values,
        "train_class": train_lookup.loc[overlap["hash"]].values,
    }).reset_index(drop=True)


def per_class_stats(audit: pd.DataFrame) -> pd.DataFrame:
    """Class-level descriptive statistics — the Part 1 equivalent of describe().

    Systematically different brightness or file size per class is worth knowing:
    if one class is consistently darker, the model may be learning exposure
    rather than architecture.
    """
    valid = audit[~audit["corrupt"]]
    agg = {
        "n_images": ("filename", "size"),
        "mean_bytes": ("bytes", "mean"),
        "mean_width": ("width", "mean"),
        "mean_height": ("height", "mean"),
    }
    if "mean_pixel" in valid.columns:
        agg["mean_brightness"] = ("mean_pixel", "mean")
        agg["mean_contrast"] = ("std_pixel", "mean")

    stats = valid.groupby("class").agg(**agg).round(2)
    stats["n_unique_images"] = valid.groupby("class")["hash"].nunique()
    stats["duplicate_rate_%"] = (
        100 * (1 - stats["n_unique_images"] / stats["n_images"])
    ).round(2)
    return stats.sort_values("n_images", ascending=False)


def summarise(train_dir, test_dir, sample_pixels: bool = True) -> dict:
    """Full audit of both splits, including the cross-split leakage check."""
    print("Auditing train split ...")
    train_audit = audit_images(train_dir, sample_pixels=sample_pixels)
    print("Auditing test split ...")
    test_audit = audit_images(test_dir, sample_pixels=sample_pixels)

    leakage = find_leakage(train_audit, test_audit)

    print("\n=== TRAIN quality ===")
    train_report = quality_report(train_audit)
    print(train_report.to_string(index=False))

    print("\n=== TEST quality ===")
    test_report = quality_report(test_audit)
    print(test_report.to_string(index=False))

    print(f"\n=== TRAIN/TEST LEAKAGE: {len(leakage)} images appear in both splits ===")
    if len(leakage):
        print("The held-out test score is NOT a clean estimate. Remove these from "
              "train (never from test) and re-evaluate.")
        print(leakage.head(10).to_string(index=False))
    else:
        print("None. The test set is genuinely held out.")

    return {
        "train_audit": train_audit,
        "test_audit": test_audit,
        "train_report": train_report,
        "test_report": test_report,
        "leakage": leakage,
        "train_per_class": per_class_stats(train_audit),
        "test_per_class": per_class_stats(test_audit),
    }
