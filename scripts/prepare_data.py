#!/usr/bin/env python3
"""Extract and lay out the capstone datasets, then verify the result.

Usage:
    python scripts/prepare_data.py --zip "dataset_hist_structures 2.zip" \
                                   --tourism-dir /path/to/Part\\ 2 \
                                   --out data

Handles the three things that reliably go wrong by hand:
  * the archive nests everything one folder deep
  * macOS metadata (__MACOSX, .DS_Store) breaks image loaders
  * the train folder is spelled "Stuctures_Dataset" (missing an 'r')
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLASS_NAMES, EXPECTED_TEST_COUNTS, EXPECTED_TRAIN_COUNTS  # noqa: E402
from src.data.image_data import count_images_per_class  # noqa: E402


def extract(zip_path: Path, out: Path) -> None:
    print(f"Extracting {zip_path.name} -> {out} ...")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if "__MACOSX" not in m and ".DS_Store" not in m]
        zf.extractall(out, members=members)
    print(f"  {len(members)} entries extracted")


def flatten(out: Path) -> None:
    """The archive wraps everything in 'dataset_hist_structures 2/'."""
    nested = out / "dataset_hist_structures 2" / "dataset_hist_structures"
    target = out / "dataset_hist_structures"
    if nested.exists() and not target.exists():
        shutil.move(str(nested), str(target))
        shutil.rmtree(out / "dataset_hist_structures 2", ignore_errors=True)
        print("  flattened nested folder")


def scrub(out: Path) -> None:
    removed = 0
    for junk in list(out.rglob("__MACOSX")):
        shutil.rmtree(junk, ignore_errors=True)
        removed += 1
    for junk in list(out.rglob(".DS_Store")):
        junk.unlink(missing_ok=True)
        removed += 1
    print(f"  removed {removed} macOS metadata entries")


def copy_tourism(src: Path, out: Path) -> None:
    dest = out / "tourism"
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pattern in ("user.csv", "tourism_rating.csv", "tourism_with_id.*"):
        for f in src.glob(pattern):
            shutil.copy(f, dest / f.name)
            copied += 1
            print(f"  copied {f.name}")
    if copied < 3:
        print(f"  WARNING: only {copied} of 3 expected Part 2 files found in {src}")


def verify(out: Path) -> bool:
    """Compare what landed on disk against the counts we expect."""
    train = out / "dataset_hist_structures" / "Stuctures_Dataset"
    test = out / "dataset_hist_structures" / "Dataset_test" / "Dataset_test_original_1478"

    ok = True
    count_mismatches = 0
    for label, directory, expected in [
        ("train", train, EXPECTED_TRAIN_COUNTS),
        ("test", test, EXPECTED_TEST_COUNTS),
    ]:
        if not directory.exists():
            print(f"  MISSING {label}: {directory}")
            ok = False
            continue

        actual = count_images_per_class(directory)
        print(f"\n  {label}: {sum(actual.values()):,} images in {len(actual)} classes")

        missing = set(CLASS_NAMES) - set(actual)
        extra = set(actual) - set(CLASS_NAMES)
        if missing:
            print(f"    MISSING classes: {sorted(missing)}")
            ok = False
        if extra:
            print(f"    UNEXPECTED classes: {sorted(extra)}")

        for cls, want in expected.items():
            got = actual.get(cls, 0)
            if got != want:
                count_mismatches += 1
                print(f"    {cls}: {got} images (expected {want})")

    if count_mismatches:
        # Not fatal - the layout is correct, but the contents differ from the
        # archive this project was built against. Worth knowing before training.
        print(f"\n  NOTE: {count_mismatches} class counts differ from the reference "
              f"archive. The folder structure is valid; verify you have the right zip.")

    tourism = out / "tourism"
    if tourism.exists():
        files = sorted(f.name for f in tourism.iterdir() if f.is_file())
        print(f"\n  tourism: {files}")
    else:
        print("\n  tourism folder not created (pass --tourism-dir)")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, help="path to 'dataset_hist_structures 2.zip'")
    parser.add_argument("--tourism-dir", type=Path, help="folder holding the Part 2 files")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if not args.verify_only:
        if args.zip:
            if not args.zip.exists():
                print(f"ERROR: {args.zip} not found")
                return 1
            extract(args.zip, args.out)
            flatten(args.out)
            scrub(args.out)
        if args.tourism_dir:
            copy_tourism(args.tourism_dir, args.out)

    print("\nVerification")
    ok = verify(args.out)
    print("\n" + ("All checks passed." if ok else "Problems found - see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
