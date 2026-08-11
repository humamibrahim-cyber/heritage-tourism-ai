# CLAUDE.md

Context for Claude Code working in this repository.

## What this is

AIML Capstone 2 — "Preserving Heritage: Enhancing Tourism with AI". Two
deliverables for a government tourism agency:

- **Part 1** (`notebooks/01_image_classification.ipynb`) — TensorFlow CNN
  classifying photographs of historical structures into 10 categories.
- **Part 2** (`notebooks/02_recommendation_engine.ipynb`) — EDA plus a
  collaborative filtering recommender for Indonesian tourist destinations.

`docs/REQUIREMENTS_MAPPING.md` maps every numbered task in the assignment brief
to the code that satisfies it. **Read it before changing anything** — a change
that breaks a mapped requirement breaks the submission.

## Architecture

Notebooks orchestrate and narrate; all logic lives in `src/` so it is testable.
When adding functionality, put it in `src/` and call it from the notebook —
do not inline substantial logic in a cell.

```
src/config.py              ImageConfig / RecoConfig dataclasses — all tunables
src/data/image_data.py     tf.data pipeline, OpenCV sample grids, class weights
src/data/tourism_data.py   loading with schema validation, cleaning, merging
src/models/backbones.py    BackboneSpec registry, transfer-learning assembly
src/models/item_cf.py      item-item cosine CF, popularity baseline
src/models/recommender_net.py  Keras embedding matrix factorisation
src/training/callbacks.py  custom accuracy-threshold callback (brief task 6)
src/training/train_classifier.py  two-stage schedule, backbone benchmark
src/evaluation/metrics.py  classification metrics, error analysis
src/evaluation/ranking.py  RMSE, Precision/Recall/NDCG@K, coverage
src/viz/plots.py           shared matplotlib style — use it, do not restyle ad hoc
```

## Data quirks — these are real, not bugs to "fix"

1. **`Stuctures_Dataset` is misspelled** in the supplied archive (missing the
   `r`). Do not "correct" the path.
2. **10 classes, not 11.** A `portal` folder appears in the zip's `__MACOSX`
   metadata but contains no images. Anything that assumes 11 classes is wrong.
3. **Classes are imbalanced 4.7:1** — `column` 1,920 vs `flying_buttress` 408.
   Always report macro-F1 / balanced accuracy alongside accuracy.
4. **`tourism_with_id` is `.xlsx`**, though the brief calls it `.csv`.
   `load_raw` accepts either.
5. **The places file ships two trailing unnamed columns.** `_normalise_columns`
   drops them.
6. **`time_minutes` is ~50% null.** Leave it as `NaN`. Do not mean-impute —
   that invents tour durations and corrupts any figure derived from them.
7. **The rating matrix is 92.7% sparse** (300 users × 437 places, 10k ratings,
   9,597 after removing 403 duplicate user–place pairs). Do not scale up the
   embedding dimension "to improve results" — it will memorise.
8. **`Place_Ratings` contains no preference signal — verified, not suspected.**
   Four independent checks fail: near-uniform distribution (mean 3.07, skew
   −0.05), no significant between-place ANOVA (p=0.13), split-half reliability
   of place means r=0.005 against a shuffled null of 0.001±0.049, and zero
   correlation with the independent listed `Rating` (r=0.010, p=0.83). Random,
   popularity and item-CF recommenders all score NDCG@10 ≈ 0.016–0.019.

   **This means no model can win here, and that is the correct finding.** Do not
   "fix" it by tuning hyperparameters, changing the split, evaluating on the
   training set, or quietly reporting RMSE alone (RMSE ≈ 1.41 is just the
   predict-the-mean floor and looks deceptively fine). Run
   `src/evaluation/signal.py::diagnose` and report the verdict.

   The `Rating` column in `tourism_with_id.xlsx` *is* genuine — only the
   user-level ratings are synthetic. Content-based work using Category / City /
   Description / Rating is therefore still meaningful.

## Gotchas that have already bitten this codebase

- **Per-backbone preprocessing.** EfficientNet/EfficientNetV2 normalise inside
  the graph and expect raw `[0, 255]`; ResNet50V2 and MobileNetV2 need their own
  `preprocess_input`. Encoded in `BackboneSpec.preprocess`. Getting it wrong
  silently costs several accuracy points — the model still trains, just worse.
- **Recompile after changing `trainable`.** Keras ignores the change otherwise,
  so "fine-tuning" would train nothing.
- **Keep BatchNorm frozen while fine-tuning.** `unfreeze_top` does this.
  Unfreezing it destroys pre-trained statistics on small batches.
- **Unnamed pandas `Index` loses the index name.** `df.loc[unnamed_index]`
  returns a frame whose index name is `None`, so `reset_index()` yields a column
  called `index` and `place_id` vanishes — silently, because the column filter
  then drops it. This caused a real bug; `tests::test_all_recommenders_return_place_id`
  guards it. Set `out.index.name` explicitly before `reset_index()`.
- **Keras 3 `Callback.model` is read-only.** Use `set_model()` in tests.
- **Final Dense must be `dtype="float32"`** when mixed precision is on.

## Conventions

- Python ≥ 3.10, type hints on public functions, `from __future__ import annotations`.
- Comments explain *why*, not *what*. If a line encodes a non-obvious decision
  (shrinkage, mean-centring, frozen BatchNorm), say why it is there.
- Charts go through `src/viz/plots.py::use_house_style` and the `PALETTE`
  constant. Do not introduce a second visual style.
- New behaviour needs a test in `tests/test_pipeline.py`. Tests must run without
  a GPU and without downloading ImageNet weights — pass `weights=None`.

## Commands

```bash
pytest tests/ -v                    # 21 tests, ~25 s on CPU
python -c "from src.config import describe; print(describe())"
jupyter lab notebooks/
```

## When asked to improve results

Prefer, in this order: (1) verify the evaluation is honest — no test-set leakage,
no metric that hides class imbalance; (2) error analysis — look at the confusion
matrix and the confident mistakes before changing the model; (3) data quality;
(4) then architecture or hyperparameters. Report what did *not* work too —
negative results belong in the report.

Do not fabricate metrics. Bracketed `[…]` placeholders in the README and the
notebook conclusions are meant to be filled in from an actual run.
