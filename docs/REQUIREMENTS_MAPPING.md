# Brief → Implementation traceability

Every numbered task in the capstone problem statement, mapped to the notebook
section that demonstrates it and the source function that implements it. Use
this as the checklist before submitting.

---

## Part 1 — Image classification

| # | Task (from the brief) | Notebook section | Implementation |
|---|---|---|---|
| 1 | Plot sample images (8–10) from each class, hint: OpenCV | `01` §3 | `src/data/image_data.py::plot_class_samples` |
| 2 | Select a CNN architecture, configure for transfer learning, set up the TensorFlow environment, load pre-trained weights | `01` §5, §7 | `src/models/backbones.py::build_backbone`, `BACKBONES` registry; benchmark in `src/training/train_classifier.py::benchmark_backbones` |
| 3 | Use pre-trained CNN weights and **freeze all convolutional layers** | `01` §5 | `build_backbone(trainable=False)`; verified by `tests::test_classifier_builds_and_freezes` |
| 4 | Modify the top: appropriate dense layers with activation, dropout for regularisation | `01` §5 | `src/models/backbones.py::build_classifier` — `GAP → BatchNorm → Dense(256, relu) → Dropout(0.4) → Dense(10, softmax)` |
| 5 | Compile with the right optimizer, loss function and metric | `01` §5 | `src/models/backbones.py::compile_model` — Adam, SparseCategoricalCrossentropy, accuracy |
| 6 | Define a callback class to stop training once validation accuracy reaches a chosen number | `01` §6 | `src/training/callbacks.py::make_accuracy_threshold_callback` (target 93%); verified by `tests::test_accuracy_threshold_callback_stops_training` |
| 7 | Set up train/test directories and review sample counts per class | `01` §2 | `src/data/image_data.py::dataset_summary`, `count_images_per_class` |
| — | Image data quality audit before training | `01` §3 | `src/data/image_audit.py` — corrupt files, duplicates, cross-class label noise, blank images, aspect-ratio outliers, **train/test leakage**, per-class brightness statistics |
| 8 | Train **without** augmentation, monitoring validation accuracy | `01` §8 | `src/training/train_classifier.py::train_two_stage` (`run_name="no_aug"`) |
| 10 | Train **with** augmentation, monitoring validation accuracy | `01` §9 | same, with `augmentation=build_augmentation()` (`run_name="with_aug"`) |
| 12 | Visualise training and validation accuracy per epoch to see if the model overfits | `01` §10 | `src/viz/plots.py::plot_training_curves`, `compare_histories` |

> The brief numbers its tasks 1–12 but skips 9 and 11; the numbering above
> follows the source document exactly rather than renumbering.

### Beyond the brief

| Addition | Why | Where |
|---|---|---|
| Three-backbone benchmark | The brief says "select the one that performs best" — this makes that claim evidence-based | `01` §7 |
| Stage B fine-tuning | Frozen-only transfer learning leaves several accuracy points on the table | `01` §8–9 |
| Class weights + macro-F1 + balanced accuracy | Classes are imbalanced 4.7:1; plain accuracy hides failure on rare classes | `01` §2, §11 |
| Confusion matrix + most-confused pairs + confident mistakes | Turns a single number into an error analysis | `01` §11 |
| Held-out test set never used for model selection | Keeps the headline number honest | `01` §4, §11 |

---

## Part 2 — Recommendation engine

| # | Task (from the brief) | Notebook section | Implementation |
|---|---|---|---|
| 1 | Import all datasets and perform preliminary inspection | `02` §2–3 | `src/data/tourism_data.py::load_raw`, `inspect`; `src/data/eda.py` |
| 1.I | Check for missing values and duplicates | `02` §2, §3.4 | `inspect`; `eda.missingness_report`, `eda.missingness_mechanism` (MCAR vs MAR chi-square) |
| 1.II | Remove any anomalies found in the data | `02` §2, §3.2–3.3 | `clean` — duplicate (user, place) pairs, out-of-range ratings, impossible ages, orphan foreign keys; `eda.detect_outliers` (IQR/z-score/MAD), `eda.geographic_outliers`, `eda.coordinate_consistency`, `eda.country_bounds_check` |
| — | Descriptive statistics before modelling | `02` §3.1 | `eda.numeric_summary` (incl. skew/kurtosis/CV), `eda.categorical_summary`, `eda.correlation_matrix` |
| — | Normalisation / scaling | `02` §3.5 | `eda.transform_report`, `log1p_transform`, `robust_scale`, `minmax_scale` |
| 2.I | Analyse the age distribution of users who rate places | `02` §3 | histogram + age bands |
| 2.I | Identify where most of these tourists come from | `02` §3 | `location` / derived `province` frequency charts |
| 3.I | What are the different categories of tourist spots? | `02` §4 | `places["category"].value_counts()` |
| 3.II | What kind of tourism is each location most famous or suitable for? | `02` §4 | city × category crosstab + stacked share chart |
| 3.III | Which city would be best for a nature enthusiast? | `02` §4 | nature-category analysis by **volume** and by **concentration** — both reported, since the two answers differ |
| 4 | Create combined data of places and their user ratings | `02` §5 | `src/data/tourism_data.py::build_merged` |
| 4.I | Which spots are most loved? Which city has the most-loved spots? | `02` §5 | `place_popularity` with a Bayesian-shrunken score, plus per-city aggregation |
| 4.II | Which category of places do users like most? | `02` §5 | category means **with bootstrap confidence intervals**, so a difference is only claimed if it is real |
| 5.I | Develop a collaborative filtering model and use it to recommend other places from the current tourist location (place name) | `02` §6, §9 | `src/models/item_cf.py::ItemBasedCF.recommend_similar` — accepts a place name, returns ranked places |

### Beyond the brief

| Addition | Why | Where |
|---|---|---|
| **Data signal diagnostic** | **Establishes that `Place_Ratings` is statistically indistinguishable from random before any model is built — without it, the modelling results are uninterpretable** | `02` §6, `src/evaluation/signal.py` |
| Random-recommender floor | The popularity baseline is not the floor; random is. Any model that cannot beat random has learned nothing | `signal.make_random_recommend_fn`, `02` §9 |
| Keras embedding matrix factorisation | Keeps the whole capstone on TensorFlow; gives personalised top-N and a second route to place similarity | `02` §8 |
| Popularity baseline | Without it, no one can tell whether the CF models add value | `02` §9 |
| Precision/Recall/NDCG@10 + catalogue coverage | RMSE alone does not measure the quality of the list a tourist actually sees | `src/evaluation/ranking.py` |
| Cross-model agreement check | If two independent methods disagree completely on a 92.7%-sparse matrix, both are fitting noise | `02` §8 |
| Bootstrap CIs on category means | The category gaps are small; this tests whether they are real (they are not) | `02` §5 |
| Shrinkage on item similarities | Two co-raters can otherwise produce a similarity of 1.0 that means nothing | `ItemBasedCF.fit` |

---

## Pre-submission checklist

- [ ] Both notebooks run top to bottom without errors
- [ ] `pytest tests/ -v` passes (44 tests)
- [ ] Bracketed `[…]` placeholders filled in — README results table, both Conclusions sections, `docs/PROJECT_REPORT.md`
- [ ] `REPO_URL` updated in both notebooks' first cell
- [ ] Colab badge URLs in README point at your GitHub username
- [ ] Notebook outputs saved (a grader should see results without re-running)
- [ ] Final model artefacts exported from `artifacts/`
- [ ] Every table and figure referenced in the report actually appears in a notebook
