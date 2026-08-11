# Preserving Heritage: Enhancing Tourism with AI

**AIML Capstone 2** — two production-shaped ML systems for a government tourism agency:

1. **Part 1 · Image classification** — a TensorFlow CNN that identifies the category of a historical structure from a photograph, so the agency can automate condition monitoring of heritage sites.
2. **Part 2 · Recommendation engine** — EDA on Indonesian tourism data plus a collaborative filtering recommender that suggests where a tourist should go next.

---

## Results

| Part | Model | Headline metric |
|---|---|---|
| 1 | EfficientNetV2-B0, two-stage transfer learning | `[fill in: X% test accuracy, Y macro-F1]` |
| 2 | Item-based CF + Keras matrix factorisation | `[fill in: NDCG@10 vs popularity baseline]` |

> Run the notebooks, then replace the bracketed values here and in each notebook's Conclusions section.

---

## Quick start

### Google Colab (recommended)

1. Upload the dataset to Google Drive in this layout:

   ```
   MyDrive/heritage-data/
   ├── dataset_hist_structures 2.zip
   └── tourism/
       ├── user.csv
       ├── tourism_rating.csv
       └── tourism_with_id.xlsx
   ```

2. Open a notebook in Colab:

   [![Part 1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/heritage-tourism-ai/blob/main/notebooks/01_image_classification.ipynb) **Part 1 — Image Classification**

   [![Part 2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/heritage-tourism-ai/blob/main/notebooks/02_recommendation_engine.ipynb) **Part 2 — Recommendation Engine**

3. Set **Runtime → Change runtime type → GPU** (Part 1 only), edit `REPO_URL` in the first cell, and run all.

### Local

```bash
git clone https://github.com/USERNAME/heritage-tourism-ai.git
cd heritage-tourism-ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# place the data
mkdir -p data && unzip "dataset_hist_structures 2.zip" -d data
mkdir -p data/tourism && cp user.csv tourism_rating.csv tourism_with_id.xlsx data/tourism/

pytest tests/ -v
jupyter lab notebooks/
```

Full instructions, including the Drive layout and common errors, are in [`docs/SETUP.md`](docs/SETUP.md).

---

## Repository layout

```
heritage-tourism-ai/
├── notebooks/
│   ├── 01_image_classification.ipynb    Part 1 — run top to bottom
│   └── 02_recommendation_engine.ipynb   Part 2 — run top to bottom
├── src/
│   ├── config.py                 all hyperparameters and paths
│   ├── data/
│   │   ├── image_data.py         tf.data pipeline, sample grids, class weights
│   │   └── tourism_data.py       loading, schema validation, cleaning
│   ├── models/
│   │   ├── backbones.py          backbone registry, transfer-learning assembly
│   │   ├── item_cf.py            item-item CF + popularity baseline
│   │   └── recommender_net.py    Keras embedding matrix factorisation
│   ├── training/
│   │   ├── callbacks.py          custom accuracy-threshold callback
│   │   └── train_classifier.py   two-stage schedule, backbone benchmark
│   ├── evaluation/
│   │   ├── metrics.py            classification metrics + error analysis
│   │   └── ranking.py            RMSE, Precision/Recall/NDCG@K, coverage
│   └── viz/plots.py              shared chart style
├── tests/test_pipeline.py        21 smoke + regression tests
├── docs/
│   ├── SETUP.md                  environment, data placement, troubleshooting
│   ├── REQUIREMENTS_MAPPING.md   every brief task → where it is satisfied
│   └── PROJECT_REPORT.md         report template to submit
└── CLAUDE.md                     project context for Claude Code
```

The notebooks are thin: they orchestrate, narrate and visualise. All logic lives in `src/`, so it is testable and reviewable — and so a grader can see the engineering, not just a wall of cells.

---

## Part 1 — Image classification

**Data.** 10,245 training and 1,487 test images across **10 classes**: altar, apse, bell_tower, column, dome(inner), dome(outer), flying_buttress, gargoyle, stained_glass, vault.

> The archive's macOS metadata references an eleventh class, `portal`, but contains no `portal` images. This is a 10-class problem.

**Approach.**

- A short bake-off between **EfficientNetV2-B0**, **ResNet50V2** and **MobileNetV2** under identical conditions, so the architecture choice is evidence-based rather than asserted.
- **Stage A** — the entire convolutional base frozen, training only a new head (`GAP → BatchNorm → Dense(256, relu) → Dropout(0.4) → Dense(10, softmax)`).
- **Stage B** — unfreeze the top 30% of the base at a 100× smaller learning rate. BatchNorm layers stay frozen throughout.
- Both stages run **twice** — with and without augmentation — as the brief requires, and the training curves are compared to expose overfitting.

**Two details that matter more than they look:**

- *Preprocessing is per-backbone.* EfficientNet variants normalise inside the graph and expect raw `[0, 255]` pixels; ResNet50V2 and MobileNetV2 need their own `preprocess_input`. This is encoded in `BackboneSpec` so it cannot be got wrong by accident — it silently costs several accuracy points when it is.
- *The classes are imbalanced 4.7:1.* Inverse-frequency class weights are applied, and results are reported as macro-F1 and balanced accuracy alongside plain accuracy.

## Part 2 — Recommendation engine

**Data.** ~300 users, ~437 places, ~10,000 ratings — a **~92% sparse** matrix. That sparsity drives every modelling decision here.

**Three models, deliberately:**

| Model | What it answers |
|---|---|
| **Popularity baseline** | The bar the others must clear. On small data this is a strong strategy. |
| **Item-based CF** | *"You are at X — where next?"* The brief's literal requirement. Cosine similarity on the mean-centred matrix, with shrinkage so two co-raters cannot manufacture a similarity of 1.0. |
| **Keras matrix factorisation** | Personalised top-N per user, and a second independent route to place-to-place similarity via the learned embeddings. |

**Evaluated on both axes** — RMSE/MAE for rating prediction, and Precision/Recall/NDCG@10 plus catalogue coverage for ranking quality. A model can win on RMSE and still produce a useless top-10, which is why both appear.

Embedding dimension is capped at 32 with L2 regularisation and early stopping. With 10k ratings, a bigger network memorises rather than learns — and if the collaborative models do *not* beat the popularity baseline, that is reported as the finding rather than hidden.

---

## Testing

```bash
pytest tests/ -v
```

21 tests covering: the classifier graph builds and freezes correctly for all three backbones, BatchNorm stays frozen during fine-tuning, the custom callback fires at its threshold, augmentation is active in training and inert at inference, item-CF recovers a planted cluster structure and produces a symmetric bounded similarity matrix, NDCG rewards higher placement, the Keras model trains and its rating scaling round-trips, and every recommender returns a well-formed frame.

That last one is a regression test for a real bug found during development: indexing with an unnamed pandas `Index` discards the index name, so `reset_index()` produced a column called `index` and `place_id` was silently dropped from the output.

---

## Documentation

- [`docs/SETUP.md`](docs/SETUP.md) — environment setup, data placement, troubleshooting
- [`docs/REQUIREMENTS_MAPPING.md`](docs/REQUIREMENTS_MAPPING.md) — every numbered task in the brief mapped to the notebook section and source function that satisfies it
- [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) — report template to fill in and submit

## Licence

MIT — see [`LICENSE`](LICENSE).
