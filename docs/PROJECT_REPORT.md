# Preserving Heritage: Enhancing Tourism with AI
### AIML Capstone 2 — Project Report

**Author:** `[your name]`
**Date:** `[date]`
**Repository:** `https://github.com/USERNAME/heritage-tourism-ai`

> Fill in every `[…]` from an actual notebook run. Do not estimate figures.

---

## 1. Executive summary

A government tourism agency needs two capabilities: automated monitoring of
historical structures, and a way to recommend destinations to visitors. This
project delivers both.

**Part 1** classifies photographs of historical structures into 10 architectural
categories using transfer learning on `[BACKBONE]`, reaching **`[X]%` accuracy**
and **`[Y]` macro-F1** on 1,487 held-out test images.

**Part 2** analyses 9,597 ratings from 300 visitors across 437 Indonesian
destinations and delivers a collaborative filtering recommender that, given the
place a tourist is currently visiting, suggests where to go next.

It also establishes — through five independent statistical checks — that the
supplied user ratings contain **no preference signal**: they are statistically
indistinguishable from random numbers. No recommender can outperform random
selection on this data. The system is built and validated correctly; the
dataset cannot support the conclusion the brief anticipates, and this report
says so rather than presenting noise as a result.

**Headline recommendation to the agency:** `[one or two sentences — what should
they actually do with this?]`

---

## 2. Business context

Centuries-old historical structures preserve a community's history and drive
tourism. The agency wants to (a) monitor their condition at scale and (b) market
destinations more effectively by understanding visitor preferences.

Manual inspection of heritage sites does not scale. An image model that
identifies structural elements is the first building block of an automated
monitoring pipeline. Separately, generic marketing wastes budget; a recommender
that surfaces relevant destinations improves conversion and can spread visitors
beyond the handful of already-crowded sites.

---

## 3. Part 1 — Image classification

### 3.1 Data

| | |
|---|---|
| Training images | 10,245 |
| Test images | 1,487 |
| Classes | 10 |
| Source resolution | 128 × 128 |
| Model input | 224 × 224 |
| Imbalance | 4.7× (column 1,920 · flying_buttress 408) |

Classes: altar, apse, bell_tower, column, dome(inner), dome(outer),
flying_buttress, gargoyle, stained_glass, vault.

**Data-quality note.** The archive's macOS metadata references a `portal` class
that contains no images. This was verified before modelling and the problem
treated as 10-class. Reporting it as 11 classes with one empty would have been
incorrect.

### 3.2 Architecture selection

`[Paste the benchmark table from notebook 01 §7]`

| Backbone | Params (M) | Best val accuracy | Sec/epoch |
|---|---|---|---|
| EfficientNetV2-B0 | 7.1 | `[…]` | `[…]` |
| ResNet50V2 | 25.6 | `[…]` | `[…]` |
| MobileNetV2 | 3.5 | `[…]` | `[…]` |

**Selected: `[BACKBONE]`** — `[why, referencing the numbers above]`.

### 3.3 Transfer learning setup

```
input (224×224×3)
  → [augmentation]           (run 2 only)
  → [backbone preprocessing] (architecture-specific)
  → pre-trained conv base    FROZEN in stage A
  → GlobalAveragePooling2D
  → BatchNormalization
  → Dense(256, relu)
  → Dropout(0.4)
  → Dense(10, softmax)
```

- **Stage A** — base entirely frozen, head only, Adam @ 1e-3.
- **Stage B** — top 30% unfrozen, Adam @ 1e-5, BatchNorm kept frozen.
- Custom callback stops training at 93% validation accuracy.
- Inverse-frequency class weights applied throughout.

`GlobalAveragePooling2D` was chosen over `Flatten` because flattening a
7×7×1280 feature map into a dense layer creates ~16M parameters in one step and
overfits almost immediately on 10k images.

### 3.4 Results

| Run | Best val accuracy | Epoch | Train acc at best | Generalisation gap |
|---|---|---|---|---|
| No augmentation | `[…]` | `[…]` | `[…]` | `[…]` |
| With augmentation | `[…]` | `[…]` | `[…]` | `[…]` |

**Test set (1,487 images, never used for training or selection):**

| Metric | Value |
|---|---|
| Accuracy | `[…]` |
| Balanced accuracy | `[…]` |
| Macro F1 | `[…]` |
| Weighted F1 | `[…]` |
| Top-3 accuracy | `[…]` |
| Cohen's κ | `[…]` |

### 3.5 Effect of augmentation

`[Did augmentation raise peak accuracy, or mainly narrow the generalisation
gap? Quote both numbers. A narrower gap at similar accuracy is still the better
model — say so and explain why.]`

### 3.6 Error analysis

`[Paste the most-confused-pairs table]`

`[Which confusions are semantically plausible — dome(inner)/vault,
altar/apse — versus which indicate a pipeline problem? What would you need to
fix the plausible ones: more data, higher resolution, multiple viewpoints?]`

---

## 4. Part 2 — Recommendation engine

### 4.1 Data and cleaning

| | |
|---|---|
| Users | 300 |
| Places | 437 |
| Ratings | 10,000 raw → 9,597 after cleaning |
| Sparsity | 92.68% |

Anomalies removed: **403 duplicate (user, place) pairs**. No out-of-range
ratings, impossible ages or orphan foreign keys were present.

`Time_Minutes` (53.1% null) was left as `NaN` rather than mean-imputed —
inventing tour durations would corrupt every figure derived from it. The two
trailing unnamed columns in the source file were dropped (`Unnamed: 12`
duplicates `Place_Id`).

### 4.2 Data quality: the ratings contain no preference signal

**This is the central finding of Part 2 and it governs everything after it.**

Before modelling, five independent diagnostics were run on `Place_Ratings`
(notebook 02 §6). Four failed:

| Check | Observed | Expected if the ratings were real |
|---|---|---|
| Distribution shape | mean 3.066, skew −0.048, near-uniform across 1–5 | mean 3.8–4.3, clearly negative skew |
| Between-place ANOVA | F = 1.079, **p = 0.128** | significant |
| Split-half reliability of place means | r = **0.005** (shuffled null 0.001 ± 0.049, z = 0.09) | clearly positive |
| Correlation with the independent listed `Rating` | r = **0.010, p = 0.829** | positive |
| Category / city effects | 0.107% / 0.082% of variance, p = 0.067 / 0.098 | meaningful |

The split-half test is the most direct: a place's average rating computed from
one random half of the data does not predict its average from the other half.
The external-validity test is the hardest to explain away: the places table
carries an independently sourced rating, and the user ratings show **zero**
correlation with it.

**Conclusion:** the user-level ratings are synthetic — randomly generated
numbers, not observed behaviour. The `Rating` column in `tourism_with_id.xlsx`
is genuine; only `tourism_rating.csv` is not.

This was confirmed empirically. All three recommenders — including one that
picks places at random — score the same:

| Model | NDCG@10 |
|---|---|
| Random | 0.019 |
| Popularity baseline | 0.016 |
| Item-based CF | 0.018 |

Note also that RMSE ≈ 1.41 for every model, which is exactly the standard
deviation of the ratings — i.e. the "predict the global mean" floor. Reporting
RMSE alone would have made the models look reasonable and hidden this entirely.

### 4.3 Exploratory findings

**Who rates? (task 2)** `[age distribution, dominant home locations, and what
this implies about who the marketing actually reaches]`

**What and where are the spots? (task 3)** `[categories; what each city is known
for]`

**Best city for a nature enthusiast (task 3.III):** `[Answer BOTH readings —
most nature attractions by volume vs highest concentration — and state which
you would recommend and why.]`

**Most-loved spots (task 4.I):** `[top spots by Bayesian-adjusted score, and
which city leads. Explain why a raw mean would have been misleading.]`

**Most-liked category (task 4.II):** `[the ranking, AND whether the bootstrap
confidence intervals actually separate. If they overlap, say that the
differences are not statistically meaningful — that is the honest finding.]`

### 4.4 Models

| Model | Approach | Answers |
|---|---|---|
| Popularity baseline | Bayesian-shrunken global ranking | The bar to beat |
| Item-based CF | Cosine similarity, mean-centred, shrunk | "You are at X, go to…" |
| Keras MF | 32-dim embeddings + biases, sigmoid | Personalised top-N |

Embedding dimension was capped at 32 with L2 regularisation and early stopping.
With ~10k ratings a larger model memorises rather than generalises.

### 4.5 Results

**Rating prediction:**

| Model | RMSE | MAE |
|---|---|---|
| Popularity baseline | `[…]` | `[…]` |
| Item-based CF | `[…]` | `[…]` |
| Keras MF | `[…]` | `[…]` |

**Top-10 ranking:**

| Model | Precision@10 | Recall@10 | NDCG@10 | Hit rate | Coverage |
|---|---|---|---|---|---|
| Popularity baseline | `[…]` | `[…]` | `[…]` | `[…]` | `[…]` |
| Item-based CF | `[…]` | `[…]` | `[…]` | `[…]` | `[…]` |
| Keras MF | `[…]` | `[…]` | `[…]` | `[…]` | `[…]` |

**Cross-model agreement:** the two collaborative models shared `[X]%` of their
top-10 neighbours, indicating `[genuine shared signal / mostly noise]`.

### 4.6 Interpretation

No model beats the random floor, and §4.2 explains why: there is no preference
structure in `Place_Ratings` to learn. The honest reading is therefore:

1. **The models are correctly implemented and correctly evaluated.** Item-based
   CF recovers a planted cluster structure on synthetic data (see
   `tests/test_pipeline.py::test_item_cf_recovers_cluster_structure`), so the
   flat result here reflects the data, not a bug.
2. **The agency cannot personalise from this dataset.** Any apparent
   "recommendation quality" would be an artefact.
3. **One qualitative difference survives and is worth reporting.** Catalogue
   coverage: the popularity baseline recommends the same ~3% of destinations to
   everyone, while item-CF spans ~99% of the catalogue. If the agency's goal is
   to spread visitors beyond a handful of crowded sites rather than to maximise
   click-through, coverage matters independently of accuracy.
4. **What would be needed to do this properly:** real logged interactions —
   bookings, visits, dwell time, or verified reviews — with timestamps. Roughly
   50–100k genuine interactions would support meaningful collaborative
   filtering at this catalogue size.

`[Add your own reading here, and state explicitly which of the above you would
put in front of the client.]`

---

## 5. Limitations

**Part 1**

- Source images are 128×128 and upsampled; native higher resolution would likely add accuracy.
- Residual confusions reflect genuine visual ambiguity from a single viewpoint.
- **The model classifies structures; it does not assess condition.** The business scenario asks about maintenance needs, which would require a labelled damage dataset this project does not have. This is the largest gap between the deliverable and the stated business goal.

**Part 2**

- **The user ratings are synthetic** (§4.2). This is the dominant limitation: no collaborative filtering result from this dataset can be trusted, and the recommender is best understood as a correctly built system awaiting real data.
- 92.7% sparse matrix; all conclusions rest on 9,597 interactions.
- No cold-start handling — a place nobody has rated cannot be recommended. A content-based model over Category/City/Description is the natural next step.
- No timestamps, so the split is random rather than chronological and cannot detect taste drift.
- Popularity bias: the system tends to reinforce existing tourist flows.
- The rating population is young and Java-centric; findings should not be extrapolated to other segments.

---

## 6. Recommendations to the agency

1. `[…]`
2. `[…]`
3. `[…]`

---

## 7. Reproducibility

```bash
git clone https://github.com/USERNAME/heritage-tourism-ai.git
cd heritage-tourism-ai
pip install -r requirements.txt
pytest tests/ -v
```

Seeds fixed at 42. `docs/REQUIREMENTS_MAPPING.md` maps every task in the brief
to the code that implements it. All logic sits in `src/` under test; the
notebooks orchestrate and narrate.

## 8. References

- Llamas, J. et al. *Classification of Architectural Heritage Images Using Deep Learning Techniques.* Applied Sciences, 2017.
- Tan, M. & Le, Q. *EfficientNetV2: Smaller Models and Faster Training.* ICML, 2021.
- Sarwar, B. et al. *Item-Based Collaborative Filtering Recommendation Algorithms.* WWW, 2001.
- Koren, Y., Bell, R. & Volinsky, C. *Matrix Factorization Techniques for Recommender Systems.* IEEE Computer, 2009.
