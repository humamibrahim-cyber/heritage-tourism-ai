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

**Part 2** analyses `[N]` ratings from `[U]` visitors across `[P]` Indonesian
destinations and delivers a collaborative filtering recommender that, given the
place a tourist is currently visiting, suggests where to go next. It achieves
**NDCG@10 of `[Z]`** against a popularity baseline of `[B]`.

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
| Users | `[…]` |
| Places | `[…]` |
| Ratings | `[…]` |
| Sparsity | `[…]%` |

Anomalies removed: `[paste the cleaning log from notebook 02 §2]`

`time_minutes` (~50% null) was left as `NaN` rather than mean-imputed —
inventing tour durations would corrupt every figure derived from it.

### 4.2 Exploratory findings

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

### 4.3 Models

| Model | Approach | Answers |
|---|---|---|
| Popularity baseline | Bayesian-shrunken global ranking | The bar to beat |
| Item-based CF | Cosine similarity, mean-centred, shrunk | "You are at X, go to…" |
| Keras MF | 32-dim embeddings + biases, sigmoid | Personalised top-N |

Embedding dimension was capped at 32 with L2 regularisation and early stopping.
With ~10k ratings a larger model memorises rather than generalises.

### 4.4 Results

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

### 4.5 Interpretation

`[Did the CF models beat the popularity baseline on NDCG@10? If yes,
personalisation adds value and should ship. If no, say plainly that with ~10k
ratings there is insufficient signal to personalise, and that promoting popular
destinations is the better interim strategy. Then use coverage to nuance this:
a model with similar accuracy but far higher coverage spreads visitors across
the country instead of funnelling them to the same five sites, which serves the
agency's actual goal.]`

---

## 5. Limitations

**Part 1**

- Source images are 128×128 and upsampled; native higher resolution would likely add accuracy.
- Residual confusions reflect genuine visual ambiguity from a single viewpoint.
- **The model classifies structures; it does not assess condition.** The business scenario asks about maintenance needs, which would require a labelled damage dataset this project does not have. This is the largest gap between the deliverable and the stated business goal.

**Part 2**

- ~92% sparse matrix; all conclusions rest on ~10k interactions.
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
