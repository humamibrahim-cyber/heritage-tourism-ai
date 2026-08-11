# Setup guide

Three ways to run this project. Colab is the path of least resistance for Part 1
because it needs a GPU.

---

## 1. Google Colab

### Step 1 — put the data on Drive

Create this exact structure in your Google Drive:

```
MyDrive/
└── heritage-data/
    ├── dataset_hist_structures 2.zip      ← Part 1 (133 MB)
    └── tourism/
        ├── user.csv                       ← Part 2
        ├── tourism_rating.csv
        └── tourism_with_id.xlsx
```

Upload the zip **as a zip**. Do not unzip it into Drive: extracting 11,000 small
files onto Drive takes a long time, and reading them back over the Drive FUSE
mount is roughly an order of magnitude slower than local disk — it will
bottleneck the GPU. The notebook copies the zip to local disk and extracts there.

### Step 2 — open the notebook

From GitHub: `https://colab.research.google.com/github/USERNAME/heritage-tourism-ai/blob/main/notebooks/01_image_classification.ipynb`

Or **File → Open notebook → GitHub** and paste the repo URL.

### Step 3 — configure

1. **Runtime → Change runtime type → GPU** (Part 1; Part 2 runs fine on CPU).
2. In the first code cell, set `REPO_URL` to your repository.
3. **Runtime → Run all**.

### Sizing for your GPU

`ImageConfig` in the notebook's config cell:

| GPU | `batch_size` | `image_size` | Approx. time for the full Part 1 |
|---|---|---|---|
| A100 / L4 (Colab Pro) | 64 | (224, 224) | ~45–70 min including the benchmark |
| V100 (Colab Pro) | 48 | (224, 224) | ~70–100 min |
| T4 (free tier) | 32 | (192, 192) | ~2–3 h, or set `RUN_BENCHMARK = False` |

If you hit `ResourceExhaustedError`, halve `batch_size` first, then reduce
`image_size`.

---

## 2. Local

Requires Python 3.10–3.12. A GPU is optional but Part 1 is slow without one.

```bash
git clone https://github.com/USERNAME/heritage-tourism-ai.git
cd heritage-tourism-ai

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Place the data:

```bash
mkdir -p data
unzip "dataset_hist_structures 2.zip" -d data

# flatten the nested folder the archive creates
mv "data/dataset_hist_structures 2/dataset_hist_structures" data/

# remove macOS metadata, which confuses image loaders
find data -name "__MACOSX" -type d -exec rm -rf {} +
find data -name ".DS_Store" -delete

mkdir -p data/tourism
cp user.csv tourism_rating.csv tourism_with_id.xlsx data/tourism/
```

Expected result:

```
data/
├── dataset_hist_structures/
│   ├── Stuctures_Dataset/              ← train (note the typo, it is in the source)
│   │   ├── altar/ … vault/
│   └── Dataset_test/
│       └── Dataset_test_original_1478/
│           ├── altar/ … vault/
└── tourism/
    ├── user.csv
    ├── tourism_rating.csv
    └── tourism_with_id.xlsx
```

Verify:

```bash
pytest tests/ -v
python -c "from src.config import describe; print(describe())"
jupyter lab notebooks/
```

---

## 3. Claude Code

The repo ships a `CLAUDE.md` that gives Claude Code the project's conventions,
data quirks and gotchas up front.

```bash
npm install -g @anthropic-ai/claude-code
cd heritage-tourism-ai
claude
```

Useful prompts once inside:

```
Read CLAUDE.md, then explain how transfer learning is set up in src/models/backbones.py
Add DenseNet121 to the backbone benchmark and run the tests
Why does the confusion matrix mix up dome(inner) and vault? Suggest an experiment
Write the PROJECT_REPORT.md conclusions using the numbers in artifacts/part1_results.json
```

---

## Troubleshooting

**`FileNotFoundError` on `Stuctures_Dataset`**
Not a typo on your side — the folder really is spelled `Stuctures_Dataset` in the
supplied archive. Check the nested folder was flattened (step above).

**`Found 0 files belonging to 0 classes`**
`__MACOSX` and `.DS_Store` were not removed, or `train_dir` points one level too
high. Run `ls data/dataset_hist_structures/Stuctures_Dataset` — you should see
the 10 class folders directly.

**`ResourceExhaustedError: OOM`**
Reduce `batch_size` (64 → 32 → 16), then `image_size` (224 → 192 → 160). Restart
the runtime afterwards: TensorFlow does not release GPU memory between cells.

**Accuracy stuck near 10%**
10% is chance on 10 classes. Almost always wrong preprocessing for the backbone —
EfficientNet expects raw `[0, 255]`, ResNet/MobileNet need `preprocess_input`.
`build_classifier` handles this automatically, so check you did not bypass it.

**Fine-tuning makes accuracy collapse**
The learning rate is too high, or BatchNorm was unfrozen. `unfreeze_top` keeps
BatchNorm frozen; `cfg.finetune_lr` should stay around `1e-5`.

**`ImportError: No module named src`**
The repo root is not on `sys.path`. In Colab, confirm the `git clone` in cell 1
succeeded. Locally, launch Jupyter from the repo root.

**`ValueError: ... is ambiguous. Did you mean: …`**
`recommend_similar` matched several place names. Pass a longer or exact name.

**`KeyError: '<place>' has fewer than N ratings`**
Cold start — that place was excluded from the similarity matrix. Lower
`cfg.min_ratings_per_place`, accepting that similarities from very few co-raters
are unreliable.

**Drive mount fails in Colab**
Re-run the cell and complete the auth prompt. If it persists,
`drive.mount('/content/drive', force_remount=True)`.
