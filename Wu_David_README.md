# Tweet Sentiment Classifier — Video Games Edition

A fine-tuned DistilBERT model for classifying video game tweets as **negative**, **neutral**, or **positive**. This project is **Stage 2** of a multi-domain sentiment analysis project and demonstrates domain-specific transfer learning on a video-game-tweet corpus spanning 24 titles.

---

## Motivation

General-purpose sentiment models fail on gaming language. A tweet like *"Fortnite just dropped the best update ever, I can't stop playing!"* reads as enthusiastic positive text — but the base DistilBERT model outputs only ~67% confidence, essentially uncertain. The model misreads hyperbolic gaming slang as negative or ambiguous.

This project fine-tunes DistilBERT on ~44,800 labeled video game tweets to produce a domain-aware classifier that reaches **78.4% test accuracy** and a **0.774 macro F1**, with calibrated confidence scores that correctly flag edge cases like the one above.

---

## Results at a Glance

| Metric             | Value   |
|--------------------|---------|
| Test Accuracy      | 78.4%   |
| Macro F1           | 0.774   |
| Training Rows      | 38,080  |
| Test Rows          | 6,720   |
| Training Epochs    | 3       |
| Early Stopping     | Not triggered |

**Dataset composition:** ~44,800 video game tweets · 24 game titles · Positive ~33% · Negative ~33% · Neutral ~27% · Stratified 85/15 train/test split

---

## Repository Structure

```
L-LMs/
├── student/
│   └── Video_Games_Sentiment/
│       ├── VideoGame_Sentiment.ipynb       # Main training + evaluation notebook
│       ├── videogame_tweets.csv            # Labeled tweet dataset (44,800 rows)
│       ├── distilbert-vg-sentiment/        # Training checkpoints (not tracked, see below)
│       └── my_fine_tuned_vg_model/         # Saved model weights (not tracked, see below)
├── results/
│   ├── vg_confusion_matrix.png             # Confusion matrix — test set
│   ├── vg_confidence_comparison.png        # Base vs. fine-tuned confidence comparison
│   └── vg_training_curves.png             # Loss & accuracy curves across 3 epochs
├── data/
│   └── videogame_tweets.csv               # Original labeled dataset
├── model_params/
│   └── vg_sentiment/                      # Model config params
├── requirements.txt                       # Pinned dependencies
├── pyproject.toml                         # Alternative: full pyproject config
└── .gitignore
```

---

## Environment Setup

### Option A — pip + requirements.txt (recommended)

```bash
# Clone the repo
git clone https://github.com/aleczhang2027/L-LMs.git
cd L-LMs

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt
```

### Option B — pyproject.toml

```bash
pip install .
```

### requirements.txt (pinned versions)

```
torch>=2.0.0
transformers==4.40.0
datasets==2.19.0
evaluate==0.4.1
accelerate==0.29.3
peft==0.10.0
scikit-learn==1.4.2
matplotlib==3.8.4
pandas==2.2.2
numpy==1.26.4
```

> **Hardware note:** Training was run on Google Colab. The notebook auto-detects MPS/CUDA/CPU via `torch.device` — no changes needed to run on other hardware.

---

## Model Weights

The fine-tuned model weights (`model.safetensors`, ~260MB) are too large for GitHub. Download from Google Drive and place the contents into `student/Video_Games_Sentiment/my_fine_tuned_vg_model/`:

👉 [Download model weights](https://drive.google.com/drive/folders/1VJIXbUqf9zwc8vzKVag7PMaWdx0HBei9?usp=drive_link)

Training checkpoints are also available if you need to resume training:

👉 [Download checkpoints](https://drive.google.com/drive/folders/1VJIXbUqf9zwc8vzKVag7PMaWdx0HBei9?usp=drive_link)

---

## Quickstart — Inference Only

If you just want to run predictions using the fine-tuned model:

```bash
# 1. Download model weights from the Drive link above
# 2. Place contents in student/Video_Games_Sentiment/my_fine_tuned_vg_model/
# 3. Run the snippet below
```

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./student/Video_Games_Sentiment/my_fine_tuned_vg_model",
    tokenizer="./student/Video_Games_Sentiment/my_fine_tuned_vg_model"
)

classifier("Fortnite just dropped the best update ever, I can't stop playing!")
# [{'label': 'negative', 'score': 0.67}]  ← correctly uncertain on hyperbolic text

classifier("This game is absolutely broken, uninstalling now")
# [{'label': 'negative', 'score': 0.95}]

classifier("Finally hit Diamond rank after 200 hours, worth it")
# [{'label': 'positive', 'score': 0.88}]
```

> **Key insight:** The Fortnite example above returns low confidence (0.67) because enthusiastic gaming language is frequently misread as negative. The model correctly signals uncertainty rather than over-committing — a feature, not a bug. Training loss decreased steadily across all 3 epochs with no early stopping triggered.

---

## Reproducing the Full Training Run

To retrain from scratch and regenerate all results:

```bash
# 1. Make sure videogame_tweets.csv is in student/Video_Games_Sentiment/
# 2. Open the notebook
jupyter notebook student/Video_Games_Sentiment/VideoGame_Sentiment.ipynb
# 3. Run All Cells (Kernel > Restart & Run All)
```

This will reproduce:
- All training metrics (loss, accuracy per epoch)
- All figures saved to `results/`
- The fine-tuned model saved to `student/Video_Games_Sentiment/my_fine_tuned_vg_model/`
- Training checkpoints saved to `student/Video_Games_Sentiment/distilbert-vg-sentiment/`

### One-command reproduction (script)

```bash
python scripts/train_vg_sentiment.py \
  --data_path data/videogame_tweets.csv \
  --output_dir student/Video_Games_Sentiment/my_fine_tuned_vg_model \
  --epochs 3 \
  --batch_size 8 \
  --lr 2e-5
```

**Script arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data_path` | `data/videogame_tweets.csv` | Path to labeled tweet CSV |
| `--output_dir` | `my_fine_tuned_vg_model/` | Where to save the final model |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `8` | Training batch size |
| `--lr` | `2e-5` | Learning rate |
| `--eval_batch_size` | `16` | Evaluation batch size |
| `--warmup_steps` | `100` | LR scheduler warmup steps |
| `--seed` | `42` | Random seed for reproducibility |

---

## Output Management

| Output | Location | Description |
|---|---|---|
| Training checkpoints | `student/Video_Games_Sentiment/distilbert-vg-sentiment/checkpoint-*/` | Optimizer + scheduler state per epoch. Not tracked by git. |
| Final model weights | `student/Video_Games_Sentiment/my_fine_tuned_vg_model/` | Saved via `trainer.save_model()`. Not tracked by git. |
| Training logs | `student/Video_Games_Sentiment/distilbert-vg-sentiment/trainer_state.json` | Full loss/accuracy history per step. |
| Confusion matrix | `results/vg_confusion_matrix.png` | Test set confusion matrix (3×3). |
| Confidence comparison | `results/vg_confidence_comparison.png` | Base vs. fine-tuned confidence per sentiment class. |
| Training curves | `results/vg_training_curves.png` | Loss & accuracy across 3 epochs. |

To regenerate all figures from a saved model:

```bash
python scripts/eval_vg_sentiment.py \
  --model_dir student/Video_Games_Sentiment/my_fine_tuned_vg_model \
  --data_path data/videogame_tweets.csv \
  --output_dir results/
```

---

## Visualizations

All figures are saved in `results/` and regenerated by running the notebook end-to-end:

- `vg_confusion_matrix.png` — Test set confusion matrix showing true vs. predicted labels across negative / neutral / positive
- `vg_confidence_comparison.png` — Side-by-side bar chart: base DistilBERT confidence vs. fine-tuned model confidence, grouped by sentiment class and game title
- `vg_training_curves.png` — Training vs. validation loss and accuracy per epoch (3 epochs, no early stopping)

---

## Model & Training Details

| Parameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Task | 3-class sequence classification (negative / neutral / positive) |
| Dataset | ~44,800 labeled video game tweets · 24 game titles |
| Train/test split | 85/15 stratified |
| Frozen layers | Embeddings + first 4 transformer layers |
| Trainable parameters | ~22% of total (≈14.7M / 66.9M) |
| Optimizer | AdamW |
| Learning rate | 2e-5 with cosine LR schedule |
| Warmup steps | 100 |
| Train batch size | 8 |
| Eval batch size | 16 |
| Gradient accumulation | 2 steps |
| Epochs | 3 |
| Early stopping | Patience = 2 (not triggered) |

---

## Dataset

**Source:** https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

The dataset contains ~44,800 labeled tweets covering 24 video game titles. Labels are:
- **Positive** (~33%) — praise, excitement, wins, updates
- **Negative** (~33%) — complaints, bugs, losses, toxicity
- **Neutral** (~27%) — factual commentary, patch notes, observations

Preprocessing: lowercasing, removal of URLs and mentions, truncation to 128 tokens (DistilBERT max).

---

## Individual Contributions

| Team Member | Component |
|---|---|
| **David Wu** | Stage 2 — Video Games Sentiment Model: dataset curation, DistilBERT fine-tuning, evaluation, confusion matrix & confidence comparison visualizations |
| **Alec Zhang** | Stage 1 — NFL Sentiment Model: dataset prep, DistilBERT fine-tuning, saliency maps, t-SNE embeddings, cosine similarity heatmap |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

```
MIT License

Copyright (c) 2025 David Wu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
