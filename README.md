# 🏈 NFL Tweet Sentiment Classifier

A fine-tuned DistilBERT model for classifying NFL-related tweets as **negative**, **neutral**, or **positive**. This project demonstrates transfer learning and parameter-efficient fine-tuning on a sports-domain NLP dataset, adapted from a general-purpose language model to a domain-specific sentiment classifier.

---

## Motivation

General-purpose sentiment models struggle with sports tweets — phrases like "Mahomes just threw an interception" are factual but negative in fan context, and the base DistilBERT model outputs ~51% confidence (essentially random) on these inputs. This project fine-tunes DistilBERT on labeled NFL tweets to produce a domain-aware classifier that reaches **73.4% validation accuracy**.

---

## Repository Structure
```
L-LMs/
├── student/
│   └── Final_Project/
│       ├── Sports_sentiment.ipynb       # Main training + evaluation notebook
│       ├── nfl_sentiments.csv           # Labeled tweet dataset
│       ├── distilbert-nfl-sentiment/    # Training checkpoints (not tracked, see below)
│       └── my_fine_tuned_model/         # Saved model weights (not tracked, see below)
├── results/
│   ├── final_result1.png                # Loss & accuracy curves
│   ├── final_result2.png                # Base vs. fine-tuned confidence comparison
│   └── final_result3.png                # Saliency maps + t-SNE + cosine similarity
├── data/
│   └── nfl_sentiments.csv               # Original dataset
├── model_params/
│   └── sports_sentiment/                # Model config params
├── Final_Project_LLM_Handout.pdf
├── pyproject.toml                       # Pinned dependencies
└── .gitignore
```

---

## Environment Setup

This project uses `pyproject.toml` for dependency management. To recreate the environment:
```bash
# Clone the repo
git clone https://github.com/aleczhang2027/L-LMs.git
cd L-LMs

# Install dependencies (pip)
pip install transformers[torch] datasets evaluate accelerate peft scikit-learn matplotlib pandas numpy

# Or install from pyproject.toml directly
pip install .
```

Dependencies and pinned versions are specified in `pyproject.toml` at the repo root.

> **Hardware note:** Training was run on Apple Silicon (MPS). The notebook auto-detects MPS/CUDA/CPU — no changes needed to run on other hardware.

---

## Model Weights

The fine-tuned model weights (`model.safetensors`, ~260MB) are too large for GitHub. Download from Google Drive and place the contents into `student/Final_Project/my_fine_tuned_model/`:

👉 [Download model weights](https://drive.google.com/drive/folders/1fgeZaEo6pd9fQXJq_JlIBCGYanT4hW2_?usp=sharing)

Training checkpoints (optimizer state, ~118MB each) are also available on Google Drive if you need to resume training from step 388 or 582:

👉 [Download checkpoints](https://drive.google.com/drive/folders/1mZiLMuiEbS7QdzuBDtwpDboGg3Ec5ogu?usp=sharing)

---

## Quickstart — Inference Only

If you just want to run predictions using the fine-tuned model:
```bash
# 1. Download model weights from the Drive link above
# 2. Place contents in student/Final_Project/my_fine_tuned_model/
# 3. Run:

from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./student/Final_Project/my_fine_tuned_model",
    tokenizer="./student/Final_Project/my_fine_tuned_model"
)

classifier("Mahomes just threw an interception")
# [{'label': 'negative', 'score': 0.567}]

classifier("I hate this game so much")
# [{'label': 'negative', 'score': 0.923}]
```

---

## Reproducing the Full Training Run

To retrain from scratch and regenerate all results:
```bash
# 1. Make sure nfl_sentiments.csv is in student/Final_Project/
# 2. Open the notebook
jupyter notebook student/Final_Project/Sports_sentiment.ipynb
# 3. Run All Cells (Kernel > Restart & Run All)
```

This will reproduce:
- All training metrics (loss, accuracy)
- All figures saved to `results/`
- The fine-tuned model saved to `student/Final_Project/my_fine_tuned_model/`
- Training checkpoints saved to `student/Final_Project/distilbert-nfl-sentiment/`

---

## Output Management

| Output | Location | Description |
|---|---|---|
| Training checkpoints | `student/Final_Project/distilbert-nfl-sentiment/checkpoint-388/` and `checkpoint-582/` | Optimizer + scheduler state at each epoch. Not tracked by git. |
| Final model weights | `student/Final_Project/my_fine_tuned_model/` | Saved via `trainer.save_model()`. Not tracked by git. |
| Training logs | `student/Final_Project/distilbert-nfl-sentiment/trainer_state.json` | Full loss/accuracy history per step. |
| Result figures | `results/final_result1.png`, `final_result2.png`, `final_result3.png` | Generated inline in notebook, saved to `results/`. |

---

## Results

| Metric | Value |
|---|---|
| Validation Accuracy | 73.4% |
| Training Loss | 0.698 |
| Validation Loss | 0.638 |

**Example predictions:**

| Tweet | Prediction | Confidence |
|---|---|---|
| "I hate this game so much" | negative | 92.3% |
| "Mahomes just threw an interception" | negative | 56.7% |
| Base model (untrained) | ~LABEL_0 | ~51–53% |

---

## Visualizations

All figures are saved in `results/` and regenerated by running the notebook end-to-end:

- `final_result1.png` — Training vs. validation loss and accuracy curves per epoch
- `final_result2.png` — Side-by-side confidence comparison: base model vs. fine-tuned
- `final_result3.png` — Saliency maps (token importance), t-SNE embedding clusters, cosine similarity heatmap

---

## Model & Training Details

- **Base model:** `distilbert-base-uncased`
- **Task:** 3-class sequence classification (negative / neutral / positive)
- **Parameter-efficient training:** embeddings + first 4 transformer layers frozen → only 22% of parameters trainable (14.7M / 66.9M)
- **Optimizer:** AdamW, lr = 2e-5, cosine LR schedule, 100 warmup steps
- **Batch size:** 8 (train) · 16 (eval), gradient accumulation steps = 2
- **Epochs:** 3 with early stopping (patience = 2)
- **Hardware:** Apple Silicon MPS (M-series GPU)
