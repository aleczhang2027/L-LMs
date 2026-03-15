# 🏈 NFL Tweet Sentiment Classifier

A fine-tuned DistilBERT model for classifying NFL-related tweets as **negative**, **neutral**, or **positive**. Built as a demonstration of transfer learning and parameter-efficient fine-tuning on a sports-domain NLP dataset.

---

## Overview

The base `distilbert-base-uncased` model has no sentiment knowledge out of the box — it outputs ~51% confidence on classification tasks before any fine-tuning. This project adapts it to NFL tweet sentiment using a labeled dataset of ~5,000 tweets, achieving **73.4% validation accuracy** after 3 epochs of training.

---

## Dataset

- **Source:** `nfl_sentiments.csv`
- **Size:** 5,153 tweets
- **Classes:** negative (1,757) · neutral (2,235) · positive (1,161)
- **Splits (stratified):**
  - Train: 3,091 (60%)
  - Validation: 1,031 (20%)
  - Test: 1,031 (20%)

---

## Model & Training

- **Base model:** `distilbert-base-uncased`
- **Task:** 3-class sequence classification
- **Parameter-efficient training:** embeddings + first 4 transformer layers frozen → only 22% of parameters trainable (14.7M / 66.9M)
- **Optimizer:** AdamW, lr = 2e-5, cosine schedule, 100 warmup steps
- **Batch size:** 8 (train) · 16 (eval), with gradient accumulation steps = 2
- **Epochs:** 3 (with early stopping, patience = 2)
- **Hardware:** Apple Silicon MPS (M-series GPU)

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
| Base model on any tweet | ~LABEL_0 | ~51–53% |

---

## Visualizations

- **Loss & accuracy curves** — training vs. validation performance per epoch
- **Confidence comparison** — base model vs. fine-tuned side by side on sample tweets
- **Saliency maps** — token-level heatmaps showing which words drove each prediction
- **t-SNE plots** — 2D visualization of how tweet embeddings cluster by sentiment class
- **Cosine similarity heatmap** — inter-tweet similarity across sentiment groups in the fine-tuned embedding space

---

## Project Structure
```
├── Sports_sentiment.ipynb   # Main notebook
├── nfl_sentiments.csv       # Labeled tweet dataset
└── my_fine_tuned_model/     # Saved model + tokenizer
```

---

## Requirements
```
transformers
datasets
evaluate
accelerate
peft
torch
scikit-learn
matplotlib
pandas
numpy
```

Install with:
```bash
pip install transformers[torch] datasets evaluate accelerate peft
```

---

## Usage
```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./my_fine_tuned_model",
    tokenizer="./my_fine_tuned_model"
)

classifier("Mahomes just threw an interception")
# [{'label': 'negative', 'score': 0.567}]
```

---

## Notes

- Confidence scores are softmax probabilities across all 3 classes. A score of 0.92 means the model assigned 92% of its probability mass to that class.
- Lower confidence on factual/descriptive tweets (e.g. game recaps) is expected and appropriate — those tweets are genuinely ambiguous in sentiment.
- The frozen-layer approach keeps training fast and reduces overfitting risk on a relatively small dataset.
