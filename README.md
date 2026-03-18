# Topic-Routed Sentiment Analysis of Tweets

**Authors:** Alec Zhang · David Wu · Kenneth Cao · Nolan White  
**Course Track:** Option 3 — Multiclass Identification and Sentiment Analysis  
**GitHub:** [https://github.com/aleczhang2027/L-LMs](https://github.com/aleczhang2027/L-LMs)

---

## Overview

General-purpose sentiment models treat all text the same, ignoring how word meaning shifts across domains. A tweet containing "interception" is negative in an NFL context but irrelevant in a cooking thread; "crash" signals disaster in finance but may be neutral in gaming.

This project builds a **two-stage NLP pipeline** to address this:

1. **Stage 1 — Topic Classifier:** A DistilBERT model routes each tweet to its domain (finance, politics, sports, or video games), achieving ~95%+ validation accuracy across five classes.
2. **Stage 2 — Sentiment Classifiers:** Four domain-specific fine-tuned DistilBERT models classify sentiment as positive, negative, or neutral — one per domain.

### Results at a Glance

| Domain | Test Accuracy | Macro F1 |
|---|---|---|
| Untrained DistilBERT (baseline) | ~31–33% | ~0.23 |
| Finance | 69.1% | 0.693 |
| Sports (NFL) | 73.4% | ~0.73 |
| Politics | 76.0% | ~0.76 |
| Video Games | 78.4% | 0.776 |

Polysemy analysis using t-SNE and cosine similarity heatmaps confirmed that domain-specific fine-tuning produces measurably distinct contextual embeddings for the same token across domains, validating the routing hypothesis.

---

## Motivation

Large language models learn sentiment from general English corpora, carrying those biases into every inference call. Real-world text is domain-specific: the same word can be positive, negative, or entirely neutral depending on context. Words like "bear," "draft," "goal," and "trade" carry opposite or unrelated meanings in finance versus sports versus politics. A single general-purpose model trained on mixed text learns an average representation for each token — correct for no domain in particular.

This project explores whether **explicit domain routing** (classify topic before sentiment) yields meaningfully better results than a single general-purpose classifier.

---

## Repository Structure

Each team member owns one Jupyter notebook for their Stage 2 domain. Stage 1 (topic classifier) is shared across the team.

```
L-LMs/
├── data/
│   ├── Financial_tweets.csv          # Finance (Kaggle)
│   ├── combined_df.csv               # Combined multi-domain dataset for Stage 1
│   ├── nfl_sentiments.csv            # NFL/Sports (Kaggle)
│   ├── none_sentiment.csv            # Out-of-distribution "None" class
│   ├── politics_sentiment.csv        # Politics
│   └── videogames_sentiment.csv      # Video Games (Kaggle)
├── environment_setup/                # Poetry/environment configuration files
├── notebooks/
│   ├── Sports_sentiment.ipynb        # Alec Zhang — NFL/Sports sentiment (Stage 2)
│   ├── finance_sentiment_final.ipynb # Nolan White — Finance sentiment (Stage 2)
│   ├── politics_sentiment.ipynb      # Kenneth Cao — Politics sentiment (Stage 2)
│   ├── testing_model.ipynb           # Shared — Stage 1 topic classifier & pipeline testing
│   ├── tweet_types.ipynb             # Shared — domain routing / tweet type analysis
│   └── videogames_sentiment_final.ipynb  # David Wu — Video Games sentiment (Stage 2)
├── results/
│   ├── finance_outputs/              # Finance model outputs & evaluation artifacts
│   ├── nfl_outputs/                  # NFL model outputs & evaluation artifacts
│   ├── politics_outputs/             # Politics model outputs & evaluation artifacts
│   ├── testing_model_outputs/        # Stage 1 classifier outputs
│   ├── tweet_types/                  # Domain routing analysis outputs
│   └── videogames_outputs/           # Video games model outputs & evaluation artifacts
├── .gitignore
├── LICENSE
└── README.md
```

---

## Setup

### Requirements

- **Python:** 3.12.10 (all team members downgraded to this version to resolve dependency conflicts — use this exact version)
- **Package manager:** [Poetry](https://python-poetry.org/)

### Installation

```bash
# Clone the repository
git clone https://github.com/aleczhang2027/L-LMs.git
cd L-LMs

# Install dependencies via Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Key Dependencies

All dependencies are managed through Poetry. Core packages include:

- `transformers` — HuggingFace Transformers (Trainer API, DistilBERT)
- `torch` — PyTorch backend
- `datasets` — HuggingFace Datasets
- `scikit-learn` — Evaluation metrics (accuracy, macro F1, confusion matrices)
- `matplotlib` / `seaborn` — Visualizations
- `pandas` / `numpy` — Data handling

> **Note:** All scripts were run on MacBook M-series models, except for the video games model which was run on Google Colab. For systems without CUDA or Apple Silicon support, running the scripts on Google Colab could potentially be faster.

---

## Reproducing Results

Each stage corresponds to a single Jupyter notebook. Run them in order:

### Step 1 — Train the Topic Classifier (shared)

Open and run **`notebooks/testing_model.ipynb`** and **`notebooks/tweet_types.ipynb`** top to bottom.

These train the 5-class DistilBERT classifier on the 14,115-tweet "Frankenstein" corpus and produce:
- A trained topic routing model
- t-SNE visualizations of domain embeddings
- Cosine similarity heatmaps between domain centroids
- Saliency maps for topic-specific keywords

**Data needed:** All CSVs in `data/` (see Repository Structure above).

### Step 2 — Train Domain-Specific Sentiment Models

Run each notebook independently. They do not depend on one another, but all depend on Stage 1 for the routing logic if running end-to-end inference.

| Notebook | Owner | Data File |
|---|---|---|
| `notebooks/finance_sentiment_final.ipynb` | Nolan White | `Financial_tweets.csv` |
| `notebooks/politics_sentiment.ipynb` | Kenneth Cao | `politics_sentiment.csv` |
| `notebooks/Sports_sentiment.ipynb` | Alec Zhang | `nfl_sentiments.csv` |
| `notebooks/videogames_sentiment_final.ipynb` | David Wu | `videogames_sentiment.csv` |

Each notebook is self-contained and includes all preprocessing, training, and evaluation steps. Just place the corresponding data file in `data/` and run the notebook cell by cell.

---

## Model Architecture

All models use `distilbert-base-uncased` (66M parameters) as the base:

- **Stage 1:** 5-class classification head (finance / politics / NFL / video games / none)
- **Stage 2:** 3-class classification head per domain (positive / negative / neutral)

To improve training efficiency, the embedding layer and bottom 4 of 6 transformer layers are frozen, leaving ~22% of parameters trainable (~14.7M of 66.9M).

A **confidence-based fallback** is implemented in Stage 1: if the maximum softmax probability falls below a configurable threshold, the tweet is flagged as out-of-distribution and excluded from Stage 2 routing.

### Shared Hyperparameters (all models)

| Hyperparameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Learning rate | 2e-5 |
| LR scheduler | Cosine |
| Max sequence length | 128 tokens |
| Epochs | 3 |
| Frozen layers | Embeddings + layers 0–3 |
| Training framework | HuggingFace Trainer API |

---

## Datasets

### Stage 1 — Topic Classification

Balanced corpus of 14,115 tweets across five classes (2,823 per class), with an 80/20 train/validation split.

| Domain | Source |
|---|---|
| Finance | Financial_tweets.csv (Kaggle) |
| Politics | politics_sentiment.csv |
| Sports (NFL) | nfl_sentiments.csv (Kaggle) |
| Video Games | videogames_sentiment.csv (Kaggle) |
| None (OOD) | none_sentiment.csv |

### Stage 2 — Sentiment Classification

| Domain | Training Rows | Test Rows | Notes |
|---|---|---|---|
| Finance | 7,452 | 2,484 | Balanced across 3 classes |
| Politics | 3,270 | 1,091 | Neutral class underrepresented (~10%) |
| NFL | 3,091 | 1,031 | Keyword-filtered; neutral-heavy |
| Video Games | 38,080 | 6,720 | Stratified 85/15 split; 24 game titles |

---

## Individual Contributions

**Everyone** — Stage 1 (Topic Classifier) & Stage 2 NFL support: assembled the multi-domain Frankenstein dataset; fine-tuned DistilBERT for 5-class topic classification; generated saliency maps, t-SNE visualizations, and cosine similarity heatmaps for Stage 1 domain separation. Everyone contributed equally to the presentation slideshow and project proposal.

**David Wu** — Stage 2, Video Games: filtered and preprocessed the raw tweet dataset to the video games domain; performed stratified train/test split and ensured test set isolation; fine-tuned DistilBERT for 3-class sentiment classification using HuggingFace Trainer API; designed and ran diagnostic evaluation pipeline (class distribution, confusion matrix, confidence analysis, worst-case prediction analysis); conducted baseline vs. fine-tuned comparison.

**Alec Zhang** — Stage 2, NFL: fine-tuned DistilBERT NFL sentiment classifier (73.4% test accuracy) on 5.1K tweets via parameter-efficient transfer learning (3-class head, ~78% frozen); implemented cosine LR scheduling and early stopping; validated performance using embedding diagnostics (t-SNE clustering, cosine similarity heatmaps) and saliency analysis, demonstrating improved sentiment separability and domain-specific semantic shifts (e.g., "interception" → negative).

**Kenneth Cao** — Stage 2, Politics: applied keyword-based filtering across 44 political terms to isolate 5,454 politically relevant tweets from a 3M-tweet corpus; performed stratified 60/20/20 train/val/test split; fine-tuned DistilBERT for 3-class political sentiment classification (76.06% validation accuracy); conducted before/after fine-tuning confidence comparison; performed saliency analysis on high-attribution tokens.

**Nolan White** — Stage 2 Finance: fine-tuned DistilBERT for 3-class financial sentiment classification (69.1% test accuracy, 0.693 macro F1); per-class F1 analysis (Negative 0.77, Positive 0.69, Neutral 0.63); generated confusion matrix and training/validation diagnostic curves. Authored report abstract, finance specific evaluation metrics and results sections. Edited and reviewed other portions of the report. *(Note: Due to environment issues, Nolan's notebook was uploaded to the repository by Alec Zhang.)*

---

## References

[1] Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108.
[2] ProsusAI/finbert — Financial Sentiment Analysis. Hugging Face Model Hub. https://huggingface.co/ProsusAI/finbert
[3] Twitter Sentiment Dataset — 3 million labelled rows. Kaggle. https://www.kaggle.com/datasets/prkhrawsthi/twitter-sentiment-dataset-3-million-labelled-rows
[4] Labelled Financial Tweets Dataset. Kaggle. https://www.kaggle.com/datasets/dawoodaijaz/stock-market-tweets-labelled-with-gcp-nlp
[5] jp797498e. Twitter Entity Sentiment Analysis. Kaggle. https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
[6] NFL Sentiments Dataset. Kaggle. https://www.kaggle.com/datasets/cammyc/nfl-twitter-sentiment-analysis
[7] Politics Sentiment Dataset. Kaggle — annotated political tweet corpus.
[8] Wolf, T., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. EMNLP 2020. https://aclanthology.org/2020.emnlp-demos.6
[9] Dmonte, A., et al. (2024). An Evaluation of Large Language Models in Financial Sentiment Analysis. IEEE.
[10] van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. Journal of Machine Learning Research, 9, 2579–2605.

