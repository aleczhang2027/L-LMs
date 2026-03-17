import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset

TRAIN_PATH  = "/content/videogames_train.csv"
TEST_PATH   = "/content/videogames_test.csv"
MODEL_PATH  = "/content/video_games_model"
RESULTS_DIR = "/content/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL  = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_NAMES = ["Negative", "Neutral", "Positive"]

print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Running on: {device}")

# Image 1

print("\n[1/4] Plotting class distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
train_df["sentiment"].value_counts().plot(kind="bar", ax=axes[0], color=["red","gray","green"], title="Train Set Class Distribution")
test_df["sentiment"].value_counts().plot(kind="bar",  ax=axes[1], color=["red","gray","green"], title="Test Set Class Distribution")
for ax in axes:
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/1_class_distribution.png")
print(f"  Saved → {RESULTS_DIR}/1_class_distribution.png")
print("  Train distribution:\n", train_df["sentiment"].value_counts())
print("  Test distribution:\n",  test_df["sentiment"].value_counts())

# Batch prediction helper 

class TweetDataset(Dataset):
    def __init__(self, texts):
        self.encodings = tokenizer(list(texts), truncation=True, max_length=128, padding=True, return_tensors="pt")
    def __len__(self):
        return self.encodings["input_ids"].shape[0]
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items() if k != "token_type_ids"}

def batch_predict(df, batch_size=64):
    dataset = TweetDataset(df["text"].astype(str).tolist())
    loader  = DataLoader(dataset, batch_size=batch_size)
    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            all_preds.extend(preds)
            all_probs.extend(probs)
    return all_preds, all_probs

# Image 2

print("\n[2/4] Running predictions for confusion matrix...")
preds, probs = batch_predict(test_df)
true_labels  = test_df["sentiment"].map(LABEL2ID).tolist()

cm = confusion_matrix(true_labels, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, cmap="Blues")
plt.title("Confusion Matrix — Test Set")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/2_confusion_matrix.png")
print(f"  Saved → {RESULTS_DIR}/2_confusion_matrix.png")
print("\n  Classification Report:")
print(classification_report(true_labels, preds, target_names=LABEL_NAMES))

# Image 3

print("\n[3/4] Plotting confidence distribution...")
probs_arr = np.array(probs)
pred_arr  = np.array(preds)
plt.figure(figsize=(10, 4))
for i, label in ID2LABEL.items():
    mask = pred_arr == i
    if mask.sum() > 0:
        plt.hist(probs_arr[mask, i], bins=20, alpha=0.6, label=f"{label} (n={mask.sum()})")
plt.title("Prediction Confidence Distribution by Class")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/3_confidence_distribution.png")
print(f"  Saved → {RESULTS_DIR}/3_confidence_distribution.png")

# Image 4
print("\n[4/4] Finding worst predictions...")
results_df = test_df.copy().reset_index(drop=True)
results_df["predicted"] = [ID2LABEL[p] for p in preds]
results_df["confidence"] = [probs_arr[i, preds[i]] for i in range(len(preds))]
results_df["correct"] = results_df["sentiment"] == results_df["predicted"]

wrong_df = results_df[~results_df["correct"]].sort_values("confidence", ascending=False)
wrong_df[["text", "sentiment", "predicted", "confidence"]].head(20).to_csv(f"{RESULTS_DIR}/4_worst_predictions.csv", index=False)
print(f"  Saved → {RESULTS_DIR}/4_worst_predictions.csv")
print(f"\n  Total wrong: {len(wrong_df)} / {len(results_df)}")
print("\n  Top 10 high-confidence wrong predictions:")
pd.set_option("display.max_colwidth", 80)
print(wrong_df[["text","sentiment","predicted","confidence"]].head(10).to_string(index=False))

print("\nAll diagnostics complete!")
print(f"Charts saved to: {RESULTS_DIR}")
