import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

TEST_PATH  = "/content/videogames_test.csv"
RESULTS_DIR = "/content/results"

import os
os.makedirs(RESULTS_DIR, exist_ok=True)

LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
LABEL_NAMES = ["Negative", "Neutral", "Positive"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

print("\nLoading test data...")
test_df = pd.read_csv(TEST_PATH)
true_labels = test_df["sentiment"].map(LABEL2ID).tolist()

print("Loading base DistilBERT (untrained)...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
model.eval()
model.to(device)

class TweetDataset(Dataset):
    def __init__(self, texts):
        self.encodings = tokenizer(list(texts), truncation=True, max_length=128, padding=True, return_tensors="pt")
    def __len__(self):
        return self.encodings["input_ids"].shape[0]
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items() if k != "token_type_ids"}

print("Running predictions on test set...")
dataset = TweetDataset(test_df["text"].astype(str).tolist())
loader  = DataLoader(dataset, batch_size=64)

all_preds = []
with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)

accuracy = accuracy_score(true_labels, all_preds)
macro_f1 = f1_score(true_labels, all_preds, average="macro")

print("\n" + "="*60)
print("BASE MODEL (UNTRAINED) EVALUATION RESULTS")
print("="*60)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Macro F1:  {macro_f1:.4f}")
print("\nClassification Report:")
print(classification_report(true_labels, all_preds, target_names=LABEL_NAMES))
print("="*60)

output = (
    f"BASE MODEL (UNTRAINED) EVALUATION RESULTS\n"
    f"{'='*60}\n"
    f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n"
    f"Macro F1:  {macro_f1:.4f}\n\n"
    f"Classification Report:\n"
    f"{classification_report(true_labels, all_preds, target_names=LABEL_NAMES)}"
)

out_path = f"{RESULTS_DIR}/6_base_model_eval.txt"
with open(out_path, "w") as f:
    f.write(output)
print(f"\nSaved → {out_path}")
