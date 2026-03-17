import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE

TEST_PATH   = "/content/videogames_test.csv"
MODEL_PATH  = "/content/video_games_model"
RESULTS_DIR = "/content/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
COLORS   = {0: "red", 1: "gray", 2: "green"}
NAMES    = {0: "Negative", 1: "Neutral", 2: "Positive"}

SAMPLE_SIZE = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

print("\nLoading test data...")
test_df = pd.read_csv(TEST_PATH).sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
true_labels = test_df["sentiment"].map(LABEL2ID).tolist()

class TweetDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(list(texts), truncation=True, max_length=128, padding=True, return_tensors="pt")
    def __len__(self):
        return self.encodings["input_ids"].shape[0]
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items() if k != "token_type_ids"}

def extract_embeddings(model, tokenizer, texts, batch_size=64):
    dataset = TweetDataset(texts, tokenizer)
    loader  = DataLoader(dataset, batch_size=batch_size)
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model.distilbert(**batch)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

def plot_tsne(embeddings, labels, title, save_path):
    print(f"  Running t-SNE for: {title}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    reduced = tsne.fit_transform(embeddings)
    plt.figure(figsize=(9, 7))
    for label_id, label_name in NAMES.items():
        mask = np.array(labels) == label_id
        plt.scatter(reduced[mask, 0], reduced[mask, 1],
                    c=COLORS[label_id], label=label_name, alpha=0.6, s=15)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")

texts = test_df["text"].astype(str).tolist()

print("\n[1/2] Extracting base model embeddings...")
base_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
base_model     = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
base_model.to(device)
base_embeddings = extract_embeddings(base_model, base_tokenizer, texts)
plot_tsne(base_embeddings, true_labels,
          "t-SNE — Base DistilBERT (Untrained)",
          f"{RESULTS_DIR}/7_tsne_base_model.png")

del base_model

print("\n[2/2] Extracting fine-tuned model embeddings...")
ft_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
ft_model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
ft_model.to(device)
ft_embeddings = extract_embeddings(ft_model, ft_tokenizer, texts)
plot_tsne(ft_embeddings, true_labels,
          "t-SNE — Fine-Tuned DistilBERT (Video Games)",
          f"{RESULTS_DIR}/7_tsne_finetuned_model.png")

print("\nDone! Both t-SNE plots saved to /content/results/")
