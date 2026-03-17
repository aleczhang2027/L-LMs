import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

FINETUNED_MODEL_PATH = "/content/video_games_model"
RESULTS_DIR = "/content/results"

import os
os.makedirs(RESULTS_DIR, exist_ok=True)

SAMPLE_TWEETS = [
    "Fortnite just dropped the best update ever, I can't stop playing!",
    "This new Cyberpunk patch is absolute garbage, nothing works.",
    "League of Legends is having server maintenance today.",
    "I just got a victory royale in Fortnite, absolute peak gaming.",
    "Dota 2 matchmaking is completely broken, every game is unplayable.",
]

TWEET_LABELS = [
    "Fortnite update",
    "Cyberpunk patch",
    "LoL maintenance",
    "Fortnite victory",
    "Dota 2 matchmaking",
]

ID2LABEL_BASE      = {0: "NEGATIVE", 1: "POSITIVE"}
ID2LABEL_FINETUNED = {0: "Negative", 1: "Neutral", 2: "Positive"}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}\n")

print("Loading base DistilBERT...")
base_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
base_model.eval()
base_model.to(device)

print("Loading fine-tuned video games model...")
ft_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
ft_model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_MODEL_PATH)
ft_model.eval()
ft_model.to(device)

results = []

for tweet in SAMPLE_TWEETS:
    base_inputs = base_tokenizer(tweet, return_tensors="pt", truncation=True, max_length=128, padding=True)
    base_inputs = {k: v.to(device) for k, v in base_inputs.items() if k != "token_type_ids"}
    with torch.no_grad():
        base_outputs = base_model(**base_inputs)
    base_probs = softmax(base_outputs.logits, dim=-1).squeeze()
    base_pred_id = torch.argmax(base_probs).item()
    base_label = ID2LABEL_BASE[base_pred_id]
    base_conf  = base_probs[base_pred_id].item()

    ft_inputs = ft_tokenizer(tweet, return_tensors="pt", truncation=True, max_length=128, padding=True)
    ft_inputs = {k: v.to(device) for k, v in ft_inputs.items() if k != "token_type_ids"}
    with torch.no_grad():
        ft_outputs = ft_model(**ft_inputs)
    ft_probs = softmax(ft_outputs.logits, dim=-1).squeeze()
    ft_pred_id = torch.argmax(ft_probs).item()
    ft_label = ID2LABEL_FINETUNED[ft_pred_id]
    ft_conf  = ft_probs[ft_pred_id].item()

    results.append({
        "tweet": tweet,
        "base_label": base_label,
        "base_conf": base_conf,
        "ft_label": ft_label,
        "ft_conf": ft_conf,
    })

lines = ["BEFORE vs AFTER FINE-TUNING COMPARISON", "="*80]
for r in results:
    lines.append(f"\nTweet:      {r['tweet']}")
    lines.append(f"  Base model:       {r['base_label']:<10} (confidence: {r['base_conf']:.4f})")
    lines.append(f"  Fine-tuned model: {r['ft_label']:<10} (confidence: {r['ft_conf']:.4f})")
lines.append("="*80)

output_text = "\n".join(lines)
print(output_text)

txt_path = f"{RESULTS_DIR}/5_baseline_comparison.txt"
with open(txt_path, "w") as f:
    f.write(output_text)
print(f"\nSaved → {txt_path}")

x = np.arange(len(TWEET_LABELS))
width = 0.35

base_confs = [r["base_conf"] for r in results]
ft_confs   = [r["ft_conf"]   for r in results]
ft_labels  = [r["ft_label"]  for r in results]

colors = []
for r in results:
    if r["ft_label"] == "Positive":
        colors.append("green")
    elif r["ft_label"] == "Negative":
        colors.append("red")
    else:
        colors.append("gray")

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, base_confs, width, label="Base Model", color="lightblue", edgecolor="black")
bars2 = ax.bar(x + width/2, ft_confs,   width, label="Fine-Tuned Model", color=colors, edgecolor="black", alpha=0.8)

for bar, label in zip(bars2, ft_labels):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            label, ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xlabel("Tweet")
ax.set_ylabel("Confidence Score")
ax.set_title("Base vs Fine-Tuned DistilBERT — Confidence Comparison")
ax.set_xticks(x)
ax.set_xticklabels(TWEET_LABELS, rotation=15, ha="right")
ax.set_ylim(0, 1.1)
ax.axhline(y=0.5, color="black", linestyle="--", linewidth=0.8, label="Random baseline (0.5)")
ax.legend()
plt.tight_layout()

chart_path = f"{RESULTS_DIR}/5_baseline_comparison.png"
plt.savefig(chart_path)
print(f"Saved → {chart_path}")
