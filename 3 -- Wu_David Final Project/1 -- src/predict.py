import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "2 -- models/video_games_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def predict(tweet: str):
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze()
    predicted_id = torch.argmax(probs).item()
    predicted_label = model.config.id2label[predicted_id]
    confidence = probs[predicted_id].item()

    return {
        "tweet": tweet,
        "sentiment": predicted_label,
        "confidence": round(confidence, 4),
        "scores": {
            model.config.id2label[i]: round(probs[i].item(), 4)
            for i in range(len(probs))
        }
    }

if __name__ == "__main__":
    test_tweets = [
        "Fortnite just dropped the best update ever, I can't stop playing!",
        "This new Cyberpunk patch is absolute garbage, nothing works.",
        "League of Legends is having server maintenance today.",
    ]

    for tweet in test_tweets:
        result = predict(tweet)
        print(f"\nTweet:      {result['tweet']}")
        print(f"Sentiment:  {result['sentiment']} (confidence: {result['confidence']})")
        print(f"Scores:     {result['scores']}")
