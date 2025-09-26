# evaluate.py
import json
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------------------
# 1️⃣ Load fine-tuned model and tokenizer
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
model.eval()

# ------------------------------
# 2️⃣ Load test dataset
# ------------------------------
test_data = load_from_disk("prepared_data/test")

# ------------------------------
# 3️⃣ Make predictions
# ------------------------------
preds = []
labels = test_data['label']

for example in test_data:
    inputs = tokenizer(example['text'], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        pred_label = torch.argmax(outputs.logits, dim=-1).item()
    preds.append(pred_label)

# ------------------------------
# 4️⃣ Compute metrics
# ------------------------------
acc = accuracy_score(labels, preds)
cm = confusion_matrix(labels, preds).tolist()  # convert to list for JSON

metrics = {
    "accuracy": acc,
    "confusion_matrix": cm
}

# ------------------------------
# 5️⃣ Save metrics to JSON
# ------------------------------
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Evaluation complete. Accuracy: {acc}")
print("Metrics saved to metrics.json")
