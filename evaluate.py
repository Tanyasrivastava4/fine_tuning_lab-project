# evaluate.py
import json
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix
import wandb

# ------------------------------
# 1️⃣ Init Weights & Biases
# ------------------------------
wandb.init(project="phi3_finetune_support_tickets", name="evaluation")

# ------------------------------
# 2️⃣ Load merged fine-tuned model and tokenizer
# ------------------------------
model_path = "./fine_tuned_full_model"  # merged model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# ------------------------------
# 3️⃣ Load test dataset
# ------------------------------
test_dataset = load_from_disk("prepared_data/test")

# ------------------------------
# 4️⃣ Make predictions
# ------------------------------
preds = []
labels = test_dataset["label"]

for example in test_dataset:
    inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_label = torch.argmax(outputs.logits, dim=-1).item()
    preds.append(pred_label)

# ------------------------------
# 5️⃣ Compute metrics
# ------------------------------
acc = accuracy_score(labels, preds)
cm = confusion_matrix(labels, preds).tolist()  # convert to list for JSON

metrics = {
    "accuracy": acc,
    "confusion_matrix": cm
}

# ------------------------------
# 6️⃣ Save metrics to JSON
# ------------------------------
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# ------------------------------
# 7️⃣ Log metrics to W&B
# ------------------------------
wandb.log({
    "eval_accuracy": acc,
    "confusion_matrix": wandb.plot.confusion_matrix(
        y_true=labels,
        preds=preds,
        title="Confusion Matrix"
    )
})

wandb.finish()

print(f"✅ Evaluation complete. Accuracy: {acc:.4f}")
print("📊 Metrics saved to metrics.json and logged to Weights & Biases")
