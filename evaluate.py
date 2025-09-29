# evaluate.py
import json
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
import wandb
import pandas as pd

# ------------------------------
# 1️⃣ Initialize W&B
# ------------------------------
wandb.init(project="phi3_finetune_support_tickets", name="evaluation")

# ------------------------------
# 2️⃣ Load merged fine-tuned model and tokenizer
# ------------------------------
model_path = "./fine_tuned_full_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# ------------------------------
# 3️⃣ Move model to GPU if available
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# ------------------------------
# 4️⃣ Load test dataset
# ------------------------------
test_dataset = load_from_disk("prepared_data/prepared_data/test")

# ------------------------------
# 5️⃣ Tokenize test dataset
# ------------------------------
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

test_dataset = test_dataset.map(tokenize, batched=True)

# ------------------------------
# 6️⃣ Set format for PyTorch
# ------------------------------
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# ------------------------------
# 7️⃣ Create DataLoader for batched inference
# ------------------------------
batch_size = 32  # adjust based on GPU memory
dataloader = DataLoader(test_dataset, batch_size=batch_size)

preds = []
labels = []

# ------------------------------
# 8️⃣ Batched predictions
# ------------------------------
for batch in dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_preds = torch.argmax(outputs.logits, dim=-1)
        preds.extend(batch_preds.cpu().tolist())
        labels.extend(batch['label'].tolist())

# ------------------------------
# 9️⃣ Compute metrics
# ------------------------------
acc = accuracy_score(labels, preds)
cm = confusion_matrix(labels, preds).tolist()

metrics = {
    "accuracy": acc,
    "confusion_matrix": cm
}

# ------------------------------
# 🔟 Save metrics to JSON
# ------------------------------
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# ------------------------------
# 1️⃣1️⃣ Log metrics to W&B (memory-safe)
# ------------------------------
# Log only accuracy (very lightweight)
wandb.log({"eval_accuracy": acc})

# Optional: log confusion matrix for first 200 predictions only
sample_size = 200
sample_cm = confusion_matrix(labels[:sample_size], preds[:sample_size])
wandb.log({"confusion_matrix_sample": wandb.Table(data=sample_cm.tolist())})

wandb.finish()

print(f"✅ Evaluation complete. Accuracy: {acc:.4f}")
print("📊 Metrics saved to metrics.json and logged to Weights & Biases")
