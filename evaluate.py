# evaluate.py
import json
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
import wandb

# ------------------------------
# 1️⃣ Initialize W&B
# ------------------------------
wandb.init(project="phi3_finetune_support_tickets", name="evaluation")

# ------------------------------
# 2️⃣ Load tokenizer from fine-tuned adapter
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

# ------------------------------
# 3️⃣ Load base model with correct number of labels
# ------------------------------
num_labels = 100  # same as training
base_model_name = "microsoft/phi-3-mini-4k-instruct"
base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name,
    num_labels=num_labels
)

# ------------------------------
# 4️⃣ Load PEFT LoRA adapter safely
# ------------------------------
model = PeftModel.from_pretrained(base_model, "./fine_tuned_model")
model.eval()

# ------------------------------
# 5️⃣ Move model to GPU if available
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ------------------------------
# 6️⃣ Load test dataset
# ------------------------------
test_dataset = load_from_disk("prepared_data/prepared_data/test")

# ------------------------------
# 7️⃣ Tokenize test dataset
# ------------------------------
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

test_dataset = test_dataset.map(tokenize, batched=True)

# ------------------------------
# 8️⃣ Set format for PyTorch
# ------------------------------
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# ------------------------------
# 9️⃣ Create DataLoader for batched inference
# ------------------------------
batch_size = 32  # adjust based on GPU memory
dataloader = DataLoader(test_dataset, batch_size=batch_size)

preds = []
labels = []

# ------------------------------
# 🔟 Batched predictions
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
# 1️⃣1️⃣ Compute metrics
# ------------------------------
acc = accuracy_score(labels, preds)
cm = confusion_matrix(labels, preds).tolist()

metrics = {
    "accuracy": acc,
    "confusion_matrix": cm
}

# ------------------------------
# 1️⃣2️⃣ Save metrics to JSON
# ------------------------------
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# ------------------------------
# 1️⃣3️⃣ Log metrics to W&B
# ------------------------------
wandb.log({"eval_accuracy": acc})

# Optional: log confusion matrix for first 200 predictions
#import pandas as pd
#sample_size = 200
#sample_cm = confusion_matrix(labels[:sample_size], preds[:sample_size])
#wandb.log({"confusion_matrix_sample": wandb.Table(data=sample_cm)})

import matplotlib.pyplot as plt
import seaborn as sns

sample_size = 200
sample_cm = confusion_matrix(labels[:sample_size], preds[:sample_size])

plt.figure(figsize=(6, 5))
sns.heatmap(sample_cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Sample")
plt.tight_layout()

wandb.log({"confusion_matrix_sample": wandb.Image(plt)})
plt.close()

wandb.finish()

print(f"✅ Evaluation complete. Accuracy: {acc:.4f}")
print("📊 Metrics saved to metrics.json and logged to Weights & Biases")


# Use this code when you are not able to see the output in wandb clearly.
# evaluate_wandb.py
#import json
#import torch
#from datasets import load_from_disk
#from transformers import AutoTokenizer, AutoModelForSequenceClassification
#from peft import PeftModel
#from sklearn.metrics import accuracy_score, confusion_matrix
#from torch.utils.data import DataLoader
#import wandb
#import matplotlib.pyplot as plt
#import seaborn as sns

# 1️⃣ Initialize W&B project
#wandb.init(project="phi3_finetune_support_tickets", name="evaluation_full")

# 2️⃣ Load tokenizer and base model
#tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
#num_labels = 100  # same as training
#base_model_name = "microsoft/phi-3-mini-4k-instruct"
#base_model = AutoModelForSequenceClassification.from_pretrained(
 #   base_model_name,
  #  num_labels=num_labels
#)

# 3️⃣ Load PEFT LoRA adapter safely
#model = PeftModel.from_pretrained(base_model, "./fine_tuned_model")
#model.eval()

# 4️⃣ Move model to GPU if available
#device = "cuda" if torch.cuda.is_available() else "cpu"
#model.to(device)

# 5️⃣ Load test dataset
#test_dataset = load_from_disk("prepared_data/prepared_data/test")

# 6️⃣ Tokenize test dataset
#def tokenize(batch):
 #   return tokenizer(batch['text'], padding=True, truncation=True)

#test_dataset = test_dataset.map(tokenize, batched=True)

# 7️⃣ Set format for PyTorch
#test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 8️⃣ Create DataLoader
#batch_size = 32
#dataloader = DataLoader(test_dataset, batch_size=batch_size)

#preds = []
#labels = []

# 9️⃣ Batched predictions
#for batch in dataloader:
 #   input_ids = batch['input_ids'].to(device)
  #  attention_mask = batch['attention_mask'].to(device)
   # with torch.no_grad():
    #    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
     #   batch_preds = torch.argmax(outputs.logits, dim=-1)
      #  preds.extend(batch_preds.cpu().tolist())
       # labels.extend(batch['label'].tolist())

# 🔟 Compute metrics
#acc = accuracy_score(labels, preds)
#cm = confusion_matrix(labels, preds).tolist()

#metrics = {
 #   "accuracy": acc,
  #  "confusion_matrix": cm
#}

# 1️⃣1️⃣ Save metrics to JSON
#with open("metrics.json", "w") as f:
 #   json.dump(metrics, f, indent=4)

# 1️⃣2️⃣ Log metrics to W&B
#wandb.log({"eval_accuracy": acc})

# 1️⃣3️⃣ Log confusion matrix as an image
#plt.figure(figsize=(8, 6))
#sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
#plt.xlabel("Predicted")
#plt.ylabel("Actual")
#plt.title("Confusion Matrix")
#plt.tight_layout()
#wandb.log({"confusion_matrix": wandb.Image(plt)})
#plt.close()

# 1️⃣4️⃣ Log sample predictions (first 50 examples)
#sample_size = 50
#data = list(zip(labels[:sample_size], preds[:sample_size]))
#table = wandb.Table(data=data, columns=["Actual Label", "Predicted Label"])
#wandb.log({"sample_predictions": table})

#wandb.finish()
#print(f"✅ Evaluation complete. Accuracy: {acc:.4f}")
#print("📊 Metrics, confusion matrix, and sample predictions logged to W&B")
