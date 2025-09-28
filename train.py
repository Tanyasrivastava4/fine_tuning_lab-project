# train.py
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch
import wandb

# ------------------------------
# 1️⃣ Load dataset
# ------------------------------
train_dataset = load_from_disk("prepared_data/train")
eval_dataset = load_from_disk("prepared_data/val")

# Determine number of labels dynamically
num_labels = len(set(train_dataset['label']))

# ------------------------------
# 2️⃣ Load tokenizer and base model
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")
base_model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/phi-3-mini-4k-instruct",
    num_labels=num_labels
)

# ------------------------------
# 3️⃣ Apply LoRA/PEFT
# ------------------------------
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    target_modules=["score"],  # adjust based on your model
    lora_dropout=0.05,
    bias="none",
    fan_in_fan_out=False
)

model = get_peft_model(base_model, lora_config)

# ------------------------------
# 4️⃣ Training arguments
# ------------------------------
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=False,
    report_to="wandb"
)

# ------------------------------
# 5️⃣ Initialize Trainer
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# ------------------------------
# 6️⃣ Train model
# ------------------------------
trainer.train()

# Save PEFT/adapter model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# ------------------------------
# 7️⃣ Merge LoRA weights into base model
# ------------------------------
base_model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/phi-3-mini-4k-instruct",
    num_labels=num_labels
)
peft_model = PeftModel.from_pretrained(base_model, "./fine_tuned_model")
full_model = peft_model.merge_and_unload()
full_model.save_pretrained("./fine_tuned_full_model")
tokenizer.save_pretrained("./fine_tuned_full_model")

print("✅ Training complete. PEFT model saved to './fine_tuned_model'.")
print("✅ Merged full model saved to './fine_tuned_full_model'.")
