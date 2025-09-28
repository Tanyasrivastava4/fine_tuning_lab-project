# train.py
import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import wandb
import json

# ------------------------------
# 1️⃣ Load Dataset
# ------------------------------
#train_data = load_from_disk("prepared_data/train")
#test_data = load_from_disk("prepared_data/test")

train_data = load_from_disk("prepared_data/prepared_data/train")
eval_data = load_from_disk("prepared_data/prepared_data/test")

print(f"Train size: {len(train_data)}, Eval size: {len(eval_data)}")



# 2️⃣ Load Phi-3-mini tokenizer and model
# ------------------------------
model_name = "microsoft/phi-3-mini-4k-instruct"  # ✅ correct model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(set(train_data['label'])),  # number of intents
)


# ------------------------------
# 3️⃣ LoRA configuration
# ------------------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type=TaskType.SEQ_CLS
)

model = get_peft_model(model, peft_config)

# ------------------------------
# 4️⃣ Tokenize dataset
# ------------------------------
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

train_data = train_data.map(tokenize, batched=True)
test_data = test_data.map(tokenize, batched=True)

train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# ------------------------------
# 5️⃣ Training arguments
# ------------------------------
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    report_to="wandb"  # logs to W&B dashboard
)

# Initialize W&B
wandb.init(project="phi3_finetune_support_tickets")

# ------------------------------
# 6️⃣ Trainer
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer
)

# ------------------------------
# 7️⃣ Train
# ------------------------------
trainer.train()

# ------------------------------
# 8️⃣ Save model
# ------------------------------
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Fine-tuned model saved in './fine_tuned_model'")
