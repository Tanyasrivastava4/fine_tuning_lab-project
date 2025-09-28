# train.py
import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# ------------------------------
# 1️⃣ Load Dataset
# ------------------------------
train_data = load_from_disk("prepared_data/prepared_data/train")
eval_data = load_from_disk("prepared_data/prepared_data/test")

# ------------------------------
# 1.1️⃣ Remap labels to continuous [0, num_labels-1]
# ------------------------------
unique_labels = sorted(set(train_data['label']))
label2id = {label: idx for idx, label in enumerate(unique_labels)}

def remap_labels(batch):
    return {'label': label2id[batch['label']]}

train_data = train_data.map(remap_labels)
eval_data = eval_data.map(remap_labels)

print(f"Train size: {len(train_data)}, Eval size: {len(eval_data)}")
print("Train labels:", set(train_data['label']))
print("Eval labels:", set(eval_data['label']))

# ------------------------------
# 2️⃣ Load Phi-3-mini tokenizer and model
# ------------------------------
model_name = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(unique_labels),  # number of classes
)

# ------------------------------
# 3️⃣ LoRA configuration
# ------------------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
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
eval_data = eval_data.map(tokenize, batched=True)

train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
eval_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# ------------------------------
# 5️⃣ Training arguments
# ------------------------------
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    report_to="wandb"
)

# ------------------------------
# 6️⃣ Initialize W&B
# ------------------------------
wandb.login()  # will prompt for API key if needed
wandb.init(project="phi3_finetune_support_tickets")

# ------------------------------
# 7️⃣ Trainer
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer
)

# ------------------------------
# 8️⃣ Start training
# ------------------------------
trainer.train()

# ------------------------------
# 9️⃣ Save model
# ------------------------------
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
print("Fine-tuned model saved in './fine_tuned_model'")


# Merge LoRA weights into base model and save full model
# ------------------------------
base_model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/phi-3-mini-4k-instruct",
    num_labels=num_labels  # make sure num_labels matches your training labels
)

# Load the trained PEFT/LoRA adapter
peft_model = PeftModel.from_pretrained(base_model, "./fine_tuned_model")

# Merge LoRA weights into base model
full_model = peft_model.merge_and_unload()

# Save merged model and tokenizer
full_model.save_pretrained("./fine_tuned_full_model")
tokenizer.save_pretrained("./fine_tuned_full_model")

print("Merged full model saved to ./fine_tuned_full_model")
