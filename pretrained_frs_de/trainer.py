import torch
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import evaluate
from dataset_creator import get_dataset

# Define the accuracy metric
accuracy_metric = evaluate.load("accuracy")

import numpy as np

def compute_metrics(pred):
    pred_ids = pred.predictions
    labels = pred.label_ids

    # Convert numpy arrays to torch tensors
    pred_ids = torch.tensor(pred_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    pred_ids = pred_ids[:, 1:]

    # Ensure both tensors have the same shape
    max_len = max(pred_ids.shape[1], labels.shape[1])  # Get max sequence length
    
    pred_ids = torch.nn.functional.pad(pred_ids, (0, max_len - pred_ids.shape[1]), value=tokenizer.pad_token_id)
    labels = torch.nn.functional.pad(labels, (0, max_len - labels.shape[1]), value=tokenizer.pad_token_id)

    # Mask out padding tokens
    mask = labels != tokenizer.pad_token_id
    correct = (pred_ids == labels) & mask

    # Calculate accuracy
    accuracy = correct.sum().item() / mask.sum().item()

    return {"accuracy": accuracy}

# Define the model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-de"  # English to German
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Initialize the data collator for padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./frs_de_model",
    eval_strategy="epoch",  # Evaluate after every epoch
    save_strategy="no",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=7,
    save_total_limit=2,
    logging_dir="./logs",
    predict_with_generate=True,  # Ensure generation during evaluation
    bf16=True,  # Use bfloat16 mixed precision
    dataloader_num_workers=4,  # Parallel data loading
    dataloader_pin_memory=True,  # Faster CPU->GPU transfer
    torch_compile=True,  # PyTorch 2.0+ compilation for speed
    optim="adamw_torch_fused",  # Fused AdamW optimizer (faster on GPU)
    gradient_checkpointing=False,  # Disable for speed (enable if OOM)
)

# Load your dataset
dataset = get_dataset("data", tokenizer)
train_ds, val_ds = dataset["train"], dataset["validation"]

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=data_collator,  # Add the data collator here
    compute_metrics=compute_metrics,  # Pass the compute_metrics function here
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("./frs_de_model")
tokenizer.save_pretrained("./frs_de_model")