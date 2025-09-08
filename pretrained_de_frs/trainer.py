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
model_name = "Helsinki-NLP/opus-mt-de-nl"  # German to Dutch
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Initialize the data collator for padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./de_frs_model",
    eval_strategy="epoch",  # Evaluate after every epoch
    save_strategy="no",  # Save model after every epoch
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=7,
    save_total_limit=2,
    logging_dir="./logs",
    predict_with_generate=True,  # Ensure generation during evaluation
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
    tokenizer=tokenizer,
    data_collator=data_collator,  # Add the data collator here
    compute_metrics=compute_metrics,  # Pass the compute_metrics function here
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("./de_frs_model")
tokenizer.save_pretrained("./de_frs_model")