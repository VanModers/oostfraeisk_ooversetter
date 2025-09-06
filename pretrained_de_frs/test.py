import evaluate
import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Load model and tokenizer
model_path = "./frisian_model_1"
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

# Move model to GPU (if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model.to(device)

def compute_metrics_1():
    inputs = tokenizer("Es gab einst jemanden, den ich mochte.".lower(), return_tensors="pt", padding=True, truncation=True)
    inputs.to(device)

    pred_ids = model.generate(**inputs)
    pred_ids = pred_ids[0][1:]

    labels = tokenizer("Dat gaf aleer wel däi 'k muğ.", return_tensors="pt", padding=True, truncation=True)["input_ids"][0]

    # Ensure both tensors have the same shape
    max_len = max(pred_ids.shape[0], labels.shape[0])
    
    pred_ids = torch.nn.functional.pad(pred_ids, (0, max_len - pred_ids.shape[0]), value=tokenizer.pad_token_id)
    labels = torch.nn.functional.pad(labels, (0, max_len - labels.shape[0]), value=tokenizer.pad_token_id)

    print(pred_ids)
    print(labels)

    # Mask out padding tokens
    mask = labels != tokenizer.pad_token_id
    correct = (pred_ids == labels) & mask

    # Convert to float explicitly
    accuracy = correct.sum().item() / mask.sum().item()

    return {"accuracy": accuracy}

metric = evaluate.load("accuracy")

def compute_metrics_2():
    inputs = tokenizer("Es gab einst jemanden, den ich mochte.".lower(), return_tensors="pt", padding=True, truncation=True)
    inputs.to(device)

    pred_ids = model.generate(**inputs)

    pred_ids = pred_ids[0][1:]

    labels = tokenizer("Dat gaf aleer wel däi 'k muğ.", return_tensors="pt", padding=True, truncation=True)["input_ids"][0]

    # Ensure both tensors have the same shape
    max_len = max(pred_ids.shape[0], labels.shape[0])

    pred_ids = torch.nn.functional.pad(pred_ids, (0, max_len - pred_ids.shape[0]), value=tokenizer.pad_token_id)
    labels = torch.nn.functional.pad(labels, (0, max_len - labels.shape[0]), value=tokenizer.pad_token_id)

    result = metric.compute(predictions=pred_ids, references=labels)
    return result

print(compute_metrics_1())

print(compute_metrics_2())