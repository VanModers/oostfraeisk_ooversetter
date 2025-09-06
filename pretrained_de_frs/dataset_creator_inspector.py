import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import MarianTokenizer

from dataset_creator import get_dataset


model_name = "Helsinki-NLP/opus-mt-de-en"  # German to English (no Frisian yet)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Load dataset
dataset = get_dataset("data", tokenizer)
train_ds, val_ds = dataset["train"], dataset["validation"]

for data in val_ds:
    ger = data["input_ids"]
    frs = data["labels"]
    print(tokenizer.decode(ger, skip_special_tokens=True) + "\n"  + tokenizer.decode(frs, skip_special_tokens=True))
    user_input = input("")
    if user_input.lower() == "end":
        break