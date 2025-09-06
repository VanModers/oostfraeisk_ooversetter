import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import MarianTokenizer

class TranslationDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        print("Loading dataset")
        self.examples = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(os.path.join(path, "german.txt"), "r", encoding="utf-8") as ger_file, \
             open(os.path.join(path, "eastfrisian.txt"), "r", encoding="utf-8") as frs_file:
            
            for ger, frs in zip(ger_file, frs_file):
                self.examples.append(ger.strip())
                self.labels.append(frs.strip())
        
        print("Finished loading dataset")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Tokenizing both the example and the label
        example = self.examples[idx]
        label = self.labels[idx]
        
        # Tokenize the example (German)
        input_encodings = self.tokenizer(example, truncation=True, padding='max_length', max_length=self.max_length, return_tensors=None)
        
        # Tokenize the label (East Frisian)
        label_encodings = self.tokenizer(label, truncation=True, padding='max_length', max_length=self.max_length, return_tensors=None)
        
        # Flatten the tensors, as they are returned as batches of size 1
        #input_encodings = {key: val.squeeze(0) for key, val in input_encodings.items()}
        #label_encodings = {key: val.squeeze(0) for key, val in label_encodings.items()}
        
        # Return as dictionaries (input_ids, attention_mask, etc.)
        return {"input_ids": input_encodings["input_ids"],
                "attention_mask": input_encodings["attention_mask"],
                "labels": label_encodings["input_ids"]}
    

def split_dataset(dataset, val_size=1000):
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    return {"train": train_ds, "validation": val_ds}

def get_dataset(path, tokenizer):
    dataset = TranslationDataset(path, tokenizer)
    return split_dataset(dataset)