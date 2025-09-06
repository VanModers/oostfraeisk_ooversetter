import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 

import logging
import time

import numpy as np

from dataset_creator import get_dataset
from transformers import MarianMTModel, MarianTokenizer
import torch

# Load model and tokenizer
model_path = "./de_frs_model_5"
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

# Move model to GPU (if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
model.to(device)

# Translate function
def translate(text):
    inputs = tokenizer(text, return_tensors="pt")
    inputs.to(device)
    translated = model.generate(**inputs)

    # Decode output
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

frs = open("data/fan teksten/news/eastfrisian.txt", "a", encoding="utf-8")
ger = open("data/fan teksten/news/german.txt", "a", encoding="utf-8")

gerSentences = open("data/fan teksten/news/deu.txt", "r", encoding="utf-8")
lines = gerSentences.readlines()
gerSentences.close()

offset = len(lines) - 10

str = ""
while str != "end" and offset < len(lines):
    #gerString = lines[offset].split("\t")[1]
    gerString = lines[offset].rstrip()
    print(gerString)
    if gerString:
        frsString = translate(gerString)
        print(frsString)
        str = input("Is dat fertóólen gaud? (y/n) ")
        if str == "y":
            del lines[offset]
            print(gerString, file=ger)
            print(frsString, file=frs)
        else:
            if str == "n":
                frsString = input("Räecht fertóólen: ")
                if frsString != "":
                    del lines[offset]
                    print(gerString, file=ger)
                    print(frsString, file=frs)

frs.close()
ger.close()

gerSentences = open("data/fan teksten/news/deu.txt", "w", encoding="utf-8")
gerSentences.writelines(lines)
gerSentences.close()
