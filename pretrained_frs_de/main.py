import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Load model and tokenizer
model_path = "./frs_de_model"
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

# Move model to GPU (if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model.to(device)

# Translate function
def translate(text):
    inputs = tokenizer(text, return_tensors="pt")
    inputs.to(device)
    translated = model.generate(**inputs)

    # Decode output
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Example translation
print(translate("Hallo, wie geht es meinem Freund?"))

# Interactive translation loop
while True:
    user_input = input("Enter text to translate (or 'end' to exit): ")
    if user_input.lower() == "end":
        break
    print("Translated:", translate(user_input))