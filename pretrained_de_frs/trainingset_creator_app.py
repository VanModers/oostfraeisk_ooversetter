import gradio as gr
import os
import random
import torch
from transformers import MarianMTModel, MarianTokenizer

# Load model and tokenizer
model_path = "frisian_model_3"  # Change this if using HF storage
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

# Move model to CPU for hosting
device = "cpu"
model.to(device)

# Translation function
def translate(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# File paths
deu_file = "data/krektüren/deu.txt"
ger_output = "data/krektüren/german.txt"
frs_output = "data/krektüren/eastfrisian.txt"

# Load sentences
with open(deu_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Choose a random sentence
offset = random.randrange(0, len(lines) - 1) if len(lines) > 1 else 0

def correct_translation():
    """Fetch a new German sentence and translate it."""
    global offset, lines
    if offset >= len(lines):
        return "No more sentences!", ""

    ger_text = lines[offset].strip().split("\t")[1]
    frs_text = translate(ger_text)

    return ger_text, frs_text

def save_translation(german, frisian, correct):
    """Save user-approved translations and update dataset."""
    global offset, lines
    if correct == "Yes":
        with open(ger_output, "a", encoding="utf-8") as ger_file, open(frs_output, "a", encoding="utf-8") as frs_file:
            ger_file.write(german + "\n")
            frs_file.write(frisian + "\n")
        del lines[offset]  # Remove processed line
    elif correct == "No":
        del lines[offset]  # Skip sentence
    
    # Save remaining sentences
    with open(deu_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return correct_translation()

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🌍 German → East Frisian Translator")
    gr.Markdown("Help improve the dataset by verifying translations!")

    german_text = gr.Textbox(label="German Sentence")
    frisian_text = gr.Textbox(label="Suggested Frisian Translation")
    
    correct_btns = gr.Radio(["Yes", "No"], label="Is this translation correct?")
    submit_btn = gr.Button("Submit")
    
    submit_btn.click(save_translation, inputs=[german_text, frisian_text, correct_btns], outputs=[german_text, frisian_text])

    # Fetch first sentence on launch
    german_text.value, frisian_text.value = correct_translation()

demo.launch()