import gradio as gr
import os
import random
import spaces
import shutil
import zipfile
import torch
from transformers import MarianMTModel, MarianTokenizer

# Set environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# Define paths
DATA_DIR = "data/"
LOCAL_DIR = "./"

# Ensure local directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Files to copy
files_to_copy = ["deu.txt", "eastfrisian.txt", "german.txt"]

# Copy each file to the local directory
for file in files_to_copy:
    src_path = os.path.join(DATA_DIR, file)
    dst_path = os.path.join(LOCAL_DIR, file)
    if os.path.exists(src_path):  
        shutil.copy(src_path, dst_path)

# Load model and tokenizer
model_path = "frisian_model"
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Enable half precision if using CUDA
if device == "cuda":
    model.half()

# Compile model (PyTorch 2.0+)
try:
    model = torch.compile(model)
except:
    pass  # Ignore if PyTorch <2.0

print(device)

# Translate function
#@spaces.GPU
def translate(text):
    with torch.inference_mode():  # Disables gradient computation
        inputs = tokenizer(text, return_tensors="pt")
        inputs.to(device)
        translated = model.generate(**inputs, num_beams=3)

    # Decode output
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# File paths
deu_file = "deu.txt"
ger_output = "german.txt"
frs_output = "eastfrisian.txt"
zip_file = "translations.zip"  # The final ZIP file

# Load sentences
with open(deu_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

#offset = random.randint(0, len(lines) - 1) if len(lines) > 1 else 0
offset = len(lines) - 1

def correct_translation():
    """Fetch a new German sentence and translate it."""
    global offset, lines
    if offset >= len(lines):
        return "No more sentences!", ""

    ger_text = lines[offset].strip().split("\t")[1]
    frs_text = translate(ger_text)

    return ger_text, frs_text

def exists_in_file(ger_file, ger):
    with open(ger_file, "r", encoding="utf-8"):
        for line in ger_file:
            if ger in line:
                return True
    return False

def save_translation(german, frisian):
    """Save user-approved translations and update dataset."""
    global offset, lines
    if german == lines[offset].strip().split("\t")[1] and not exists_in_file(ger_output, german):
        with open(ger_output, "a", encoding="utf-8") as ger_file, open(frs_output, "a", encoding="utf-8") as frs_file:
            ger_file.write(german + "\n")
            frs_file.write(frisian + "\n")
        del lines[offset]  # Remove processed line
        offset = offset - 1

    # Save remaining sentences
    with open(deu_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # Create ZIP file
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        zipf.write(ger_output)
        zipf.write(frs_output)
        zipf.write(deu_file)

    ger_text, frs_text = correct_translation()
    return ger_text, frs_text, zip_file  # Return ZIP file path

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🌍 German → East Frisian Translator")
    gr.Markdown("Help improve the dataset by verifying translations!")

    german_text = gr.Textbox(label="German Sentence")
    frisian_text = gr.Textbox(label="Suggested Frisian Translation")
    
    submit_btn = gr.Button("Submit")

    # Download ZIP file
    zip_download = gr.File(label="Download Translations ZIP")

    submit_btn.click(save_translation, inputs=[german_text, frisian_text], outputs=[german_text, frisian_text, zip_download])

    # Fetch first sentence on launch
    german_text.value, frisian_text.value = correct_translation()