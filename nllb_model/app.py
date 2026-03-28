
import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dataset_creator import add_frs_lang

LANG_CODES = {
    "Deutsch": "deu_Latn",
    "Oostfräisk": "frs_Latn",
    "English": "eng_Latn",
}

MODEL_PATH = "VanModers114/East_Frisian_NLLB_Model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
add_frs_lang(tokenizer)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def translate(text, source_lang, target_lang):
    if not text.strip() or source_lang == target_lang:
        return text

    src_code = LANG_CODES[source_lang]
    tgt_code = LANG_CODES[target_lang]

    tokenizer.src_lang = src_code
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    tgt_id = tokenizer.convert_tokens_to_ids(tgt_code)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            forced_bos_token_id=tgt_id,
            max_new_tokens=256,
            max_length=None,
            num_beams=4,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)

with gr.Blocks(title="Oostfräisk Ooversetter") as demo:
    gr.Markdown("# Oostfräisk Ooversetter")
    gr.Markdown("Translate between German, East Frisian, and English.")

    with gr.Row():
        src = gr.Dropdown(
            choices=list(LANG_CODES.keys()), value="Deutsch", label="From"
        )
        tgt = gr.Dropdown(
            choices=list(LANG_CODES.keys()), value="Oostfräisk", label="To"
        )

    input_text = gr.Textbox(label="Input", lines=3, placeholder="Enter text to translate...")
    output_text = gr.Textbox(label="Translation", lines=3)

    btn = gr.Button("Translate", variant="primary")
    btn.click(translate, inputs=[input_text, src, tgt], outputs=output_text)

demo.launch()
