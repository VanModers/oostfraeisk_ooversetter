import gradio as gr
import json
import spaces
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, M2M100Config
from dataset_creator import add_frs_lang

# Fix transformers 5.4.0 bug: M2M100Config.scale_embedding has a bool default
# but is typed as int, breaking strict HF dataclass validation in two places:
# 1. Class default patch — prevents M2M100Config() (no-args) from failing in repr/to_diff_dict.
_se_field = M2M100Config.__dataclass_fields__.get("scale_embedding")
if _se_field is not None and isinstance(_se_field.default, bool):
    _se_field.default = int(_se_field.default)

LANG_CODES = {
    "Deutsch": "deu_Latn",
    "Oostfräisk": "frs_Latn",
    "English": "eng_Latn",
}

MODEL_PATH = "VanModers114/East_Frisian_NLLB_Model"

# 2. Config dict patch — AutoTokenizer/AutoModel load config.json internally and
#    pass it to cls(**config_dict), so we must pre-build the config with the fixed value.
_config_file = hf_hub_download(repo_id=MODEL_PATH, filename="config.json")
with open(_config_file, "r", encoding="utf-8") as _f:
    _config_dict = json.load(_f)
if isinstance(_config_dict.get("scale_embedding"), bool):
    _config_dict["scale_embedding"] = int(_config_dict["scale_embedding"])
_model_config = M2M100Config.from_dict(_config_dict)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, config=_model_config)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, config=_model_config)
add_frs_lang(tokenizer)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

@spaces.GPU
def translate(text, source_lang, target_lang):
    if not text.strip() or source_lang == target_lang:
        return text

    src_code = LANG_CODES[source_lang]
    tgt_code = LANG_CODES[target_lang]

    tokenizer.src_lang = src_code
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    tgt_id = tokenizer.convert_tokens_to_ids(tgt_code)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            forced_bos_token_id=tgt_id,
            max_new_tokens=1024,
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
