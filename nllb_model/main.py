import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dataset_creator import FRS_LANG, DEU_LANG, ENG_LANG, add_frs_lang

MODEL_PATH = "./nllb_frs_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
add_frs_lang(tokenizer)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

DIRECTIONS = {
    "1": ("Deutsch -> Oostfräisk", DEU_LANG, FRS_LANG),
    "2": ("Oostfräisk -> Deutsch", FRS_LANG, DEU_LANG),
    "3": ("English -> Oostfräisk", ENG_LANG, FRS_LANG),
    "4": ("Oostfräisk -> English", FRS_LANG, ENG_LANG),
}


def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    with torch.inference_mode():
        out = model.generate(**inputs, forced_bos_token_id=tgt_id, max_new_tokens=256)

    return tokenizer.decode(out[0], skip_special_tokens=True)


print("NLLB Oostfräisk Ooversetter")
for k, (name, _, _) in DIRECTIONS.items():
    print(f"  {k}: {name}")

direction = input("\nSelect direction (1-4): ").strip()
if direction not in DIRECTIONS:
    direction = "1"

name, src_lang, tgt_lang = DIRECTIONS[direction]
print(f"\n{name}")
print("Type 'end' to exit, 'switch' to change direction.\n")

while True:
    text = input("> ").strip()
    if text.lower() == "end":
        break
    if text.lower() == "switch":
        for k, (n, _, _) in DIRECTIONS.items():
            print(f"  {k}: {n}")
        direction = input("Select (1-4): ").strip()
        if direction in DIRECTIONS:
            name, src_lang, tgt_lang = DIRECTIONS[direction]
            print(f"{name}\n")
        continue
    if text:
        print(f"  {translate(text, src_lang, tgt_lang)}")
