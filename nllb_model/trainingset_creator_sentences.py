import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from dataset_creator import FRS_LANG, DEU_LANG, add_frs_lang


DEFAULT_MODEL_PATH = "VanModers114/East_Frisian_NLLB_Model"
DEFAULT_INPUT_FILE = Path("data/krektüren/deu.txt")
DEFAULT_GERMAN_OUTPUT = Path("data/auto_translated/german.txt")
DEFAULT_EASTFRISIAN_OUTPUT = Path("data/auto_translated/eastfrisian.txt")
DEFAULT_COUNT = 10000
DEFAULT_BATCH_SIZE = 16


def parse_german_text(raw_line: str) -> str:
    """Extract German text from a plain line or TSV row."""
    line = raw_line.rstrip("\n")
    if not line:
        return ""

    parts = line.split("\t")
    if len(parts) >= 2:
        return parts[1].strip()
    return line.strip()


def batched(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def translate_batch(texts, tokenizer, model, device):
    tokenizer.src_lang = DEU_LANG
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(FRS_LANG)
    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=256,
            num_beams=4,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Translate the last N German sentences to East Frisian for later correction."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--german-output", type=Path, default=DEFAULT_GERMAN_OUTPUT)
    parser.add_argument("--eastfrisian-output", type=Path, default=DEFAULT_EASTFRISIAN_OUTPUT)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be larger than 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be larger than 0")

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    add_frs_lang(tokenizer)

    if hasattr(model.config, "max_length") and model.config.max_length is not None:
            model.config.max_length = None
    if hasattr(model, "generation_config") and getattr(model.generation_config, "max_length", None) is not None:
        model.generation_config.max_length = None

    model.to(device)
    model.eval()

    all_lines = args.input.read_text(encoding="utf-8").splitlines()
    if not all_lines:
        print("Input file is empty. Nothing to translate.")
        return

    selected = all_lines[-args.count :]
    selected.reverse()  # Bottom-up: longest sentences first if source is length-sorted.

    german_texts = [parse_german_text(line) for line in selected]
    german_texts = [text for text in german_texts if text]

    if not german_texts:
        print("No usable sentences found in selected lines.")
        return

    args.german_output.parent.mkdir(parents=True, exist_ok=True)
    args.eastfrisian_output.parent.mkdir(parents=True, exist_ok=True)

    total = len(german_texts)
    print(f"Translating {total} sentence(s)...")

    written = 0
    with args.german_output.open("a", encoding="utf-8") as ger_file, args.eastfrisian_output.open(
        "a", encoding="utf-8"
    ) as frs_file:
        for batch in batched(german_texts, args.batch_size):
            frs_batch = translate_batch(batch, tokenizer, model, device)

            for ger, frs in zip(batch, frs_batch):
                ger_file.write(ger + "\n")
                frs_file.write(frs + "\n")
                written += 1

            if written % 200 == 0 or written == total:
                print(f"  Progress: {written}/{total}")

    print("Done.")
    print(f"German output: {args.german_output}")
    print(f"East Frisian output: {args.eastfrisian_output}")


if __name__ == "__main__":
    main()
