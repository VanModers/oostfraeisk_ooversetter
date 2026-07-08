import argparse
import os
from pathlib import Path

from dataset_creator import FRS_LANG, DEU_LANG, add_frs_lang, MAX_LENGTH


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "VanModers114/East_Frisian_NLLB_Model"
DEFAULT_INPUT_FILE = REPO_ROOT / "data" / "krektüren" / "deu.txt"
DEFAULT_CORRECTED_GERMAN_FILE = REPO_ROOT / "data" / "fan teksten" / "german.txt"
DEFAULT_GERMAN_OUTPUT = REPO_ROOT / "data" / "auto_translated" / "german.txt"
DEFAULT_EASTFRISIAN_OUTPUT = REPO_ROOT / "data" / "auto_translated" / "eastfrisian.txt"
DEFAULT_COUNT = 10000
DEFAULT_BATCH_SIZE = 16


def parse_german_text(raw_line: str) -> str:
    """Extract German text from a plain line or TSV row."""
    line = raw_line.strip()
    if not line:
        return ""

    parts = line.split("\t", 2)
    if len(parts) >= 2:
        return parts[1].strip()
    return line.strip()


def read_sentence_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }


def select_sentences(raw_lines, count: int, excluded: set[str]) -> list[str]:
    """Select unique, uncorrected German sentences from the end of the source."""
    selected = []
    seen = set(excluded)

    for raw_line in reversed(raw_lines):
        text = parse_german_text(raw_line)
        if not text or text in seen:
            continue
        seen.add(text)
        selected.append(text)
        if len(selected) == count:
            break

    return selected


def remove_sentences_from_source(path: Path, excluded: set[str]) -> tuple[int, int]:
    """Atomically remove source rows whose German sentence has been corrected."""
    temp_path = path.with_name(f".{path.name}.tmp")
    removed = 0
    kept = 0

    try:
        with path.open("r", encoding="utf-8-sig") as source, temp_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as destination:
            for line in source:
                if parse_german_text(line) in excluded:
                    removed += 1
                    continue
                destination.write(line.rstrip("\r\n") + "\n")
                kept += 1
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)

    return removed, kept


def batched(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def translate_batch(texts, tokenizer, model, device, torch):
    tokenizer.src_lang = DEU_LANG
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(FRS_LANG)
    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=MAX_LENGTH,
            num_beams=4,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Translate the last N German sentences to East Frisian for later correction."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument(
        "--corrected-german",
        type=Path,
        default=DEFAULT_CORRECTED_GERMAN_FILE,
        help="German corrections to exclude from selection.",
    )
    parser.add_argument("--german-output", type=Path, default=DEFAULT_GERMAN_OUTPUT)
    parser.add_argument("--eastfrisian-output", type=Path, default=DEFAULT_EASTFRISIAN_OUTPUT)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--clean-input-only",
        action="store_true",
        help="Remove corrected sentences from the input file and exit without loading the model.",
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be larger than 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be larger than 0")

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    corrected = read_sentence_set(args.corrected_german)
    if args.clean_input_only:
        removed, kept = remove_sentences_from_source(args.input, corrected)
        print(f"Removed {removed} corrected row(s); kept {kept} row(s) in {args.input}")
        return

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # The pushed tokenizer uses the newer list form for extra_special_tokens.
    # Passing an empty mapping keeps Transformers 4.49 compatible; all language
    # tokens, including frs_Latn, are already embedded in tokenizer.json.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, extra_special_tokens={})
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    add_frs_lang(tokenizer)

    model.to(device)
    model.eval()

    all_lines = args.input.read_text(encoding="utf-8-sig").splitlines()
    if not all_lines:
        print("Input file is empty. Nothing to translate.")
        return

    german_texts = select_sentences(all_lines, args.count, corrected)

    if not german_texts:
        print("No usable sentences found in selected lines.")
        return

    args.german_output.parent.mkdir(parents=True, exist_ok=True)
    args.eastfrisian_output.parent.mkdir(parents=True, exist_ok=True)

    total = len(german_texts)
    print(f"Translating {total} sentence(s)...")

    written = 0
    with (
        args.german_output.open("w", encoding="utf-8", newline="\n") as ger_file,
        args.eastfrisian_output.open("w", encoding="utf-8", newline="\n") as frs_file,
    ):
        for batch in batched(german_texts, args.batch_size):
            frs_batch = translate_batch(batch, tokenizer, model, device, torch)

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
