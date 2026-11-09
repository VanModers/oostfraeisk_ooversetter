"""Evaluate both German/East Frisian directions on the reviewed benchmark."""
import argparse
import os

import torch
from sacrebleu.metrics import BLEU, CHRF
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

from data_split import default_validation_path
from dataset_creator import (
    NLLBTranslationDataset, load_parallel_pairs, make_lang_pairs,
    FRS_LANG, DEU_LANG, add_frs_lang, MAX_LENGTH,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("NLLB_OUTPUT_DIR", "./nllb_frs_model"))
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--validation-path", default=default_validation_path())
    parser.add_argument("--results-path", default="validation_results.txt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=1,
                        help="Number of beams for generation; default is greedy decoding.")
    parser.add_argument("--direction", choices=("both", "de-frs", "frs-de"), default="both")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}; model: {args.model_path}; benchmark: {args.validation_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, dtype=dtype)
    add_frs_lang(tokenizer)
    model.to(device)
    model.eval()
    pairs = [(s, t) for s, t in load_parallel_pairs(args.validation_path) if s and t]
    if not pairs:
        raise ValueError("Validation dataset contains no nonempty aligned pairs")
    directions = [("de-frs", DEU_LANG, FRS_LANG, pairs),
                  ("frs-de", FRS_LANG, DEU_LANG, [(t, s) for s, t in pairs])]
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    for name, source_language, target_language, directed_pairs in directions:
        if args.direction not in ("both", name):
            continue
        dataset = NLLBTranslationDataset(
            make_lang_pairs(directed_pairs, source_language, target_language, bidirectional=False), tokenizer)
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)
        total_tokens = correct_tokens = 0
        total_loss = 0.0
        predictions = []
        with torch.inference_mode():
            for batch in tqdm(loader, desc=name):
                inputs = {key: value.to(device) for key, value in batch.items()
                          if key in ("input_ids", "attention_mask", "labels")}
                outputs = model(**inputs)
                labels = inputs["labels"]
                mask = labels != -100
                count = mask.sum().item()
                total_loss += outputs.loss.item() * count
                total_tokens += count
                correct_tokens += (outputs.logits.argmax(dim=-1) == labels).masked_select(mask).sum().item()
                del outputs  # Release vocabulary logits before autoregressive generation.
                generated = model.generate(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_language),
                    max_new_tokens=MAX_LENGTH, num_beams=args.num_beams, do_sample=False,
                )
                predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        # Score against full reviewed references, never truncated decoder labels.
        references = [[target for _, target in directed_pairs]]
        bleu, chrf = BLEU(tokenize="13a"), CHRF(char_order=6, word_order=0, beta=2)
        bleu_score = bleu.corpus_score(predictions, references).score
        chrf_score = chrf.corpus_score(predictions, references).score
        result = (
            "-------------------------------------------------------------------------------------\n"
            f"{args.model_label or args.model_path} ({name})\n"
            f"Model Path: {args.model_path}\nValidation Path: {args.validation_path}\n"
            f"Pairs: {len(pairs)}; beams: {args.num_beams}; dtype: {dtype}\n"
            f"Token-wise Accuracy: {correct_tokens / total_tokens:.4f}\n"
            f"Token-weighted Avg Loss: {total_loss / total_tokens:.4f}\n"
            f"BLEU Score: {bleu_score:.2f}\nchrF Score: {chrf_score:.2f}\n"
            f"BLEU signature: {bleu.get_signature()}\nchrF signature: {chrf.get_signature()}\n"
        )
        print(result)
        with open(args.results_path, "a", encoding="utf-8") as stream:
            stream.write(result + "\n")
        for prediction, (_, reference) in list(zip(predictions, directed_pairs))[:3]:
            print(f"Pred: {prediction}\nRef: {reference}\n")


if __name__ == "__main__":
    main()
