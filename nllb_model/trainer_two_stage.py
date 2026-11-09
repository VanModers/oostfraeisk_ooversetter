"""Train on the broad corpus, then adapt on original text pairs and synthetic prose.

Reads data/two_stage, built by data/create_two_stage_dataset.py. Both stages train
German/English <-> East Frisian and select checkpoints by validation loss.
"""
import argparse
from collections import Counter
import gc
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path

from data_split import default_validation_path

MODEL_NAME = "facebook/nllb-200-distilled-600M"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="./nllb_frs_two_stage")
    parser.add_argument("--checkpoint-dir", help="Optimizer checkpoints; defaults to OUTPUT/checkpoints.")
    parser.add_argument("--data-path", default="data/two_stage")
    parser.add_argument("--validation-path", default=default_validation_path())
    parser.add_argument("--stage1-epochs", type=float, default=10)
    parser.add_argument("--stage2-epochs", type=float, default=3)
    parser.add_argument("--stage1-learning-rate", type=float, default=1e-4)
    parser.add_argument("--stage2-learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=768,
                        help="Token limit on either side; overlong pairs fail instead of being silently truncated.")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--inspect-data", action="store_true", help="Print both data mixtures without loading a model or writing files.")
    args = parser.parse_args(argv)
    for key in ("stage1_epochs", "stage2_epochs", "stage1_learning_rate", "stage2_learning_rate",
                "batch_size", "gradient_accumulation_steps", "eval_steps"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive and finite")
    if not 1 <= args.max_length <= 1024:
        parser.error("--max-length must be between 1 and NLLB's 1024-position limit")
    return args


def prepare_data(root, validation_path):
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Build the clean corpus first: python data/create_two_stage_dataset.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema"] != "original-aligned-two-stage-v1":
        raise ValueError("Unsupported two-stage corpus schema")
    def text_digest(path):
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()

    for name, expected in manifest["files"].items():
        if text_digest(root / name) != expected:
            raise ValueError(f"Corpus changed since generation: {name}")
    for name, expected in manifest["external_sha256"].items():
        if text_digest(Path(validation_path) / name) != expected:
            raise ValueError(f"Benchmark changed; rebuild the corpus to protect it: {name}")

    def read(name):
        fields = ("source", "target", "source_language", "target_language", "corpus", "source_id")
        return [tuple(item[field] for field in fields) for item in
                (json.loads(line) for line in (root / name).read_text(encoding="utf-8").splitlines())]

    train, validation = read("train.jsonl"), read("validation.jsonl")
    indices = [i for i, row in enumerate(train) if row[4] in manifest["stage2_corpora"]]
    actual = {"stage1_pairs": len(train), "stage2_pairs": len(indices), "validation_pairs": len(validation)}
    if not all(actual.values()) or any(manifest["counts"][k] != v for k, v in actual.items()):
        raise ValueError("Corpus counts do not match its manifest")
    for name, rows in (("stage1", train), ("stage2", [train[i] for i in indices]),
                       ("validation", validation)):
        print(f"{name}: {len(rows):,} pairs / {2 * len(rows):,} directional examples; "
              f"sources: {dict(Counter(row[4] for row in rows))}", flush=True)
    return train, validation, indices, manifest


def expand(rows):
    return [example for src, tgt, src_lang, tgt_lang, *_ in rows
            for example in ((src, tgt, src_lang, tgt_lang), (tgt, src, tgt_lang, src_lang))]


def check_token_lengths(dataset, limit):
    longest = {"source": 0, "target": 0}
    for i, example in enumerate(dataset):
        for name, key in (("source", "input_ids"), ("target", "labels")):
            length = len(example[key])
            if length > limit:
                raise ValueError(f"Example {i} {name} exceeds {limit} tokens. Increase --max-length "
                                 "or align shorter passages before training; no data was truncated for training.")
            longest[name] = max(longest[name], length)
    return longest


def training_settings(args, train_size, stage, checkpoint_dir, cuda, bf16):
    epochs = getattr(args, f"{stage}_epochs")
    steps_per_epoch = math.ceil(math.ceil(train_size / args.batch_size)
                                / args.gradient_accumulation_steps)
    return dict(
        output_dir=str(checkpoint_dir), num_train_epochs=epochs,
        learning_rate=getattr(args, f"{stage}_learning_rate"),
        lr_scheduler_type="linear", warmup_steps=math.ceil(steps_per_epoch * epochs * 0.05),
        weight_decay=0.01, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps", save_strategy="steps", eval_steps=args.eval_steps,
        save_steps=args.eval_steps, save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        bf16=bf16, optim="adamw_torch_fused" if cuda else "adamw_torch",
        seed=args.seed, data_seed=args.seed, logging_steps=100, report_to="none",
        dataloader_num_workers=4, dataloader_pin_memory=cuda,
        gradient_checkpointing=args.gradient_checkpointing,
    )


def train_stage(args, stage, initial_model, tokenizer, train, validation, output, checkpoints):
    import torch
    from transformers import (AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, set_seed)
    from dataset_creator import add_frs_lang

    class FinalEvaluation(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= state.max_steps:
                control.should_evaluate = control.should_save = True
            return control

    set_seed(args.seed)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(initial_model))
    if stage == "stage1":
        add_frs_lang(tokenizer, model, random_init=False)
    # Stage 2 reloads the learned language embedding, never initializes it again.
    model.config.use_cache = False
    settings = training_settings(args, len(train), stage, checkpoints,
                                 torch.cuda.is_available(),
                                 torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    training_args = Seq2SeqTrainingArguments(**settings)
    print(f"{stage}: initializing weights from {initial_model}; new optimizer and scheduler", flush=True)
    trainer = Seq2SeqTrainer(
        model=model, args=training_args, train_dataset=train, eval_dataset=validation,
        processing_class=tokenizer, data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[FinalEvaluation()],
    )
    trainer.train()
    model.tie_weights()
    model.config.use_cache = True
    output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    trainer.state.save_to_json(str(output / "trainer_state.json"))
    result = {"initial_model": str(initial_model), "output": str(output),
              "best_checkpoint": trainer.state.best_model_checkpoint,
              "best_eval_loss": trainer.state.best_metric,
              "training_arguments": training_args.to_dict()}
    (output / "run_config.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    # A separate model/Trainer in the next call prevents optimizer state carrying over.
    return result


def require_empty(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Use a new empty directory for this experiment: {path}")


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir)
    checkpoints = Path(args.checkpoint_dir) if args.checkpoint_dir else output / "checkpoints"
    if not args.inspect_data:
        require_empty(output)
        require_empty(checkpoints)
        if output.resolve() == checkpoints.resolve() or checkpoints.resolve() in output.resolve().parents:
            raise ValueError("Checkpoint directory must not equal or contain the model output directory")
    train_rows, val_rows, indices, manifest = prepare_data(Path(args.data_path), args.validation_path)
    if args.inspect_data:
        return

    import torch
    from torch.utils.data import Subset
    from transformers import AutoTokenizer
    from check_training_env import require_safe_torch_load
    from dataset_creator import add_frs_lang, NLLBTranslationDataset

    require_safe_torch_load()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    add_frs_lang(tokenizer)
    # One extra token detects every overlong example without materializing
    # arbitrarily long inputs. Accepted examples retain their complete text.
    train = NLLBTranslationDataset(expand(train_rows), tokenizer, max_length=args.max_length + 1)
    validation = NLLBTranslationDataset(expand(val_rows), tokenizer, max_length=args.max_length + 1)
    manifest["tokenization"] = {
        "limit": args.max_length, "truncation": "reject overlong examples",
        "train_maximum": check_token_lengths(train, args.max_length),
        "validation_maximum": check_token_lengths(validation, args.max_length),
    }
    text_train = Subset(train, [j for i in indices for j in (2 * i, 2 * i + 1)])
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    metadata = {"arguments": vars(args), "versions": {
        name: version(name) for name in ("torch", "transformers", "accelerate")}, "stages": {}}
    initial_model = MODEL_NAME
    for stage, dataset in (("stage1", train), ("stage2", text_train)):
        metadata["stages"][stage] = train_stage(
            args, stage, initial_model, tokenizer, dataset, validation,
            output / stage, checkpoints / stage)
        (output / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        initial_model = output / stage
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Both stages complete. Models: {output / 'stage1'} and {output / 'stage2'}", flush=True)


if __name__ == "__main__":
    main()
