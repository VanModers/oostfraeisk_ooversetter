"""Fine-tune base NLLB with bidirectional real and synthetic parallel data."""
import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path

import torch
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
    EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed,
)

from check_training_env import require_safe_torch_load
from data_split import default_validation_path
from dataset_creator import add_frs_lang, get_dataset

MODEL_NAME = "facebook/nllb-200-distilled-600M"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=os.environ.get("NLLB_OUTPUT_DIR", "./nllb_frs_model"))
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--validation-path", default=default_validation_path())
    parser.add_argument("--no-synthetic", action="store_true", help="Run the real-data-only control.")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=float, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    require_safe_torch_load()
    set_seed(args.seed)
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    add_frs_lang(tokenizer, model, random_init=False)
    model.config.use_cache = False  # Decoder caches are for inference, not full-sequence training.

    root = Path(args.data_path)
    dataset = get_dataset(
        root, tokenizer, tatoeba_path=root / "tatoeba/training", db_eng_path=root,
        external_path=args.validation_path,
        synthetic_path=None if args.no_synthetic else root / "synthetic",
        manifest_path=Path(args.output_dir) / "data_split_manifest.json",
        bidirectional=True,
    )
    print("Mixture: full dataset, shuffled once per epoch in both directions")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir, eval_strategy="epoch", save_strategy="epoch",
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        lr_scheduler_type="cosine", warmup_steps=args.warmup_steps, weight_decay=0.05,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=100, dataloader_num_workers=4, dataloader_pin_memory=torch.cuda.is_available(),
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        save_total_limit=3, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        seed=args.seed, data_seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    metadata = {
        "model": MODEL_NAME, "arguments": vars(args),
        "versions": {name: version(name) for name in ("torch", "transformers", "accelerate")},
        "training_arguments": training_args.to_dict(),
    }
    (Path(args.output_dir) / "run_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    trainer = Seq2SeqTrainer(
        model=model, args=training_args, train_dataset=dataset["train"],
        eval_dataset=dataset["validation"], processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.tie_weights()
    model.config.use_cache = True
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
