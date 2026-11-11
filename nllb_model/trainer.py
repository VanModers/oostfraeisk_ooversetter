"""Fine-tune NLLB with bidirectional real and synthetic parallel data."""
import argparse
import json
import os
import shutil
import sys
import math
from importlib.metadata import version
from pathlib import Path

import torch
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
    EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed,
)

from check_training_env import require_safe_torch_load
from data_split import default_validation_path
from dataset_creator import add_frs_lang, get_dataset, FRS_LANG

MODEL_NAME = "facebook/nllb-200-distilled-600M"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=os.environ.get("NLLB_OUTPUT_DIR", "./nllb_frs_model"))
    parser.add_argument("--checkpoint-dir", default=os.environ.get("NLLB_CHECKPOINT_DIR"),
                        help="Training checkpoints and optimizer state; defaults to --output-dir.")
    parser.add_argument("--adapt", action="store_true", help="Conservative three-epoch adaptation of the published model.")
    parser.add_argument("--initial-model", default=MODEL_NAME)
    parser.add_argument("--scheduler", choices=("cosine", "linear"), default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--export-epochs", action="store_true", help="Keep all epoch checkpoints and export their inference weights.")
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
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Finish the complete learning-rate schedule; still export the best validation-loss model.")
    parser.add_argument("--save-last-model", action="store_true",
                        help="Also export the final epoch's weights to OUTPUT/last-epoch.")
    parser.add_argument("--resume-from-checkpoint", default=None)
    if "--adapt" in sys.argv[1:]:
        parser.set_defaults(initial_model="VanModers114/East_Frisian_NLLB_Model",
                            epochs=3, learning_rate=3e-6, scheduler="linear", warmup_ratio=0.05,
                            no_early_stopping=True, export_epochs=True,
                            output_dir="./nllb_frs_adapted")
    args = parser.parse_args()
    if args.warmup_ratio is not None and not 0 <= args.warmup_ratio <= 1:
        parser.error("--warmup-ratio must be between zero and one")
    return args


def main():
    args = parse_args()
    final_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else final_dir
    if args.adapt and not args.resume_from_checkpoint:
        for destination in (final_dir, checkpoint_dir):
            if destination.exists() and any(destination.iterdir()):
                raise ValueError(f"Use an empty directory for adaptation: {destination}")
    require_safe_torch_load()
    set_seed(args.seed)
    print(f"Loading {args.initial_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.initial_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.initial_model)
    # Existing East Frisian checkpoints already contain a learned language row.
    if FRS_LANG in tokenizer.get_vocab():
        add_frs_lang(tokenizer)
    else:
        add_frs_lang(tokenizer, model, random_init=False)
    model.config.use_cache = False  # Decoder caches are for inference, not full-sequence training.

    root = Path(args.data_path)
    dataset = get_dataset(
        root, tokenizer, tatoeba_path=root / "tatoeba/training", db_eng_path=root,
        external_path=args.validation_path,
        synthetic_path=None if args.no_synthetic else root / "synthetic",
        manifest_path=checkpoint_dir / "data_split_manifest.json",
        bidirectional=True,
    )
    print("Mixture: full dataset, shuffled once per epoch in both directions")

    warmup_steps = args.warmup_steps
    if args.warmup_ratio is not None:
        steps_per_epoch = math.ceil(math.ceil(len(dataset["train"]) / args.batch_size)
                                    / args.gradient_accumulation_steps)
        warmup_steps = math.ceil(steps_per_epoch * args.epochs * args.warmup_ratio)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_dir), eval_strategy="epoch", save_strategy="epoch",
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        lr_scheduler_type=args.scheduler, warmup_steps=warmup_steps, weight_decay=0.05,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=100, dataloader_num_workers=4, dataloader_pin_memory=torch.cuda.is_available(),
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        save_total_limit=None if args.export_epochs else 3, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        seed=args.seed, data_seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    metadata = {
        "model": args.initial_model, "arguments": vars(args),
        "versions": {name: version(name) for name in ("torch", "transformers", "accelerate")},
        "training_arguments": training_args.to_dict(),
    }
    (checkpoint_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    trainer = Seq2SeqTrainer(
        model=model, args=training_args, train_dataset=dataset["train"],
        eval_dataset=dataset["validation"], processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[] if args.no_early_stopping else [EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.tie_weights()
    model.config.use_cache = True
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    trainer.state.save_to_json(str(final_dir / "trainer_state.json"))
    if checkpoint_dir.resolve() != final_dir.resolve():
        for name in ("run_config.json", "data_split_manifest.json"):
            shutil.copyfile(checkpoint_dir / name, final_dir / name)
    exports = []
    if args.export_epochs:
        for checkpoint in sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])):
            state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
            exports.append((checkpoint, final_dir / f"epoch-{state['epoch']:g}"))
    if args.save_last_model:
        exports.append((checkpoint_dir / f"checkpoint-{trainer.state.global_step}", final_dir / "last-epoch"))
    for last_checkpoint, last_dir in exports:
        # train() has restored the best weights. Load each requested checkpoint
        # on CPU to export that epoch's actual weights.
        last_model = AutoModelForSeq2SeqLM.from_pretrained(str(last_checkpoint))
        last_model.tie_weights()
        last_model.config.use_cache = True
        last_model.save_pretrained(last_dir)
        tokenizer.save_pretrained(last_dir)
        shutil.copyfile(last_checkpoint / "trainer_state.json", last_dir / "trainer_state.json")
        last_metadata = {**metadata, "exported_checkpoint": str(last_checkpoint)}
        (last_dir / "run_config.json").write_text(json.dumps(last_metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Epoch model saved to {last_dir}")
        del last_model
    print(f"Model saved to {final_dir}; training checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    main()
