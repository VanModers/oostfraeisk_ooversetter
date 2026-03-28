from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from dataset_creator import get_dataset, add_frs_lang

MODEL_NAME = "facebook/nllb-200-distilled-600M"
OUTPUT_DIR = "./nllb_frs_model"

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Register frs_Latn as a new language, seeded from Dutch (nld_Latn) embeddings
add_frs_lang(tokenizer, model)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,      # effective batch size = 64
    num_train_epochs=20,
    learning_rate=1e-5,
    warmup_steps=2000,
    weight_decay=0.01,
    bf16=True,
    logging_steps=100,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    optim="adamw_torch_fused",
)

# Loads data/ at the project root (run as: python nllb_model/trainer.py)
dataset = get_dataset("data", tokenizer)
train_ds = dataset["train"]
val_ds = dataset["validation"]

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# Re-tie shared embeddings before saving.  The Trainer's
# load_best_model_at_end checkpoint reload reports 'missing keys'
# for the tied weight aliases; tie_weights() guarantees they
# all point to the same tensor again.
model.tie_weights()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
