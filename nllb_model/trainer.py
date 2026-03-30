import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
from dataset_creator import get_dataset, add_frs_lang, FRS_LANG, DEU_LANG

MODEL_NAME = "facebook/nllb-200-distilled-600M"
OUTPUT_DIR = "./nllb_frs_model_longer"

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Register frs_Latn as a new language, seeded from Dutch (nld_Latn) embeddings
add_frs_lang(tokenizer, model, random_init=False)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="no",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,      # effective batch size = 64
    num_train_epochs=30,  # More epochs
    learning_rate=2e-5,   # Lower learning rate
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

# --- Custom Trainer to print translation after each evaluation ---
class PrintTranslationTrainer(Seq2SeqTrainer):
    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def evaluate(self, *args, **kwargs):
        results = super().evaluate(*args, **kwargs)
        # Print translation of the provided sentence
        sentence = ("Die Energiepreise sind seit Beginn des Iran-Kriegs drastisch gestiegen - mit Folgen für Verbraucher und Wirtschaft. "
                    "Mecklenburg-Vorpommerns Ministerpräsidentin Schwesig fordert im Bericht aus Berlin deshalb sofortige Entlastungen.")
        self.model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.tokenizer.src_lang = DEU_LANG
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        tgt_id = self.tokenizer.convert_tokens_to_ids(FRS_LANG)
        # Debug prints
        print('frs_Latn in vocab:', FRS_LANG in self.tokenizer.get_vocab())
        print('frs_Latn id:', self.tokenizer.convert_tokens_to_ids(FRS_LANG))
        print('lang_code_to_id:', getattr(self.tokenizer, 'lang_code_to_id', None))
        print('input_ids:', inputs['input_ids'])
        print('forced_bos_token_id:', tgt_id)
        with torch.inference_mode():
            out = self.model.generate(**inputs, forced_bos_token_id=tgt_id, max_new_tokens=256, num_beams=4)
        translation = self.tokenizer.decode(out[0], skip_special_tokens=True)
        print("\n[Sample translation after epoch evaluation]")
        print(f"German: {sentence}\nOostfräisk: {translation}\n")
        return results


# Add early stopping callback: patience=3 (stop if no improvement for 3 evals)
trainer = PrintTranslationTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
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
