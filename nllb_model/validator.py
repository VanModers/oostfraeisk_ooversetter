import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from dataset_creator import NLLBTranslationDataset, load_parallel_pairs, FRS_LANG, DEU_LANG, add_frs_lang
from tqdm import tqdm
import evaluate

bleu_metric = evaluate.load("sacrebleu")

MODEL_PATH = "./nllb_frs_model"
VALIDATION_PATH = "validation data"
RESULTS_PATH = "validation_results.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
add_frs_lang(tokenizer)
model.to(device)
model.eval()

# Evaluate de -> frs only (for comparison with existing MarianMT results)
val_pairs = load_parallel_pairs(VALIDATION_PATH)
print(f"Validation pairs: {len(val_pairs)}")

val_ds = NLLBTranslationDataset(val_pairs, tokenizer, bidirectional=False)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
val_loader = DataLoader(val_ds, batch_size=8, collate_fn=data_collator)

frs_lang_id = tokenizer.convert_tokens_to_ids(FRS_LANG)

total_tokens = 0
correct_tokens = 0
total_loss = 0
predictions_text = []
references_text = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating de->frs"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Loss + token accuracy (teacher forcing)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item() * input_ids.size(0)

        preds = torch.argmax(outputs.logits, dim=-1)
        mask = labels != -100
        correct_tokens += (preds == labels).masked_select(mask).sum().item()
        total_tokens += mask.sum().item()

        # BLEU (generation)
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            forced_bos_token_id=frs_lang_id,
            max_new_tokens=256,
            max_length=None,
            num_beams=4,
        )
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        labels_for_decode = labels.clone()
        labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
        decoded_labels = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)

        predictions_text.extend(decoded_preds)
        references_text.extend([[l] for l in decoded_labels])

token_accuracy = correct_tokens / total_tokens
avg_loss = total_loss / len(val_ds)
bleu_score = bleu_metric.compute(predictions=predictions_text, references=references_text)

result_str = (
    "-------------------------------------------------------------------------------------\n"
    f"NLLB-200-distilled-600M (de->frs)\n"
    f"Token-wise Accuracy: {token_accuracy:.4f}\n"
    f"Avg Loss: {avg_loss:.4f}\n"
    f"BLEU Score: {bleu_score['score']:.2f}\n"
)

print(result_str)

print("Sample translations (de -> frs):")
for i in range(min(5, len(predictions_text))):
    print(f"  Pred: {predictions_text[i]}")
    print(f"  Ref:  {references_text[i][0]}")
    print()

with open(RESULTS_PATH, "a", encoding="utf-8") as f:
    f.write(result_str + "\n")
