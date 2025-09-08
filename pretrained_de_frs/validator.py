import torch
from torch.utils.data import DataLoader, random_split
from transformers import MarianTokenizer, MarianMTModel, DataCollatorForSeq2Seq
from dataset_creator import TranslationDataset, get_dataset  # Your existing dataset code
from tqdm import tqdm
import evaluate

# BLEU metric
bleu_metric = evaluate.load("sacrebleu")

models = ["de_frs_model"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

results_path = "validation_results.txt"
open(results_path, "w").close()

for model_path in models:
    # Load model and tokenizer
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    model.to(device)

    # Load validation dataset
    val_ds = TranslationDataset("validation data", tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    val_loader = DataLoader(val_ds, batch_size=8, collate_fn=data_collator)

    # Evaluation loop
    model.eval()
    total_tokens = 0
    correct_tokens = 0
    total_loss = 0

    predictions_text = []
    references_text = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Evaluating {model_path}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass (teacher forcing)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            total_loss += outputs.loss.item() * input_ids.size(0)

            # Token-wise accuracy
            preds = torch.argmax(logits, dim=-1)
            mask = labels != tokenizer.pad_token_id
            correct_tokens += (preds == labels).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()

            # Sequence-level BLEU (using generate)
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=labels.shape[1])
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions_text.extend(decoded_preds)
            references_text.extend([[l] for l in decoded_labels])  # sacrebleu expects list of lists

    # Compute metrics
    token_accuracy = correct_tokens / total_tokens
    avg_loss = total_loss / len(val_ds)
    bleu_score = bleu_metric.compute(predictions=predictions_text, references=references_text)

    result_str = (
        "-------------------------------------------------------------------------------------\n"
        f"{model_path}\n"
        f"Token-wise Accuracy: {token_accuracy:.4f}\n"
        f"Avg Loss: {avg_loss:.4f}\n"
        f"BLEU Score: {bleu_score['score']:.2f}\n"
    )

    print(result_str)

    # Append results to file
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(result_str + "\n")