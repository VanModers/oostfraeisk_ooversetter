import torch
import onnx
from transformers import MarianMTModel, MarianTokenizer

# Load your fine-tuned model
model_path = "./frisian_model_3"
model = MarianMTModel.from_pretrained(model_path)
tokenizer = MarianTokenizer.from_pretrained(model_path)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Dummy input for ONNX conversion
text = "Hallo, wie geht's?"  # Example sentence
inputs = tokenizer(text, return_tensors="pt")
inputs.to(device)

# Define decoder input (start with BOS token)
decoder_input_ids = torch.tensor([[0]]).to(device)
decoder_attention_mask = torch.tensor([[1]]).to(device)

# Add attention mask
attention_mask = inputs["attention_mask"].to(device)

# Export to ONNX
onnx_path = "frisian_translation.onnx"
torch.onnx.export(
    model,
    (inputs["input_ids"], attention_mask, decoder_input_ids, decoder_attention_mask),
    onnx_path,
    opset_version=11,
    input_names=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask"],
    output_names=["output_ids"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "decoder_input_ids": {0: "batch_size", 1: "sequence_length"},
        "decoder_attention_mask": {0: "batch_size", 1: "sequence_length"},
        "output_ids": {0: "batch_size", 1: "sequence_length"}
    }
)

print(f"Model exported to {onnx_path}")