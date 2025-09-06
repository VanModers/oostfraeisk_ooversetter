import onnxruntime as ort
import numpy as np
import torch
from transformers import MarianTokenizer

# Load tokenizer
model_path = "./frisian_model_3"
tokenizer = MarianTokenizer.from_pretrained(model_path)

# Load ONNX model
onnx_model_path = "frisian_translation.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# Translate function using ONNX Runtime
def translate(text, max_length=50):
    inputs = tokenizer(text, return_tensors="np")

    # Prepare input tensors
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    print(attention_mask) 

    # Start decoding loop
    decoder_input_ids = np.array([[58100]])  # Start with <pad>
    decoder_attention_mask = np.array([[1]])

    for i in range(max_length):
        print(decoder_input_ids)
        print(decoder_attention_mask)

        outputs = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask
            }
        )

        # Extract next token ID from logits
        next_token_id = np.argmax(outputs[0][:, -1, :], axis=-1).reshape(1, -1)

        #print(decoder_input_ids)
        #print(next_token_id)
        #print(tokenizer.eos_token_id)

        # Stop if end token is reached
        if next_token_id[0, 0] == tokenizer.eos_token_id:
            break

        # Append new token for next iteration
        decoder_input_ids = np.hstack([decoder_input_ids, next_token_id])
        decoder_attention_mask = np.hstack([decoder_attention_mask, [[0]]])
        decoder_attention_mask[0,i+1] = 1

    # Decode translated tokens to text
    translated_text = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)

    return translated_text

# Test the translation
input_text = "Die Liebe ist dein Freund."
translated_text = translate(input_text)

print(f"Input: {input_text}")
print(f"Translation: {translated_text}")