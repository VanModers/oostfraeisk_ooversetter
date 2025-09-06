from transformers import MarianTokenizer

# Load tokenizer
tokenizer = MarianTokenizer.from_pretrained("frisian_model")

# Save vocab as JSON
tokenizer.save_pretrained("tokenizer_json")

#print(tokenizer("Gibt es Menschen auf der Erde", return_tensors="pt"))

print("Saved tokenizer JSON & vocab files.")