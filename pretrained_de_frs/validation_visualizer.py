import matplotlib.pyplot as plt
import re

# File path
results_file = "validation_results.txt"

# Lists to store parsed data
models = []
accuracy = []
loss = []

# Read and parse file
with open(results_file, "r", encoding="utf-8") as f:
    content = f.read()

# Split by sections
sections = content.strip().split("-------------------------------------------------------------------------------------")
for section in sections:
    if not section.strip():
        continue
    # Extract model name
    model_match = re.search(r"de_frs_model_\d+", section)
    acc_match = re.search(r"Token-wise Accuracy:\s*([0-9.]+)", section)
    loss_match = re.search(r"Avg Loss:\s*([0-9.]+)", section)

    if model_match and acc_match and loss_match:
        models.append(model_match.group(0))
        accuracy.append(float(acc_match.group(1)))
        loss.append(float(loss_match.group(1)))

# Visualization
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.set_xlabel('Models')
ax1.set_ylabel('Token-wise Accuracy', color='tab:blue')
ax1.plot(models, accuracy, marker='o', color='tab:blue', label='Accuracy')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Avg Loss', color='tab:red')
ax2.plot(models, loss, marker='o', color='tab:red', linestyle='--', label='Loss')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Model Performance: Accuracy vs Loss')
plt.grid(axis='x')
plt.tight_layout()
plt.show()