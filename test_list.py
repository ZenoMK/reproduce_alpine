import os
import torch
import argparse
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from model import GPTConfig, GPT
from utils_final import (
    AttentionVisualizer
)

# Argument parsing
parser = argparse.ArgumentParser(description="Transformer Output Validation")
parser.add_argument('--ckpt_iter', type=int, default=10000, help="Checkpoint iteration")
parser.add_argument('--graph_type', type=str, default='list', help="Graph type for consistency")
parser.add_argument('--config', type=str, default='1_1_120', help="Model configuration")
parser.add_argument('--temperature', type=float, default=1, help="Sampling temperature")
parser.add_argument('--device', type=str, default='cpu', help="Device (cpu/gpu)")
parser.add_argument('--num_nodes', type=int, default=100, help="Number of nodes")
parser.add_argument('--num_of_paths', type=int, default=20, help="Number of paths")

args = parser.parse_args()
dataset = args.graph_type
ckpt_iter = args.ckpt_iter
device = args.device
temperature = args.temperature
num_nodes = args.num_nodes
num_of_paths = args.num_of_paths
config = args.config

# Define paths
data_path = f'data/{dataset}/{num_nodes}_list/'
out_dir = f'out/{dataset}_{config}_{num_nodes}_list/'

# Load metadata
meta_path = f'{data_path}/meta.pkl'
print(f"Loading metadata from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi, itos = meta['stoi'], meta['itos']
max_new_tokens = meta['block_size']
top_k = len(itos)
simple_format = True

# Load model checkpoint
ckpt_path = os.path.join(out_dir, f'{ckpt_iter}_ckpt_20.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']

# Adjust state dict keys if needed
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)


# Encoding and Decoding functions
def encode(s):
    """Encodes space-separated numbers into token indices."""
    return [stoi[ch] for ch in s.split()]


def decode(l):
    """Decodes token indices into space-separated numbers."""
    return " ".join(itos[i] for i in l)


# Function to validate the generated output
def validate_output(original, generated):
    """Checks if the generated sequence is the reverse of the input."""
    if original[::-1] == generated:
        return "correct"
    else:
        return "incorrect"


# Read test data
typedata = 'test'
test_file = os.path.join(data_path, f'{typedata}.txt')

texts = []
encode_texts = []
ground_truth = []

with open(test_file, 'r', encoding='utf-8') as f:
    for line in f:
        if not simple_format:
            texts.append(line.split(':')[0] + ':')
            encode_texts.append(encode(line.split(':')[0] + ':'))
        else:
            line = line.split("%")[0]
            numbers = re.findall(r'\d+', line)
            if numbers:
                input_sequence = " ".join(numbers)
                texts.append(input_sequence)
                encode_texts.append(encode(input_sequence))

        ground_truth.append(line)

ground_truth = np.array(ground_truth)
# Store original lengths
original_lengths = [len(seq) for seq in encode_texts]
# Convert list of lists to a tensor with padding
max_len = max(len(seq) for seq in encode_texts)
encode_texts_padded = [seq + [0] * (max_len - len(seq)) for seq in encode_texts]


encode_texts = torch.tensor(encode_texts_padded, dtype=torch.long, device=device)

# Generate and validate outputs
batch_size = 1000
ix = torch.randint(len(encode_texts), (batch_size,))

pred_file = os.path.join(out_dir, f'pred_{typedata}_{ckpt_iter}.txt')

# Initialize empty file
with open(pred_file, 'w') as f:
    pass

wrong = 0
correct_lengths = []
incorrect_lengths = []

for i in tqdm(range(10), desc="Generating and validating outputs"):
    x = encode_texts[ix]
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    PAD_TOKEN = 0  # or whatever your model's padding token is
    y = [[token for token in y[t].tolist() if token != PAD_TOKEN] for t in range(batch_size)]
    y_pred = [decode(y[t]).split('\n')[0] for t in range(batch_size)]

    with open(pred_file, 'a') as f:
        for t, item in enumerate(y_pred):
            original = texts[ix[t]].split()
            generated_pre = item.split(" % ")[1]
            generated = generated_pre.split()
            validation = validate_output(original, generated)
            path_len = len(original)

            if validation == "incorrect":
                incorrect_lengths.append(path_len)
                wrong += 1
            else:
                correct_lengths.append(path_len)

            f.write(f"{texts[ix[t]]} % {generated_pre} % {validation}\n")

        f.write(f"Number of wrongs: {wrong}\n")

# Compute correctness proportions
correct_counts = Counter(correct_lengths)
incorrect_counts = Counter(incorrect_lengths)

all_lengths = set(correct_counts.keys()).union(incorrect_counts.keys())

proportions = {
    length: correct_counts[length] / (correct_counts[length] + incorrect_counts[length])
    for length in all_lengths
}

# Sorting lengths for plotting
sorted_lengths = sorted(proportions.keys())
sorted_proportions = [proportions[length] for length in sorted_lengths]

# Plot results
plt.figure(figsize=(8, 5))
plt.bar(sorted_lengths, sorted_proportions, color="green", alpha=0.7)
plt.xlabel("List Length")
plt.ylabel("Proportion of Correct Outputs")
plt.title("Proportion of Correct Outputs per List Length")
plt.xticks(sorted_lengths)
plt.ylim(0, 1)
plt.savefig(os.path.join(out_dir, f'validation_proportion.png'), dpi=400)

print("Validation complete. Results saved.")
