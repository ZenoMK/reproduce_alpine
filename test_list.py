import os
import argparse
import re
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Validate Transformer Output')
parser.add_argument('--num_nodes', type=int, default=100, help='Used for file path consistency')
parser.add_argument('--num_of_paths', type=int, default=20, help='Used for file naming consistency')
parser.add_argument('--ckpt_iter', type=int, default=10000, help='Checkpoint iteration')
parser.add_argument('--data_type', type=str, default='list', help='Graph type for consistency')
args = parser.parse_args()

dataset = args.graph_type
num_nodes = args.num_nodes
num_of_paths = args.num_of_paths
ckpt_iter = args.ckpt_iter

data_path = f'data/{dataset}/{num_nodes}_path'
out_dir = f'out/{dataset}_{num_nodes}_path/'

pred_file = os.path.join(out_dir, f'pred_test_{ckpt_iter}.txt')

if not os.path.exists(pred_file):
    raise FileNotFoundError(f"Predicted output file not found: {pred_file}")


def validate_output(line):
    """Checks if the part after '%' is the reversed version of the part before '%'."""
    parts = line.strip().split(" % ")
    if len(parts) != 2:
        return "invalid format"

    original = parts[0].strip().split()
    reversed_part = parts[1].strip().split()

    if original[::-1] == reversed_part:
        return "correct"
    else:
        return "incorrect"


# Read predictions and validate
incorrect_counts = 0
correct_counts = 0
incorrect_lengths = []
correct_lengths = []

with open(pred_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

with open(os.path.join(out_dir, f'validated_test_{ckpt_iter}.txt'), "w") as f:
    for line in tqdm(lines, desc="Validating outputs"):
        validation_result = validate_output(line)
        path_len = len(re.findall(r'\d+', line.split(" % ")[0]))

        if validation_result == "correct":
            correct_counts += 1
            correct_lengths.append(path_len)
        else:
            incorrect_counts += 1
            incorrect_lengths.append(path_len)

        f.write(line.strip() + f" % {validation_result}\n")

print(f"Correct outputs: {correct_counts}")
print(f"Incorrect outputs: {incorrect_counts}")

# Plot proportions of correct outputs by length
correct_hist = Counter(correct_lengths)
incorrect_hist = Counter(incorrect_lengths)

all_lengths = set(correct_hist.keys()).union(incorrect_hist.keys())

proportions = {
    length: correct_hist[length] / (correct_hist[length] + incorrect_hist[length])
    for length in all_lengths
}

sorted_lengths = sorted(proportions.keys())
sorted_proportions = [proportions[length] for length in sorted_lengths]

plt.figure(figsize=(8, 5))
plt.bar(sorted_lengths, sorted_proportions, color="green", alpha=0.7)
plt.xlabel("List Length")
plt.ylabel("Proportion of Correct Outputs")
plt.title("Proportion of Correct Outputs per List Length")
plt.xticks(sorted_lengths)
plt.ylim(0, 1)
plt.savefig(os.path.join(out_dir, f'validation_proportion.png'), dpi=400)

print("Validation complete. Results saved.")
