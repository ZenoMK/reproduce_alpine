import os

import tiktoken

from model import GPTConfig, GPT
import numpy as np
import networkx as nx
import argparse
import pickle
import re
import torch
#from utils_final import (
  #  AttentionVisualizer
#)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_iter', type=int, default=10000)
    parser.add_argument('--config', type=str, default='6_6_120')
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--num_of_paths', type=int, default=20)
    return parser.parse_args()


args = parse_args()
dataset = 'simple_graph'
ckpt_iter = args.ckpt_iter
device = args.device
temperature = args.temperature
num_nodes = args.num_nodes
num_of_paths = args.num_of_paths
config = args.config

data_path = f'data/{dataset}/{num_nodes}_cut'
meta_path = f'{data_path}/meta.pkl'

print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi, itos = meta['stoi'], meta['itos']
max_new_tokens = meta['block_size']
top_k = len(itos)
simple_format = meta['simple_format']

out_dir = f'out/{dataset}_{config}_{num_nodes}_cut/'

if (num_of_paths == 0):
    ckpt_path = os.path.join(out_dir, f'{ckpt_iter}_ckpt.pt')
else:
    ckpt_path = os.path.join(out_dir, f'{ckpt_iter}_ckpt_{num_of_paths}.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

tokenizer = tiktoken.get_encoding("gpt2")
print(type(out_dir))
#viz = AttentionVisualizer(model, tokenizer, out_dir=out_dir, test_path=f'{data_path}/test.txt')
#viz.infer_and_visualize_attention(heads=[0], layers=[0], problem="cut")

cut_graph = (f'{data_path}/graph.graphml')
cut_graph = nx.read_graphml(cut_graph)


def find_third_number_position(number_string):
    numbers = number_string.split()
    third_number_index = 2
    position = sum(len(num) for num in numbers[:third_number_index]) + third_number_index - 1
    return position


def encode(s):
    s = s.rstrip()
    ss = s.split(" ")
    encoded_string = [stoi[ch] for ch in ss]
    return encoded_string


def decode(l):
    dec = ""
    for i in l:
        dec = dec + itos[i] + " "
    return dec[:-1]


def check_path(G, gen_str):
    path = re.findall(r'\d+', gen_str)
    if len(path) < 4:
        return 'wrong syntax'

    for node in path:
        if int(node) > len(itos) or int(node) < 0:
            return 'wrong syntax'

    if path[2] != path[0] or path[-1] != path[1]:
        return 'incorrect start/end'

    for i in range(2, len(path) - 1):
        if not G.has_edge(path[i], path[i + 1]):
            return f'non-existence path {path[i], path[i + 1]}'

    return ''


def check_path_unreachable(G, gen_str, gt):
    path = re.findall(r'\d+|x', gen_str)
    if 'x' in path and len(path) < 4:
        return 0 if 'x' in gt else 1

    if 'x' in gt and 'x' not in gen_str:
        return 1

    return check_path(G, gen_str)

def postprocess_output(gen_str):
    parts = gen_str.strip().split(" % ")
    if len(parts) != 2:
        split = parts[0].rstrip().split()
        s, t = split[0], split[1]
        return None, False, s,t

    st_part, cut_part = parts[0], parts[1]
    st_nodes = st_part.split("!")[0].rstrip()
    st_nodes = st_nodes.split()

    if len(st_nodes) != 2:
        return None, False, None, None

    s, t = map(int, st_nodes)
    print(f"s: {s}, t : {t}")
    if not cut_part.replace(" ","").isnumeric():
        return None, False, s, t
    cut_vertices = set(map(int, cut_part.split()))

    valid_cut = check_cut(cut_graph, s, t, cut_vertices)

    return cut_vertices, valid_cut, s, t


def check_cut(G, s, t, cut_vertices):
    """
    Checks if the given set of cut_vertices forms a valid cut between s and t.

    Parameters:
    - G: networkx.DiGraph, the directed graph
    - s: int, source node
    - t: int, target node
    - cut_vertices: set of int, the vertices that form the cut

    Returns:
    - bool: True if it's a valid cut, False otherwise
    """
    if s not in cut_vertices or t in cut_vertices:
        return False
    cut_vertices.discard(s)
    cut_vertices = set(map(str, cut_vertices))
    G_cut = G.copy()
    print(list(G_cut.nodes))
    G_cut.remove_nodes_from(cut_vertices)
    print(list(G_cut.nodes))
    return not nx.has_path(G_cut, str(s), str(t))


typedata = 'test'
f = open(f'{data_path}/{typedata}.txt', encoding='gbk')
texts = []
encode_texts = []
ground_truth = []

for line in f:
    line = line.rstrip()
    pos = line.find('!')
    if (line[:pos] != ''):
        texts.append(line[:pos])
        encode_texts.append(encode(line[:pos]))

    ground_truth.append(line)

ground_truth = np.array(ground_truth)
encode_texts = torch.tensor(encode_texts, dtype=torch.long, device=device)

from tqdm import tqdm

batch_size = 1000
ix = torch.randint(len(encode_texts), (batch_size,))

with open(out_dir + f'pred_{typedata}_{ckpt_iter}.txt', 'w') as f:
    pass

wrong = 0
for i in tqdm(range(10)):
    x = encode_texts[ix]

    # x = (torch.tensor(text, dtype=torch.long, device=device))
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

    y_pred = [decode(y[t].tolist()).split('\n')[0] for t in range(batch_size)]

    with open(out_dir + f'pred_{typedata}_{ckpt_iter}.txt', 'a') as f:
        for t, item in enumerate(y_pred):
            print(item)
            output, valid, s, t = postprocess_output(item)
            if not valid:
                wrong = wrong + 1
            f.write(f"{s} {t}" +"!" + item + " " + str(valid) + '\n')
        f.write(f"Number of wrongs: {wrong}")



