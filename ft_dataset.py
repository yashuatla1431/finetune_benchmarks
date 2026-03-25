from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Use Qwen tokenizer
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
enc = AutoTokenizer.from_pretrained(model_name)

# Load finance-alpaca and take first 20k examples
dataset = load_dataset("gbharti/finance-alpaca", split='train')
dataset = dataset.select(range(min(20000, len(dataset))))

import numpy as np
pbar = tqdm(total=len(dataset), desc="Rows processed")
i = 0
train_tokens = []
val_tokens = []
labels_train_tokens = []
labels_val_tokens = []

def mask_train(tokens_ls):
    labels_ls = tokens_ls.copy()
    break_point = enc.encode("##Response:",add_special_tokens=False)
    sub_len = len(break_point)
    sub_index = 0
    for i in range(len(tokens_ls)-sub_len):
        if tokens_ls[i:i+sub_len] == break_point:
            sub_index = i+sub_len
    for i in range(sub_index):
        labels_ls[i] = -100
    return labels_ls

for obj in dataset:
    instruction = obj['instruction']
    input_text = obj.get('input', '')
    output = obj['output']

    # Skip empty outputs
    if not output.strip():
        continue

    # Format: instruction + input -> output
    if input_text.strip():
        train_exp = f"##Human:{instruction}\nInput:{input_text}\n\n##Response:{output}"
    else:
        train_exp = f"##Human:{instruction}\n\n##Response:{output}"

    tokens_ls = enc.encode(train_exp, add_special_tokens=False)
    labels_ls = mask_train(tokens_ls)
    tokens_ls.append(enc.eos_token_id)
    labels_ls.append(enc.eos_token_id)

    # Split: every 5th example goes to validation (20% split)
    if i % 5 == 0:
        labels_val_tokens.extend(labels_ls)
        val_tokens.extend(tokens_ls)
    else:
        labels_train_tokens.extend(labels_ls)
        train_tokens.extend(tokens_ls)

    i += 1
    pbar.update(1)

train_tokens_np = np.array(train_tokens)
val_tokens_np = np.array(val_tokens)
labels_train_tokens_np = np.array(labels_train_tokens)
labels_val_tokens_np = np.array(labels_val_tokens)

np.savez('finetune_shards/shard_train.npz', tokens=train_tokens_np,labels=labels_train_tokens_np)
np.savez('finetune_shards/shard_val.npz', tokens=val_tokens_np,labels=labels_val_tokens_np)

print(f"\nTrain tokens: {len(train_tokens_np)}")
print(f"Val tokens: {len(val_tokens_np)}")