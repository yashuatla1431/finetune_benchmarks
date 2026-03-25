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
    tokens_ls.append(enc.eos_token_id)

    # Split: every 5th example goes to validation (20% split)
    if i % 5 == 0:
        val_tokens.extend(tokens_ls)
    else:
        train_tokens.extend(tokens_ls)

    i += 1
    pbar.update(1)

train_tokens_np = np.array(train_tokens)
val_tokens_np = np.array(val_tokens)

np.save('finetune_shards/shard_train.npy', train_tokens_np)
np.save('finetune_shards/shard_val.npy', val_tokens_np)

print(f"\nTrain tokens: {len(train_tokens_np)}")
print(f"Val tokens: {len(val_tokens_np)}")