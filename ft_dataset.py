import tiktoken
enc = tiktoken.get_encoding('gpt2')


from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("sahil2801/CodeAlpaca-20k",streaming=True,split='train')

import numpy as np
pbar = tqdm(total=2*(10**4),desc="Rows processed")
i = 0
train_tokens = []
val_tokens = []

for obj in dataset:
    instruction = obj['instruction']
    input = obj['input']
    output = obj['output']
    if not output.strip():
        continue
    if input.strip():
        train_exp = f"##Human:{instruction}\n{input}\n\n##Response:{output}"
    else:
        train_exp = f"##Human:{instruction}\n\n##Response:{output}"

    tokens_np = enc.encode(train_exp)
    tokens_np.append(enc._special_tokens['<|endoftext|>'])

    # Split: every 5th example goes to validation (20% split)
    if i % 5 == 0:
        val_tokens.extend(tokens_np)
    else:
        train_tokens.extend(tokens_np)

    i += 1
    pbar.update(1)

train_tokens_np = np.array(train_tokens)
val_tokens_np = np.array(val_tokens)

np.save('finetune_shards/shard_train.npy', train_tokens_np)
np.save('finetune_shards/shard_val.npy', val_tokens_np)

print(f"\nTrain tokens: {len(train_tokens_np)}")
print(f"Val tokens: {len(val_tokens_np)}")