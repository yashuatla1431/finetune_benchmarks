import numpy as np

val = np.load('../finetune_shards/shard_val.npz')
print(f"Type of shard['tokens']: {type(val['tokens'])}")
print(f"len(shard['tokens']): {len(val['tokens'])}")
print(f"Shape: {val['tokens'].shape}")
print(f"Actual length: {val['tokens'].shape[0]}")

# What happens with batch size 2, block size 1024
batch_size = 2
block_size = 1024
chunk = batch_size * block_size
pointer = 0

# First batch
tokens_np = val['tokens'][pointer:pointer + chunk + 1]
labels_np = val['labels'][pointer:pointer + chunk + 1]
print(f"\nFirst batch:")
print(f"tokens_np length: {len(tokens_np)}")
print(f"labels_np length: {len(labels_np)}")

# What if pointer goes beyond?
pointer = 575000  # Near end
tokens_np = val['tokens'][pointer:pointer + chunk + 1]
labels_np = val['labels'][pointer:pointer + chunk + 1]
print(f"\nNear end:")
print(f"Pointer: {pointer}")
print(f"tokens_np length: {len(tokens_np)}")
print(f"labels_np length: {len(labels_np)}")
print(f"Can reshape to ({batch_size}, {block_size})? {len(tokens_np)-1 == batch_size * block_size}")
