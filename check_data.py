import numpy as np

# Check training data
data = np.load('finetune_shards/shard_train.npz')
print(f'Max token in train: {data["tokens"].max()}')
print(f'Min token in train: {data["tokens"].min()}')
print(f'Max label in train: {data["labels"].max()}')
print(f'Min label in train: {data["labels"].min()}')

# Check validation data
val = np.load('finetune_shards/shard_val.npz')
print(f'\nMax token in val: {val["tokens"].max()}')
print(f'Min token in val: {val["tokens"].min()}')
print(f'Max label in val: {val["labels"].max()}')
print(f'Min label in val: {val["labels"].min()}')

print(f'\nQwen vocab size should be: 151936')
print(f'Max allowed token ID: 151935')
