"""
Data loading utilities for distributed training on tokenized shards.
"""

import numpy as np
import torch


class DataLoader:
    """
    Distributed data loader for training shards.

    Args:
        shards: List of .npy shard file paths
        batch_size: Number of sequences per batch
        block_size: Sequence length
        world_size: Number of distributed processes
        rank: Current process rank
    """

    def __init__(self, shards, batch_size, block_size, world_size, rank, use_masking=True):
        self.batch_size = batch_size
        self.block_size = block_size
        self.chunk_size = batch_size * block_size
        self.savez = use_masking

        # Add extension based on masking flag
        if use_masking:
            self.shards = [s + '.npz' for s in shards]
        else:
            self.shards = [s + '.npy' for s in shards]

        self.shard_index = 0
        self.world_size = world_size
        self.rank = rank
        self.token_length = self.chunk_size * self.rank
        self.shard = np.load(self.shards[self.shard_index])

        self.shard_length = len(self.shard) if not self.savez else len(self.shard['tokens'])

    def get_batch(self):
        """Get next batch of (input, target) sequences."""
        # Check if we need to load next shard
        if self.shard_length - self.token_length < (self.chunk_size + 1):
            self.shard_index = (self.shard_index + 1) % len(self.shards)
            self.token_length = self.chunk_size * self.rank
            self.shard = np.load(self.shards[self.shard_index])
            self.shard_length = len(self.shard) if not self.savez else len(self.shard['tokens'])

        # Get tokens for this batch
        if not self.savez:
            tokens = self.shard[self.token_length:self.token_length + self.chunk_size + 1]
            self.token_length += self.chunk_size * self.world_size

            # Create input (x) and target (y) tensors
            x = torch.from_numpy(tokens[:-1]).view(self.batch_size, self.block_size)
            y = torch.from_numpy(tokens[1:]).view(self.batch_size, self.block_size)
        else:
            tokens = self.shard['tokens'][self.token_length:self.token_length + self.chunk_size + 1]
            labels = self.shard['labels'][self.token_length:self.token_length + self.chunk_size + 1]
            self.token_length += self.chunk_size * self.world_size

            # Create input (x) and target (y) tensors
            x = torch.from_numpy(tokens[:-1]).view(self.batch_size, self.block_size)
            y = torch.from_numpy(labels[1:]).view(self.batch_size, self.block_size)

        return x, y


class ValLoader:
    """
    Validation data loader (non-distributed).

    Args:
        shard_path: Path to validation shard .npy file
        batch_size: Number of sequences per batch
        block_size: Sequence length
    """

    def __init__(self, shard_path, batch_size, block_size, use_masking=True):
        self.batch_size = batch_size
        self.block_size = block_size
        self.pointer = 0
        self.batch_completed = False
        self.savez = use_masking

        # Add extension based on masking flag
        if use_masking:
            self.shard_path = shard_path + '.npz'
        else:
            self.shard_path = shard_path + '.npy'

        self.shard = np.load(self.shard_path)

    def get_batch(self):
        """Get next validation batch."""
        chunk = self.batch_size * self.block_size
        if not self.savez:
            tokens_np = self.shard[self.pointer:self.pointer + chunk + 1]
            self.pointer += chunk

            if len(self.shard) - self.pointer < chunk + 1:
                self.batch_completed = True

            x = torch.from_numpy(tokens_np[:-1]).view(self.batch_size, self.block_size)
            y = torch.from_numpy(tokens_np[1:]).view(self.batch_size, self.block_size)
        else:
            tokens_np = self.shard['tokens'][self.pointer:self.pointer + chunk + 1]
            labels_np = self.shard['labels'][self.pointer:self.pointer + chunk + 1]
            self.pointer += chunk

            if len(self.shard['tokens']) - self.pointer < chunk + 1:
                self.batch_completed = True

            x = torch.from_numpy(tokens_np[:-1]).view(self.batch_size, self.block_size)
            y = torch.from_numpy(labels_np[1:]).view(self.batch_size, self.block_size)
        return x, y

    def reset(self):
        """Reset loader to beginning."""
        self.batch_completed = False
        self.pointer = 0
