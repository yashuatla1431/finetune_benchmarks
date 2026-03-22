"""
Main training script - entry point for all experiments.

Usage:
    python scripts/train.py --config configs/lora.yaml
    python scripts/train.py --config configs/lora_8bit.yaml
    python scripts/train.py --config configs/qlora.yaml
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.model_factory import create_model
from src.training.trainer import Trainer


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_distributed():
    """Initialize distributed training."""
    # Use gloo for CPU/Mac, nccl for CUDA
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    return rank, world_size, local_rank


def get_device(local_rank):
    """Determine device to use."""
    if torch.cuda.is_available():
        return f'cuda:{local_rank}'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def main():
    parser = argparse.ArgumentParser(description='Train LLM with LoRA/QLoRA')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"\n{'='*60}")
    print(f"Experiment: {config['experiment_name']}")
    print(f"Method: {config['method']}")
    print(f"Model: {config['model_name']}")
    print(f"{'='*60}\n")

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = get_device(local_rank)

    if rank == 0:
        print(f"Device: {device}")
        print(f"World size: {world_size}\n")

    # Create model and tokenizer
    if rank == 0:
        print("Creating model...")
    model, tokenizer = create_model(config)

    # Create trainer
    if rank == 0:
        print("Initializing trainer...\n")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device
    )

    # Train
    if rank == 0:
        print("Starting training...\n")
    trainer.train(resume_from=args.resume)

    # Cleanup
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
