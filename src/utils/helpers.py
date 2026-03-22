"""
Utility functions for validation, inference, and checkpointing.
"""

import torch
import os


@torch.no_grad()
def validate(model, loader, device):
    """
    Run validation and compute average loss and perplexity.

    Args:
        model: Model to evaluate
        loader: ValLoader instance
        device: Device to run on

    Returns:
        val_loss (float): Average validation loss
        perplexity (float): Perplexity (exp(loss))
    """
    model.eval()
    total_loss = 0
    count = 0

    while not loader.batch_completed:
        x, y = loader.get_batch()
        x = x.to(device)
        y = y.to(device)
        output = model(x, labels=y)
        total_loss += output.loss.item()
        count += 1

    model.train()
    loader.reset()

    val_loss = total_loss / count
    perplexity = torch.exp(torch.tensor(val_loss)).item()

    return val_loss, perplexity


@torch.no_grad()
def test_inference(model, tokenizer, device, prompt=None):
    """
    Generate sample output for qualitative evaluation.

    Args:
        model: Model to use for generation
        tokenizer: Tokenizer for encoding/decoding
        device: Device to run on
        prompt: Optional custom prompt

    Returns:
        Generated text string
    """
    model.eval()

    if prompt is None:
        prompt = (
            '##Human:Write a replace method for a string class which replaces '
            'the given string with a given set of characters.\n'
            'string = "Hello World!" replace_with = "Greetings!"\n\n##Response:'
        )

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    generated = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.8,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    result = tokenizer.decode(generated[0], skip_special_tokens=True)
    model.train()
    return result


def save_checkpoint(model, optimizer, scheduler, step, loss, config, output_dir, device):
    """
    Save training checkpoint.

    Args:
        model: Model (possibly wrapped in DDP)
        optimizer: Optimizer state
        scheduler: Scheduler state
        step: Current training step
        loss: Current loss value
        config: Training configuration dict
        output_dir: Directory to save checkpoint
        device: Device string for RNG state
    """
    os.makedirs(output_dir, exist_ok=True)

    # Handle DDP wrapped models
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    checkpoint = {
        "step": step,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "config": config,
        "rng_state": torch.get_rng_state(),
    }

    if 'cuda' in device:
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state()

    checkpoint_path = os.path.join(output_dir, f'checkpoint_{step}')
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cuda'):
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
        device: Device for RNG state

    Returns:
        step: Training step from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle DDP wrapped models
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    torch.set_rng_state(checkpoint['rng_state'])
    if 'cuda' in device and 'cuda_rng_state' in checkpoint:
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])

    return checkpoint['step']
