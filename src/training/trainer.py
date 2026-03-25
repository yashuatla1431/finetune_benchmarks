"""
Unified trainer for all finetuning methods.
"""

import os
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import wandb

from ..data.dataloader import DataLoader, ValLoader
from ..utils.helpers import validate, test_inference, save_checkpoint


class Trainer:
    """
    Unified trainer that handles all finetuning methods.

    Args:
        model: Model to train
        tokenizer: Tokenizer for inference
        config: Training configuration dict
        rank: DDP rank
        world_size: DDP world size
        local_rank: Local GPU rank
        device: Device string
    """

    def __init__(self, model, tokenizer, config, rank, world_size, local_rank, device):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = device

        # Training config
        train_cfg = config['training']
        self.block_size = train_cfg['block_size']
        self.batch_size = train_cfg['batch_size']
        self.grad_accum_steps = train_cfg['grad_accum_steps']
        self.max_steps = train_cfg['max_steps']
        self.gradient_clip = train_cfg['gradient_clip']
        self.eval_steps = train_cfg['eval_steps']
        self.save_steps = train_cfg['save_steps']
        self.logging_steps = train_cfg['logging_steps']

        # Data
        data_cfg = config['data']
        use_masking = data_cfg.get('use_label_masking', True)
        self.train_loader = DataLoader(
            data_cfg['train_shards'],
            self.batch_size,
            self.block_size,
            world_size,
            rank,
            use_masking
        )
        self.val_loader = ValLoader(
            data_cfg['val_shard'],
            self.batch_size,
            self.block_size,
            use_masking
        )

        # Optimizer
        opt_cfg = config['optimizer']
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=opt_cfg['lr'],
            weight_decay=opt_cfg.get('weight_decay', 0.01)
        )

        # Scheduler
        sched_cfg = config['scheduler']
        if sched_cfg['type'] == 'linear_warmup':
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=sched_cfg['warmup_steps']
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_cfg['type']}")

        # Move model to device and wrap in DDP
        self.model.to(device)
        self.model = DDP(self.model, device_ids=[local_rank])

        # WandB logging (only rank 0)
        if rank == 0 and 'wandb' in config:
            wandb.init(
                entity=config['wandb']['entity'],
                project=config['wandb']['project'],
                name=config['experiment_name'],
                tags=config['wandb'].get('tags', []),
                config=config
            )

        # Output directory
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self, resume_from=None):
        """
        Main training loop.

        Args:
            resume_from: Optional checkpoint path to resume from
        """
        start_step = 0

        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            from ..utils.helpers import load_checkpoint
            start_step = load_checkpoint(
                resume_from,
                self.model,
                self.optimizer,
                self.scheduler,
                self.device
            )
            if self.rank == 0:
                print(f"Resumed from step {start_step}")

        self.optimizer.zero_grad()

        # Progress bar (rank 0 only)
        if self.rank == 0:
            pbar = tqdm(total=self.max_steps, initial=start_step, desc="Training")

        for step in range(start_step, self.max_steps):
            total_loss = 0

            # Gradient accumulation
            for micro_step in range(self.grad_accum_steps):
                x, y = self.train_loader.get_batch()
                x = x.to(self.device)
                y = y.to(self.device)

                # Skip batches where all labels are masked
                if (y == -100).all():
                    continue

                # Mixed precision for CUDA
                if 'cuda' in self.device:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        output = self.model(x, labels=y)
                else:
                    output = self.model(x, labels=y)

                loss = output.loss / self.grad_accum_steps
                total_loss += output.loss.item()

                # Gradient sync only on last micro-step
                if micro_step < self.grad_accum_steps - 1:
                    with self.model.no_sync():
                        loss.backward()
                else:
                    loss.backward()

            # Gradient clipping and optimizer step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            avg_loss = total_loss / self.grad_accum_steps

            # Update progress bar
            if self.rank == 0:
                pbar.update(1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

            # Validation and logging
            if step > 0 and step % self.eval_steps == 0 and self.rank == 0:
                val_loss, perplexity = validate(self.model, self.val_loader, self.device)
                inference_result = test_inference(self.model, self.tokenizer, self.device)

                print(f"\n\nStep {step} Validation:")
                print(f"Val Loss: {val_loss:.4f}")
                print(f"Perplexity: {perplexity:.2f}")
                print(f"Sample Generation:\n{inference_result}\n")

                if 'wandb' in self.config:
                    log_dict = {
                        "train/loss": avg_loss,
                        "val/loss": val_loss,
                        "val/perplexity": perplexity,
                        "lr": self.scheduler.get_last_lr()[0],
                        "step": step
                    }

                    # GPU memory stats
                    if 'cuda' in self.device:
                        log_dict["system/gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(self.device) / 1024**3
                        log_dict["system/gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(self.device) / 1024**3
                        log_dict["system/gpu_memory_peak_gb"] = torch.cuda.max_memory_allocated(self.device) / 1024**3

                    wandb.log(log_dict)

            # Regular logging
            if step % self.logging_steps == 0 and self.rank == 0:
                if 'wandb' in self.config:
                    log_dict = {
                        "train/loss": avg_loss,
                        "lr": self.scheduler.get_last_lr()[0],
                        "step": step
                    }

                    # GPU memory tracking every logging step
                    if 'cuda' in self.device:
                        log_dict["system/gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(self.device) / 1024**3

                    wandb.log(log_dict)

            # Save checkpoint
            if step > 0 and step % self.save_steps == 0 and self.rank == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    step,
                    avg_loss,
                    self.config,
                    self.output_dir,
                    self.device
                )

        # Close progress bar and finish wandb
        if self.rank == 0:
            pbar.close()
            if 'wandb' in self.config:
                wandb.finish()

        print(f"\nTraining completed! Results saved to: {self.output_dir}")
