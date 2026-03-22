from transformers import GPT2LMHeadModel
from peft import LoraConfig, get_peft_model, TaskType
import tiktoken
import wandb
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

block_size = 1024
batch_size = 4
grad_accum_steps = 4
lr = 2e-4  # Recommended LR for LoRA from research
warmup_steps = 100
max_steps = 20001
gradient_clip = 1.0
eval_steps = 1000
save_steps = 5000
logging_steps = 100

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ['LOCAL_RANK'])

# Determine device early
if torch.cuda.is_available():
    device = f'cuda:{local_rank}'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=bnb_config)
model = prepare_model_for_kbit_training(model)

config_dict={
    "block_size":block_size,
    "batch_size":batch_size,
    "grad_accum_steps":grad_accum_steps,
    "lr":lr,
    "warmup_steps":warmup_steps,
    "max_steps":max_steps,
    "gradient_clip":gradient_clip,
    "eval_steps":eval_steps,
    "save_steps":save_steps,
    "logging_steps":logging_steps,
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.1
}

if rank == 0:
    run = wandb.init(
        entity="yashuatla6-individual",
        project="gpt2-code-train-peft",
        config=config_dict
    )

# PEFT LoRA Configuration
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
    lora_dropout=0.1,  # Important for preventing overfitting
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA using PEFT
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


shards = ['finetune_shards/shard_train.npy']

class DataLoader:
    def __init__(self, shards, batch_size, block_size, world_size, rank):
        self.shards = shards
        self.batch_size = batch_size
        self.block_size = block_size
        self.chunk_size = batch_size * block_size

        self.shard_index = 0
        self.world_size = world_size
        self.rank = rank
        self.token_length = self.chunk_size*self.rank
        self.shard = np.load(shards[self.shard_index])
        self.shard_length = len(self.shard)

    def get_batch(self):
        # Check if we need to load next shard
        if self.shard_length - self.token_length < (self.chunk_size + 1):
            self.shard_index = (self.shard_index + 1) % len(self.shards)
            self.token_length = self.chunk_size*self.rank
            self.shard = np.load(self.shards[self.shard_index])
            self.shard_length = len(self.shard)

        # Get tokens
        tokens = self.shard[self.token_length:self.token_length + self.chunk_size + 1]
        self.token_length += self.chunk_size*self.world_size

        # Create x and y
        x = torch.from_numpy(tokens[:-1]).view(self.batch_size, self.block_size)
        y = torch.from_numpy(tokens[1:]).view(self.batch_size, self.block_size)

        return x, y


class ValLoader:
    def __init__(self,shard_path,batch_size,block_size):
        self.shard_path = shard_path
        self.batch_size = batch_size
        self.block_size = block_size
        self.pointer = 0
        self.batch_completed = False
        self.shard = np.load(self.shard_path)

    def get_batch(self):
        chunk = self.batch_size*self.block_size
        tokens_np = self.shard[self.pointer:self.pointer+chunk+1]
        self.pointer+=chunk
        if len(self.shard) - self.pointer < chunk+1:
            self.batch_completed = True
        x = torch.from_numpy(tokens_np[:-1]).view(self.batch_size,self.block_size)
        y = torch.from_numpy(tokens_np[1:]).view(self.batch_size,self.block_size)
        return x,y


@torch.no_grad()
def validate(model,loader,device):
    model.eval()
    total_loss = 0
    count = 0
    while not loader.batch_completed:
        x,y = loader.get_batch()
        x = x.to(device)
        y = y.to(device)
        output = model(x,labels=y)
        total_loss += output.loss.item()
        count+=1
    model.train()
    loader.batch_completed = False
    loader.pointer = 0
    return total_loss/count

@torch.no_grad()
def test_inference(model, device):
    model.eval()
    test_prompt = '##Human:Write a replace method for a string class which replaces the given string with a given set of characters.\nstring = "Hello World!" replace_with = "Greetings!"\n\n##Response:'
    input_ids = tokenizer.encode(test_prompt,return_tensors='pt').to(device)

    generated = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.8,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    result = tokenizer.decode(generated[0],skip_special_tokens=True)
    model.train()
    return result


def save_model(model,current_step,optimizer,scheduler,current_loss,device):
    check_point = {
        "step": current_step,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": current_loss,
        "config": config_dict,
        "rng_state": torch.get_rng_state(),
    }
    if 'cuda' in device:
        check_point["cuda_rng_state"] = torch.cuda.get_rng_state()
    torch.save(check_point,f'checkpoints/checkpoint_{current_step}')


resume_from = None

loader = DataLoader(shards,batch_size,block_size,world_size,rank)
val_loader = ValLoader('finetune_shards/shard_val.npy',batch_size,block_size)

# Weight decay helps prevent overfitting
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

# Constant LR after warmup
scheduler = LinearLR(
    optimizer,
    start_factor=0.1,
    total_iters=warmup_steps
)

model = DDP(model,device_ids=[local_rank])
start_point = 0
if resume_from and os.path.exists(resume_from):
    checkpoint = torch.load(resume_from)
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    torch.set_rng_state(checkpoint['rng_state'])
    if 'cuda' in device:
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    start_point = checkpoint['step']

optimizer.zero_grad()
os.makedirs('checkpoints',exist_ok=True)


# Progress bar only on rank 0
if rank == 0:
    pbar = tqdm(total=max_steps, initial=start_point, desc="Training")

for i in range(start_point,max_steps):
    total_loss = 0
    for micro_step in range(grad_accum_steps):
        x,y = loader.get_batch()
        x = x.to(device)
        y = y.to(device)
        if 'cuda' in device:
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                output = model(x,labels=y)
        else:
            output = model(x,labels=y)
        loss = output.loss/grad_accum_steps
        total_loss += output.loss.item()
        if micro_step < grad_accum_steps-1:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),gradient_clip)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    # Update progress bar
    if rank == 0:
        pbar.update(1)
        pbar.set_postfix({'loss': f'{total_loss/grad_accum_steps:.4f}'})

    if i>0 and i%eval_steps == 0 and rank == 0:
        val_loss = validate(model,val_loader,device)
        inference_result = test_inference(model,device)
        print(f"\n\nStep {i} Inference Test:")
        print(inference_result)
        print("\n")
        wandb.log({
            "loss":total_loss/grad_accum_steps,
            "val_loss":val_loss,
            "lr": scheduler.get_last_lr()[0],
            "step": i
        })
    if i>0 and i%save_steps == 0 and rank == 0:
        save_model(model,i,optimizer,scheduler,total_loss/grad_accum_steps,device)
    if i%100 == 0 and rank == 0:
        wandb.log({
                "loss": total_loss/grad_accum_steps,
                "lr": scheduler.get_last_lr()[0],
                "step": i
        })

# Close progress bar
if rank == 0:
    pbar.close()
    wandb.finish()
dist.destroy_process_group()
