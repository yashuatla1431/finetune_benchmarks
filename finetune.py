
from transformers import GPT2LMHeadModel, GPT2Config
import tiktoken
import wandb
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch import nn
from tqdm import tqdm
from transformers import pipeline
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training

block_size = 1024
batch_size = 4
grad_accum_steps = 4
lr = 3e-4
warmup_steps = 100
lr_scheduler = "cosine"
max_steps = 20001
gradient_clip = 1.0
dropout = 0.1 
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

# Load pretrained checkpoint from pretraining
# pretrained_checkpoint_path = 'pt_checkpoints/checkpoint_1600'
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# # Load pretrained weights if checkpoint exists
# if os.path.exists(pretrained_checkpoint_path):
#     pretrained_checkpoint = torch.load(pretrained_checkpoint_path, map_location=device)
#     model.load_state_dict(pretrained_checkpoint['model_state_dict'])
#     print(f"Loaded pretrained checkpoint from step {pretrained_checkpoint['step']}")
# else:
#     print("No pretrained checkpoint found, using base GPT2")

# Move model to device immediately after loading

class int8Linear(nn.Module):
    def __init__(self,model):
        super().__init__()
        # should take only nn.Linear modules
        self.model = model
        self.quantized_weights = None
        self.bias = model.bias
        self.scale_values = None
        self.quantize()
    
    @property
    def weight(self):
        class WeightProxy:
            def __init__(self, shape, device):
                self.shape = shape
                self.device = device
        return WeightProxy(self.quantized_weights.shape, self.quantized_weights.device)
    
    def forward(self,input):
        weights_fp16 = self.dequantize()
        return F.linear(input,weights_fp16,self.bias)

    def dequantize(self):

        original_shape = self.quantized_weights.shape
        original_numel = self.quantized_weights.numel()
        flat = self.quantized_weights.flatten()
        remainder = flat.numel()%64
        if remainder:
            flat = F.pad(flat,(0,64-remainder))
        blocks = flat.view(-1,64)
        blocks = blocks.to(self.scale_values.dtype)*self.scale_values
        param = blocks.flatten()[:original_numel].view(original_shape)
        return param
    
    def quantize(self):
        # Let's quantize every tensor 
        param = self.model.weight.data
        # first we to flatten 
        original_shape = param.shape
        original_numel = param.numel()
        flat = param.flatten()
        # now divide into blocks 
        remainder = flat.numel()%64
        if remainder:
            flat = F.pad(flat,(0,64-remainder))
        blocks = flat.view(-1,64)
        # find the max values per block 
        max_values = blocks.abs().max(1,keepdim=True).values
        # calculate scale
        scale_values = max_values/127
        # compress teh data 
        blocks = blocks/scale_values
        # clamp and convert to int 8
        blocks = blocks.clamp(-128,127).to(torch.int8)
        # revert back to actual param shape 
        param = blocks.flatten()[:original_numel].view(original_shape)
        # setting the weights and bias 
        self.quantized_weights = param
        self.scale_values = scale_values
        return             
    
for name,module in model.named_modules():
    if isinstance(module,nn.Linear):
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent,child_name,int8Linear(module))

model.to(device)

config_dict={
    "block_size":block_size,
    "batch_size":batch_size, 
    "grad_accum_steps":grad_accum_steps,
    "lr":lr,
    "warmup_steps":warmup_steps,
    "lr_scheduler":lr_scheduler,
    "max_steps":max_steps,
    "gradient_clip":gradient_clip,
    "dropout":dropout,
    "eval_steps":eval_steps,
    "save_steps":save_steps,
    "logging_steps":logging_steps,
}


if rank == 0:
    run = wandb.init(
        entity="yashuatla6-individual",
        project="gpt2-code-train",
        config={
            "block_size":block_size,
            "batch_size":batch_size, 
            "grad_accum_steps":grad_accum_steps,
            "lr":lr,
            "warmup_steps":warmup_steps,
            "lr_scheduler":lr_scheduler,
            "max_steps":max_steps,
            "gradient_clip":gradient_clip,
            "dropout":dropout,
            "eval_steps":eval_steps,
            "save_steps":save_steps,
            "logging_steps":logging_steps,
        }
    )

  

class LoraLayer(nn.Module):
    def __init__(self,original_layer,r=8,alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.alpha = alpha
        self.r = r
        # GPT2 Conv1D: weight is (in, out), Standard PyTorch layers: weight is (out, in)
        if type(original_layer).__name__ == 'Conv1D':
            in_features, out_features = original_layer.weight.shape
        else:
            out_features, in_features = original_layer.weight.shape
        # Create LoRA parameters on the same device as original layer
        device = original_layer.weight.device
        self.loraA = nn.Parameter(torch.randn(r,in_features, device=device)/self.r)
        self.loraB = nn.Parameter(torch.zeros(out_features,r, device=device))
    
    def forward(self,x):
        out = self.original_layer(x)
        act = x@(self.loraA.T@self.loraB.T)
        scaled_act = (self.alpha/self.r)*act
        return out + scaled_act
    
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
        self.token_length += self.chunk_size*self.world_size  # UPDATE THIS!

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

# Freeze all original parameters
for param in model.parameters():
    param.requires_grad = False

# adding lora layer - Optimal rank for code generation
num_layers = len(list(model.model.layers))
for i in range(num_layers):
    for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
          original_layer = getattr(model.model.layers[i].self_attn, proj)
          setattr(model.model.layers[i].self_attn, proj, LoraLayer(original_layer, r=64, alpha=128))
          
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Trainable: {trainable}")
print(f"Frozed : {frozen}")

loader = DataLoader(shards,batch_size,block_size,world_size,rank)
val_loader = ValLoader('finetune_shards/shard_val.npy',batch_size,block_size)
optimizer = optim.AdamW(model.parameters(),lr=lr)
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
if resume_from and not os.path.exists(resume_from):
    print(f"Checkpoint path {resume_from} not Exists")
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



