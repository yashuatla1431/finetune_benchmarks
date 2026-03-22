"""
Model factory for creating models with different quantization and LoRA methods.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

from .int8_linear import Int8Linear
from .lora_layer import LoraLayer


def apply_custom_quantization(model, method="int8", block_size=64):
    """
    Apply custom quantization to all Linear layers in model.

    Args:
        model: Hugging Face model
        method: Quantization method ("int8")
        block_size: Block size for quantization

    Returns:
        Quantized model
    """
    if method == "int8":
        # Replace all nn.Linear with Int8Linear
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, Int8Linear(module, block_size=block_size))
    else:
        raise ValueError(f"Unknown quantization method: {method}")

    return model


def apply_custom_lora(model, lora_config):
    """
    Apply custom LoRA layers to model.

    Args:
        model: Model to apply LoRA to
        lora_config: Dict with keys: r, alpha, target_modules

    Returns:
        Model with LoRA layers
    """
    r = lora_config['r']
    alpha = lora_config['alpha']
    target_modules = lora_config['target_modules']

    # Freeze all original parameters
    for param in model.parameters():
        param.requires_grad = False

    # Auto-detect number of layers
    num_layers = len(list(model.model.layers))

    # Apply LoRA to specified modules in each layer
    for i in range(num_layers):
        for module_name in target_modules:
            original_layer = getattr(model.model.layers[i].self_attn, module_name)
            setattr(
                model.model.layers[i].self_attn,
                module_name,
                LoraLayer(original_layer, r=r, alpha=alpha)
            )

    return model


def create_model(config):
    """
    Create model based on configuration.

    Args:
        config: Configuration dict with method, model_name, quantization, lora

    Returns:
        model: Prepared model
        tokenizer: Tokenizer
    """
    from transformers import AutoTokenizer

    method = config['method']
    model_name = config['model_name']

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if method == "lora":
        # Custom LoRA (fp16 baseline)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = apply_custom_lora(model, config['lora'])

    elif method == "lora_8bit":
        # Custom 8-bit + custom LoRA
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = apply_custom_quantization(
            model,
            method="int8",
            block_size=config['quantization'].get('block_size', 64)
        )
        model = apply_custom_lora(model, config['lora'])

    elif method == "qlora":
        # QLoRA: bitsandbytes 4-bit + PEFT LoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config['quantization']['quant_type'],
            bnb_4bit_compute_dtype=getattr(torch, config['quantization']['compute_dtype'])
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config
        )

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=config['lora']['r'],
            lora_alpha=config['lora']['alpha'],
            target_modules=config['lora']['target_modules'],
            lora_dropout=config['lora'].get('dropout', 0.1),
            bias=config['lora'].get('bias', 'none'),
            task_type=TaskType.CAUSAL_LM
        )

        model = get_peft_model(model, lora_config)
        if hasattr(model, 'print_trainable_parameters'):
            model.print_trainable_parameters()

    else:
        raise ValueError(f"Unknown method: {method}")

    # Print trainable parameters for custom methods
    if method in ["lora", "lora_8bit"]:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")
        print(f"Frozen parameters: {frozen:,}")
        print(f"Trainable %: {100 * trainable / (trainable + frozen):.2f}%")

    return model, tokenizer
