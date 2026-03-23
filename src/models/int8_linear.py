"""
Custom 8-bit quantization implementation using blockwise quantization.
Achieves ~50% memory reduction compared to fp16 with minimal accuracy loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8Linear(nn.Module):
    """
    8-bit quantized linear layer with blockwise quantization.

    Args:
        original_layer: nn.Linear module to quantize
        block_size: Number of values per quantization block (default: 64)
    """

    def __init__(self, original_layer, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.bias = original_layer.bias
        self.quantize(original_layer.weight.data)

    @property
    def weight(self):
        """Proxy for compatibility with LoRA layer."""
        class WeightProxy:
            def __init__(self, shape, device):
                self.shape = shape
                self.device = device
        return WeightProxy(self.quantized_weights.shape, self.quantized_weights.device)

    def forward(self, input):
        """Dequantize weights and compute linear transformation."""
        weights_fp = self.dequantize()
        return F.linear(input, weights_fp, self.bias)

    def dequantize(self):
        """Dequantize int8 weights to fp32 using stored scales."""
        original_shape = self.quantized_weights.shape
        original_numel = self.quantized_weights.numel()
        flat = self.quantized_weights.flatten()

        # Pad to block size
        remainder = flat.numel() % self.block_size
        if remainder:
            flat = F.pad(flat, (0, self.block_size - remainder))

        # Reshape to blocks and dequantize
        blocks = flat.view(-1, self.block_size)
        blocks = blocks.to(self.scale_values.dtype) * self.scale_values

        # Unpad and reshape to original shape
        param = blocks.flatten()[:original_numel].view(original_shape)
        return param

    def quantize(self, weight):
        """Quantize fp32 weights to int8 using blockwise quantization."""
        original_shape = weight.shape
        original_numel = weight.numel()
        flat = weight.flatten()

        # Pad to block size
        remainder = flat.numel() % self.block_size
        if remainder:
            flat = F.pad(flat, (0, self.block_size - remainder))

        # Reshape to blocks
        blocks = flat.view(-1, self.block_size)

        # Compute scale per block (symmetric quantization)
        max_values = blocks.abs().max(1, keepdim=True).values
        scale_values = max_values / 127.0

        # Quantize to int8 range [-128, 127]
        blocks = blocks / scale_values
        blocks = blocks.clamp(-128, 127).to(torch.int8)

        # Unpad and reshape to original shape
        param = blocks.flatten()[:original_numel].view(original_shape)

        # Store quantized weights and scales as buffers (keeps device consistency)
        self.register_buffer('quantized_weights', param)
        self.register_buffer('scale_values', scale_values)
