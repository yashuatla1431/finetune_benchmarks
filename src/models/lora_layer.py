"""
Custom LoRA (Low-Rank Adaptation) implementation.
Reduces trainable parameters by 99%+ while maintaining performance.
"""

import torch
import torch.nn as nn


class LoraLayer(nn.Module):
    """
    LoRA adapter layer that wraps an existing linear layer.

    Args:
        original_layer: The layer to wrap (nn.Linear, Conv1D, or Int8Linear)
        r: LoRA rank (lower = fewer parameters)
        alpha: LoRA scaling factor
    """

    def __init__(self, original_layer, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.original_layer = original_layer
        self.alpha = alpha
        self.r = r

        # Handle different layer types (GPT2 Conv1D vs standard Linear)
        if type(original_layer).__name__ == 'Conv1D':
            in_features, out_features = original_layer.weight.shape
        else:
            out_features, in_features = original_layer.weight.shape

        # Create LoRA parameters on same device as original layer
        device = original_layer.weight.device

        # A matrix: initialized with Kaiming uniform (scaled by rank)
        # B matrix: initialized to zeros (LoRA starts as identity)
        self.loraA = nn.Parameter(torch.randn(r, in_features, device=device) / self.r)
        self.loraB = nn.Parameter(torch.zeros(out_features, r, device=device))

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, x):
        """
        Forward pass: original output + scaled LoRA adaptation.

        h = Wx + (alpha/r) * (B @ A) @ x
        """
        # Original layer output
        out = self.original_layer(x)

        # LoRA adaptation: x @ A^T @ B^T (cast to input dtype)
        lora_input = self.dropout(x) if self.dropout else x
        lora_weight = (self.loraA.T @ self.loraB.T).to(x.dtype)
        act = lora_input @ lora_weight
        scaled_act = (self.alpha / self.r) * act

        return out + scaled_act
