"""
BitNet b1.58 Ternary Quantization & Fast Walsh-Hadamard Transform (FWHT)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def fast_walsh_hadamard_transform_1d(x: torch.Tensor) -> torch.Tensor:
    """
    1D In-place Fast Walsh-Hadamard Transform (FWHT) for a tensor of length N (power of 2).
    Orthogonal rotation with scale factor 1/sqrt(N).
    """
    N = x.shape[-1]
    h = 1
    out = x.clone()
    while h < N:
        for i in range(0, N, h * 2):
            for j in range(i, i + h):
                x0 = out[..., j].clone()
                x1 = out[..., j + h].clone()
                out[..., j] = x0 + x1
                out[..., j + h] = x0 - x1
        h *= 2
    return out / math.sqrt(N)

def blockwise_fwht(x: torch.Tensor, block_size: int = 512) -> torch.Tensor:
    """
    Applies FWHT block-by-block across the hidden dimension d.
    If hidden_dim = 1536 and block_size = 512, applies 3 independent 512-point rotations.
    """
    orig_shape = x.shape
    d = orig_shape[-1]
    if d % block_size != 0:
        return x  # Fallback if not divisible
    
    x_reshaped = x.view(-1, d // block_size, block_size)
    x_rotated = fast_walsh_hadamard_transform_1d(x_reshaped)
    return x_rotated.view(orig_shape)

class BitLinear(nn.Module):
    """
    BitNet b1.58 Linear Layer with Straight-Through Estimator (STE)
    and online Fast Walsh-Hadamard Transform for outlier suppression.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, use_fwht: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_fwht = use_fwht and (in_features % 512 == 0)
        
        # Latent master weights in FP32/BF16
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.rms_norm = nn.RMSNorm(in_features, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization scaled for ternary regime
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def quantize_weights(self, w: torch.Tensor) -> torch.Tensor:
        # Scale factor gamma: mean absolute value
        gamma = torch.mean(torch.abs(w)).clamp(min=1e-6)
        # Scaled round to {-1, 0, +1}
        w_scaled = w / gamma
        w_quant = torch.clamp(torch.round(w_scaled), -1.0, 1.0)
        # Straight-Through Estimator (STE)
        return w + (w_quant * gamma - w).detach()

    def quantize_activations(self, x: torch.Tensor) -> torch.Tensor:
        # 8-bit dynamic activation quantization
        eta = torch.max(torch.abs(x), dim=-1, keepdim=True).values.clamp(min=1e-6)
        scale = 127.0 / eta
        x_quant = torch.clamp(torch.round(x * scale), -128.0, 127.0)
        return x + (x_quant / scale - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Sub-Layer RMSNorm
        x_norm = self.rms_norm(x)
        
        # 2. Online FWHT to eliminate activation outliers
        if self.use_fwht:
            x_rot = blockwise_fwht(x_norm, block_size=512)
        else:
            x_rot = x_norm
            
        # 3. Quantize activations and weights
        x_q = self.quantize_activations(x_rot)
        w_q = self.quantize_weights(self.weight)
        
        # 4. Add-Only / Bit-parallel linear contraction
        out = F.linear(x_q, w_q, self.bias)
        return out
