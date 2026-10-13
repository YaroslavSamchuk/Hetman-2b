"""
BitNet b1.58 Ternary Quantization & Fast Walsh-Hadamard Transform (FWHT)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def fast_walsh_hadamard_transform_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Vectorized Fast Walsh-Hadamard Transform (FWHT) for tensor of length N (power of 2).
    Computed in float32 for high numerical precision and stability, then converted back to input dtype.
    Orthogonal rotation with scale factor 1/sqrt(N).
    """
    orig_dtype = x.dtype
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    out = x.to(compute_dtype)
    N = out.shape[-1]
    lead_shape = out.shape[:-1]
    out = out.reshape(-1, N)
    
    h = 1
    while h < N:
        out = out.reshape(-1, N // (2 * h), 2, h)
        a = out[:, :, 0, :]
        b = out[:, :, 1, :]
        out = torch.stack([a + b, a - b], dim=2).reshape(-1, N)
        h *= 2
        
    out = out.reshape(*lead_shape, N) / math.sqrt(N)
    return out.to(orig_dtype)

def blockwise_fwht(x: torch.Tensor, block_size: int = 512) -> torch.Tensor:
    """
    Applies FWHT block-by-block across the hidden dimension d.
    If hidden_dim = 1536 and block_size = 512, applies 3 independent 512-point rotations.
    """
    orig_shape = x.shape
    d = orig_shape[-1]
    if d % block_size != 0:
        return x  # Fallback if not divisible
    
    x_reshaped = x.reshape(-1, d // block_size, block_size)
    x_rotated = fast_walsh_hadamard_transform_1d(x_reshaped)
    return x_rotated.reshape(orig_shape)

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
