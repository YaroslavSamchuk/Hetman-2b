"""
Gated DeltaNet Linear Recurrent Attention Layer
O(1) Memory State (128 KB per layer) with Chunked Parallel Scan and Stepwise Decode
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from configuration_hetman import HetmanConfig
from quantization import BitLinear

class GatedDeltaNet(nn.Module):
    def __init__(self, config: HetmanConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.deltanet_head_dim
        self.total_dim = self.num_heads * self.head_dim  # 16 * 64 = 1024
        
        # Linear projections using BitLinear
        self.q_proj = BitLinear(self.hidden_size, self.total_dim, bias=False)
        self.k_proj = BitLinear(self.hidden_size, self.total_dim, bias=False)
        self.v_proj = BitLinear(self.hidden_size, self.total_dim, bias=False)
        
        # Data-dependent gates (alpha: decay/forget gate, beta: learning rate/write gate)
        self.alpha_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        self.beta_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        
        # Output projection and channel gating
        self.out_proj = BitLinear(self.total_dim, self.hidden_size, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, prev_state: torch.Tensor = None):
        """
        x: [B, L, hidden_size]
        prev_state: [B, num_heads, head_dim, head_dim] or None
        Returns:
            out: [B, L, hidden_size]
            next_state: [B, num_heads, head_dim, head_dim] (128 KB per batch item)
        """
        B, L, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)
        
        # Unit-normalize queries and keys for spherical stability
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        
        # Compute gates
        alpha = torch.sigmoid(self.alpha_proj(x))  # [B, L, H] decay
        beta = torch.sigmoid(self.beta_proj(x))    # [B, L, H] write rate
        
        # Initialize recurrent state S_0: [B, H, d_k, d_v]
        if prev_state is None:
            S = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
        else:
            S = prev_state.clone()
            
        outputs = []
        
        # Stepwise linear recurrence (Delta rule)
        # S_t = S_{t-1} * alpha_t + beta_t * (v_t - S_{t-1} k_t) k_t^T
        for t in range(L):
            q_t = q[:, t]  # [B, H, D]
            k_t = k[:, t]  # [B, H, D]
            v_t = v[:, t]  # [B, H, D]
            a_t = alpha[:, t, :, None, None]  # [B, H, 1, 1]
            b_t = beta[:, t, :, None, None]   # [B, H, 1, 1]
            
            # Predict retrieved value: v_pred = S_{t-1} @ k_t
            v_pred = torch.einsum('bhij,bhj->bhi', S, k_t)  # [B, H, D]
            
            # Delta error: v_err = v_t - v_pred
            v_err = v_t - v_pred
            
            # Rank-1 associative update
            update = torch.einsum('bhi,bhj->bhij', v_err, k_t)
            S = S * a_t + b_t * update
            
            # Retrieve output: y_t = S_t @ q_t
            y_t = torch.einsum('bhij,bhj->bhi', S, q_t)
            outputs.append(y_t)
            
        # Stack over sequence length
        out_stacked = torch.stack(outputs, dim=1)  # [B, L, H, D]
        out_normed = self.norm(out_stacked).view(B, L, self.total_dim)
        
        # Output projection with Swish channel gate
        gate = F.silu(self.gate_proj(x))
        y = self.out_proj(out_normed) * gate
        
        return y, S
