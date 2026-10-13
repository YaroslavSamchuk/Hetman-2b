"""
Gated DeltaNet Linear Recurrent Attention Layer (Frontier Edition)
O(1) Memory State (128 KB per layer) with Causal 1D Depthwise Convolution (k=4)
and Multi-Scale Decay Channels (Fast, Episodic, Anchor).
"""

import math
from typing import Optional, Tuple, Union
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
        self.conv_kernel = getattr(config, "deltanet_conv_kernel", 4)
        
        # 1. Short Causal 1D Depthwise Convolution for local token n-gram morphology
        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.conv_kernel,
            groups=self.hidden_size,
            bias=True,
            padding=0
        )
        
        # 2. Linear projections using BitLinear
        self.q_proj = BitLinear(self.hidden_size, self.total_dim, bias=False)
        self.k_proj = BitLinear(self.hidden_size, self.total_dim, bias=False)
        self.v_proj = BitLinear(self.hidden_size, self.total_dim, bias=False)
        
        # 3. Data-dependent gates (alpha: decay/forget gate, beta: learning rate/write gate)
        self.alpha_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        self.beta_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        
        # 4. Output projection and channel gating
        self.out_proj = BitLinear(self.total_dim, self.hidden_size, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        # 5. Multi-Scale Channel Decay Initialization (alpha-heads):
        with torch.no_grad():
            n_fast = max(1, self.num_heads // 4)
            n_anchor = max(1, self.num_heads // 4)
            n_episodic = max(0, self.num_heads - n_fast - n_anchor)
            alpha_init = torch.tensor([1.0] * n_fast + [3.5] * n_episodic + [6.9] * n_anchor, dtype=torch.float32)
            self.alpha_proj.bias.copy_(alpha_init)
            nn.init.constant_(self.beta_proj.bias, -1.0)

    def forward(
        self,
        x: torch.Tensor,
        prev_state: Optional[Union[torch.Tensor, Tuple[Optional[torch.Tensor], torch.Tensor]]] = None,
        state_reset_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x: [B, L, hidden_size]
        prev_state: Tuple[conv_state, S] or S or None
            conv_state: [B, hidden_size, k - 1]
            S: [B, num_heads, head_dim, head_dim]
        Returns:
            out: [B, L, hidden_size]
            next_state: Tuple[conv_state, S] (O(1) memory: ~128 KB per layer)
        """
        B, L, D = x.shape
        
        # Unpack composite recurrent state
        conv_state = None
        S = None
        if prev_state is not None:
            if isinstance(prev_state, tuple):
                conv_state, S = prev_state
            else:
                S = prev_state
                
        # 1. Causal 1D Convolution over input tokens
        k_sz = self.conv_kernel
        if L == 1 and conv_state is not None:
            # Incremental single-token decoding
            # conv_state: [B, D, k-1], x_step: [B, D, 1]
            x_in = torch.cat([conv_state, x.transpose(1, 2)], dim=-1)  # [B, D, k]
            next_conv_state = x_in[:, :, 1:].detach()
            x_conv = F.silu(self.conv1d(x_in)).transpose(1, 2)         # [B, 1, D]
        else:
            # Sequence prefill or batch training
            x_trans = x.transpose(1, 2)  # [B, D, L]
            if conv_state is not None:
                x_padded = torch.cat([conv_state, x_trans], dim=-1)
            else:
                x_padded = F.pad(x_trans, (k_sz - 1, 0))
            next_conv_state = x_padded[:, :, -(k_sz - 1):].detach() if k_sz > 1 else torch.empty(0, device=x.device)
            x_conv = F.silu(self.conv1d(x_padded)).transpose(1, 2)     # [B, L, D]
            
        x_eff = x + x_conv
        
        # 2. Project Q, K, V
        q = self.q_proj(x_eff).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x_eff).view(B, L, self.num_heads, self.head_dim)
        v = self.v_norm(self.v_proj(x_eff).view(B, L, self.num_heads, self.head_dim))
        
        # Unit-normalize queries and keys for spherical numerical stability
        q = F.normalize(q, p=2, dim=-1, eps=1e-5)
        k = F.normalize(k, p=2, dim=-1, eps=1e-5)
        
        # 3. Compute data-dependent decay and learning rate gates
        alpha = torch.sigmoid(self.alpha_proj(x_eff))  # [B, L, H] decay
        beta = torch.sigmoid(self.beta_proj(x_eff))    # [B, L, H] write rate
        
        # Initialize recurrent state matrix S_0: [B, H, d_k, d_v]
        if S is None:
            S = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
        else:
            S = S.clone()
            
        outputs = []
        
        # 4. Stepwise linear recurrence (Gated Delta Rule):
        # S_t = S_{t-1} * alpha_t + beta_t * (v_t - S_{t-1} @ k_t) @ k_t^T
        for t in range(L):
            if state_reset_mask is not None:
                reset_t = state_reset_mask[:, t, None, None, None]
                S = S * reset_t
            q_t = q[:, t]                      # [B, H, D]
            k_t = k[:, t]                      # [B, H, D]
            v_t = v[:, t]                      # [B, H, D]
            a_t = alpha[:, t, :, None, None]   # [B, H, 1, 1]
            b_t = beta[:, t, :, None, None]    # [B, H, 1, 1]
            
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
        gate = F.silu(self.gate_proj(x_eff))
        y = self.out_proj(out_normed) * gate
        
        return y, (next_conv_state, S)
