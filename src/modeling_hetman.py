"""
Hetman-2.0B: Complete Architecture Implementation
4-Stage Sequential DAG: Denoising Stem, Triad Concurrent DAG, Verified Logic Arbiter, Exit Stem
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from configuration_hetman import HetmanConfig
from quantization import BitLinear
from deltanet import GatedDeltaNet
from hopfield_core import SpeculativeLatentHopfieldCore

class SwiGLUFFN(nn.Module):
    """Ternary SwiGLU Feed-Forward Network (d_ffn = 6144)"""
    def __init__(self, config: HetmanConfig):
        super().__init__()
        self.gate_proj = BitLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = BitLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = BitLinear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class GlobalAttention(nn.Module):
    """
    Global Gated Softmax FlashAttention with Bounded QK-Norm (|S_ij| <= sqrt(128) ≈ 11.31)
    and Grouped-Query Attention (GQA, 12 Q heads, 2 KV heads)
    """
    def __init__(self, config: HetmanConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.global_attention_heads  # 12
        self.num_kv_heads = config.global_kv_heads      # 2
        self.head_dim = config.global_head_dim          # 128
        self.qk_norm_bound = config.qk_norm_bound       # 11.3137
        
        self.q_proj = BitLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = BitLinear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = BitLinear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = BitLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Learnable QK-Norm scaling gains
        self.gamma_q = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.gamma_k = nn.Parameter(torch.ones(self.num_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
        
        # Bounded QK-Norm: Unit-normalize, scale by clamped gamma (bounded by sqrt(head_dim))
        q_norm = F.normalize(q, p=2, dim=-1) * torch.clamp(self.gamma_q, 0.1, math.sqrt(self.head_dim))
        k_norm = F.normalize(k, p=2, dim=-1) * torch.clamp(self.gamma_k, 0.1, math.sqrt(self.head_dim))
        
        # Expand KV heads for GQA (6x repeat)
        k_exp = k_norm.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        v_exp = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        
        # Scaled dot-product attention
        scores = torch.einsum('blhd,bshd->bhls', q_norm, k_exp) / math.sqrt(self.head_dim)
        scores = torch.clamp(scores, -self.qk_norm_bound, self.qk_norm_bound)
        
        if mask is not None:
            scores = scores + mask
        else:
            # Causal lower-triangular mask
            causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
            scores = scores + causal_mask
            
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('bhls,bshd->blhd', attn, v_exp).reshape(B, L, -1)
        
        # Channel gating
        gate = F.silu(self.gate_proj(x))
        return self.out_proj(out) * gate

class HetmanLayer(nn.Module):
    """Single Transformer Block: either DeltaNet (Recurrent) or Global Attention + SwiGLU FFN"""
    def __init__(self, config: HetmanConfig, is_global: bool = False):
        super().__init__()
        self.is_global = is_global
        self.attn = GlobalAttention(config) if is_global else GatedDeltaNet(config)
        self.ffn = SwiGLUFFN(config)
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, prev_state=None):
        norm_x = self.norm1(x)
        if self.is_global:
            attn_out = self.attn(norm_x)
            state = None
        else:
            attn_out, state = self.attn(norm_x, prev_state)
            
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, state

class HetmanModel(nn.Module):
    def __init__(self, config: HetmanConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Stage 1: Denoising Stem (8 Layers: 6 DeltaNet + 2 Global Softmax at layers 4, 8)
        self.stem_layers = nn.ModuleList([
            HetmanLayer(config, is_global=(i in [3, 7])) for i in range(8)
        ])
        
        # Stage 2: Triad Concurrent Processing
        self.hopfield_core = SpeculativeLatentHopfieldCore(config)
        self.domain_backbone = nn.ModuleList([SwiGLUFFN(config) for _ in range(8)])
        self.hypothesis_subnet = nn.ModuleList([HetmanLayer(config, is_global=False) for _ in range(6)])
        
        # Adaptive Dynamic Fusion Gate (3 * 1536 -> 1536)
        self.fusion_gate = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.fusion_proj = BitLinear(3 * config.hidden_size, config.hidden_size, bias=False)
        self.fusion_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Stage 3: Verified Logic Arbiter (16 Layers: 12 DeltaNet + 4 Global at layers 4, 8, 12, 16)
        self.arbiter_layers = nn.ModuleList([
            HetmanLayer(config, is_global=(i in [3, 7, 11, 15])) for i in range(16)
        ])
        
        # Stage 4: Exit Stem (4 Layers DeltaNet)
        self.exit_layers = nn.ModuleList([
            HetmanLayer(config, is_global=False) for _ in range(4)
        ])
        
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor):
        B, L = input_ids.shape
        x = self.embed_tokens(input_ids)
        
        # 1. Stage 1: Denoising Stem
        stem_states = []
        for i, layer in enumerate(self.stem_layers):
            x, st = layer(x)
            if st is not None:
                stem_states.append(st)
        x_stem = x
        
        # 2. Stage 2: Triad Processing
        # Branch A: Hopfield Core fact retrieval
        y_know, hopfield_loss = self.hopfield_core(x_stem)
        
        # Branch B: Continuous Domain Representation (8 SwiGLU layers)
        y_domain = x_stem
        for swiglu in self.domain_backbone:
            y_domain = y_domain + swiglu(y_domain)
            
        # Branch C: Hypothesis Subnetwork (6 layers)
        y_hypo = x_stem
        for layer in self.hypothesis_subnet:
            y_hypo, _ = layer(y_hypo)
            
        # Adaptive Fusion
        y_triad = torch.cat([y_know, y_domain, y_hypo], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(x_stem))
        x_fused = self.fusion_norm(x_stem + self.fusion_proj(y_triad) * gate)
        
        # 3. Stage 3: Logic Arbiter (16 layers)
        x = x_fused
        for layer in self.arbiter_layers:
            x, _ = layer(x)
            
        # 4. Stage 4: Exit Stem (4 layers)
        for layer in self.exit_layers:
            x, _ = layer(x)
            
        x_out = self.final_norm(x)
        return x_out, hopfield_loss

class HetmanForCausalLM(nn.Module):
    def __init__(self, config: HetmanConfig):
        super().__init__()
        self.config = config
        self.model = HetmanModel(config)
        # Weight tying with embeddings
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        hidden_states, hopfield_loss = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            loss = ce_loss + 0.01 * hopfield_loss
            
        return {
            "loss": loss,
            "logits": logits,
            "hopfield_loss": hopfield_loss
        }

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        # Hopfield V is 536,870,912 params, but only Top-32 is active per token
        active_hopfield = 32 * 32 * self.config.hopfield_val_dim  # Top-32 slots active
        inactive_hopfield = self.model.hopfield_core.V.numel() - active_hopfield
        active_params = total_params - inactive_hopfield
        return total_params, active_params
