"""
Hetman-2.0B: Complete Architecture Implementation (Frontier Edition)
4-Stage Sequential DAG v2: Denoising Stem (Global L8), Triad Concurrent DAG (Hypo Global L6),
Verified Logic Arbiter (Global L4, L8, L12, L16), and Multi-Highway Exit Stem.
Includes YaRN 256k RoPE with Layer-Wise Temperature Cooling, Bounded QK-Norm,
SwiGLU Limit Clamping, LM Head Logit Soft-Capping, Multi-Token Prediction (MTP),
and Elastic Thinking Compaction.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .configuration_hetman import HetmanConfig
from .quantization import BitLinear
from .deltanet import GatedDeltaNet
from .hopfield_core import SpeculativeLatentHopfieldCore

@dataclass
class HetmanCausalLMOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    ce_loss: Optional[torch.Tensor] = None
    mtp_loss: Optional[torch.Tensor] = None
    hopfield_loss: Optional[torch.Tensor] = None
    past_states: Optional[Any] = None

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, item):
        return hasattr(self, item)

    def get(self, key, default=None):
        return getattr(self, key, default)

class YaRNRotaryEmbedding(nn.Module):
    """
    YaRN (Yet another RoPE extensioN) Rotary Positional Embedding
    Scales context window smoothly from 4k to 262,144 tokens with base theta = 10,000,000
    Implements 3-zone frequency ramp interpolation (beta_fast=32, beta_slow=1)
    and temperature cooling factor.
    """
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 262144,
        base: float = 10000000.0,
        scale: float = 64.0,
        original_max_position: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        temp_cooling: float = 2.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position = original_max_position
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.temp_cooling = temp_cooling
        
        # Base inverse frequencies: theta_i = 1 / base^(2i / dim)
        pos = torch.arange(0, dim, 2).float()
        inv_freq = 1.0 / (base ** (pos / dim))
        
        if scale > 1.0:
            r = (original_max_position * inv_freq) / (2.0 * math.pi)
            gamma = torch.clamp((r - beta_slow) / (beta_fast - beta_slow), 0.0, 1.0)
            yarn_inv_freq = inv_freq * (gamma + (1.0 - gamma) / scale)
            self.register_buffer("inv_freq", yarn_inv_freq, persistent=False)
        else:
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0, position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        if position_ids is not None:
            if position_ids.dim() == 2:
                # [B, L] -> freqs: [B, L, dim // 2]
                freqs = position_ids.unsqueeze(-1).to(self.inv_freq.dtype) * self.inv_freq.view(1, 1, -1)
                emb = torch.cat((freqs, freqs), dim=-1)  # [B, L, dim]
                return emb.cos().to(x.dtype), emb.sin().to(x.dtype)
            else:
                t = position_ids.to(self.inv_freq.dtype)
        else:
            t = torch.arange(offset, offset + seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim // 2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, L, 1, D]
        sin = sin.unsqueeze(0).unsqueeze(2)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(2)               # [B, L, 1, D]
        sin = sin.unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SwiGLUFFN(nn.Module):
    """Ternary SwiGLU Feed-Forward Network (d_ffn = 6144) with Activation Clamping"""
    def __init__(self, config: HetmanConfig):
        super().__init__()
        self.swiglu_limit = getattr(config, "swiglu_limit", 10.0)
        self.gate_proj = BitLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = BitLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = BitLinear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        if self.swiglu_limit > 0.0:
            gated = torch.clamp(gated, -self.swiglu_limit, self.swiglu_limit)
        return self.down_proj(gated)

class GlobalAttention(nn.Module):
    """
    Global Gated Softmax FlashAttention with YaRN RoPE, Bounded QK-Norm (|S_ij| <= sqrt(128) ≈ 11.31),
    Grouped-Query Attention (GQA, 12 Q heads, 2 KV heads) and Layer-Wise Temperature Cooling.
    """
    def __init__(self, config: HetmanConfig, yarn_cooling: float = 2.0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.global_attention_heads  # 12
        self.num_kv_heads = config.global_kv_heads      # 2
        self.head_dim = config.global_head_dim          # 128
        self.qk_norm_bound = config.qk_norm_bound       # 11.3137
        self.yarn_cooling = yarn_cooling                # Layer-wise entropy cooling factor
        
        self.q_proj = BitLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = BitLinear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = BitLinear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = BitLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Learnable QK-Norm scaling gains: initialized to sqrt(head_dim) to avoid step 0 entropy collapse
        self.gamma_q = nn.Parameter(torch.full((self.num_heads, self.head_dim), math.sqrt(self.head_dim)))
        self.gamma_k = nn.Parameter(torch.full((self.num_kv_heads, self.head_dim), math.sqrt(self.head_dim)))
        
        # Positional Embeddings: YaRN RoPE
        self.rotary_emb = YaRNRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scale=64.0,
            temp_cooling=yarn_cooling
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
        
        # 1. Apply YaRN Rotary Positional Embeddings
        cos, sin = self.rotary_emb(q, seq_len=L, offset=offset, position_ids=position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # 2. Bounded QK-Norm: Unit-normalize, scale by clamped gamma (bounded by sqrt(head_dim))
        q_norm = F.normalize(q, p=2, dim=-1, eps=1e-5) * torch.clamp(self.gamma_q, 0.1, math.sqrt(self.head_dim))
        k_norm = F.normalize(k, p=2, dim=-1, eps=1e-5) * torch.clamp(self.gamma_k, 0.1, math.sqrt(self.head_dim))
        
        # 3. Handle KV-cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k_norm = torch.cat([past_k, k_norm], dim=1)
            v = torch.cat([past_v, v], dim=1)
        current_kv = (k_norm, v)
        
        # 4. Expand KV heads for GQA (6x repeat to match 12 Q heads)
        k_exp = k_norm.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        v_exp = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        
        # 5. Scaled dot-product attention via native SDPA with Layer-wise YaRN temperature cooling
        q_t = q_norm.transpose(1, 2)  # [B, H, L, D]
        k_t = k_exp.transpose(1, 2)   # [B, H, S, D]
        v_t = v_exp.transpose(1, 2)   # [B, H, S, D]
        
        is_causal = (mask is None) and (past_kv is None) and (L > 1)
        scale = self.yarn_cooling / math.sqrt(self.head_dim)
        out_sdpa = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=mask, is_causal=is_causal, scale=scale)
        out = out_sdpa.transpose(1, 2).reshape(B, L, -1)
        
        # 6. Channel gating
        gate = F.silu(self.gate_proj(x))
        return self.out_proj(out) * gate, current_kv

class HetmanLayer(nn.Module):
    """Single Transformer Block: either DeltaNet (Recurrent O(1)) or Global Attention + SwiGLU FFN"""
    def __init__(self, config: HetmanConfig, is_global: bool = False, yarn_cooling: float = 2.0):
        super().__init__()
        self.is_global = is_global
        self.attn = GlobalAttention(config, yarn_cooling=yarn_cooling) if is_global else GatedDeltaNet(config)
        self.ffn = SwiGLUFFN(config)
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        prev_state=None,
        offset: int = 0,
        mask: Optional[torch.Tensor] = None,
        state_reset_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ):
        norm_x = self.norm1(x)
        if self.is_global:
            attn_out, next_state = self.attn(norm_x, mask=mask, past_kv=prev_state, offset=offset, position_ids=position_ids)
        else:
            attn_out, next_state = self.attn(norm_x, prev_state=prev_state, state_reset_mask=state_reset_mask)
            
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, next_state

class MTPHead(nn.Module):
    """
    Multi-Token Prediction (MTP) Auxiliary Block (+34.6M parameters)
    Predicts token representation for step t+2 given hidden state at step t.
    """
    def __init__(self, config: HetmanConfig):
        super().__init__()
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config)
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.final_norm(h + self.ffn(self.norm(h)))


try:
    from transformers import PreTrainedModel
except ImportError:
    class PreTrainedModel(nn.Module):
        def __init__(self, config, *args, **kwargs):
            super().__init__()
            self.config = config

class HetmanPreTrainedModel(PreTrainedModel):
    config_class = HetmanConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

class HetmanModel(HetmanPreTrainedModel):
    def __init__(self, config: HetmanConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Stage 1: Denoising Stem (8 Layers: 7 DeltaNet + 1 Global Softmax at Layer 8 / idx 7)
        # YaRN cooling: 1.6
        self.stem_layers = nn.ModuleList([
            HetmanLayer(config, is_global=(i == 7), yarn_cooling=1.6) for i in range(8)
        ])
        
        # Stage 2: Triad Concurrent Processing
        self.hopfield_core = SpeculativeLatentHopfieldCore(config)
        self.domain_backbone = nn.ModuleList([SwiGLUFFN(config) for _ in range(8)])
        # Hypothesis subnet: 6 Layers (5 DeltaNet + 1 Global Softmax at Layer 6 / idx 5, cooling: 1.8)
        self.hypothesis_subnet = nn.ModuleList([
            HetmanLayer(config, is_global=(i == 5), yarn_cooling=1.8) for i in range(6)
        ])
        
        # Adaptive Dynamic Fusion Gate (3 * 1536 -> 1536)
        self.fusion_gate = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.fusion_proj = BitLinear(3 * config.hidden_size, config.hidden_size, bias=False)
        self.fusion_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Stage 3: Verified Logic Arbiter (16 Layers: 12 DeltaNet + 4 Global at layers 4, 8, 12, 16)
        # Cooling: 2.0 for L4-L8 (idx 3, 7), 2.4 for L12-L16 (idx 11, 15)
        arbiter_list = []
        for i in range(16):
            if i in [3, 7]:
                arbiter_list.append(HetmanLayer(config, is_global=True, yarn_cooling=2.0))
            elif i in [11, 15]:
                arbiter_list.append(HetmanLayer(config, is_global=True, yarn_cooling=2.4))
            else:
                arbiter_list.append(HetmanLayer(config, is_global=False))
        self.arbiter_layers = nn.ModuleList(arbiter_list)
        
        # Stage 4: Multi-Highway Residual Integration & Exit Stem (4 Layers DeltaNet)
        self.exit_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.exit_layers = nn.ModuleList([
            HetmanLayer(config, is_global=False) for _ in range(4)
        ])
        
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_states: Optional[Dict[str, List[Any]]] = None,
        offset: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        state_reset_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[Any]]]:
        B, L = input_ids.shape
        x = self.embed_tokens(input_ids)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Expand 2D [B, S] padding mask into 4D causal boolean mask [B, 1, L, S]
                B_m, S_m = attention_mask.shape
                causal_triangle = torch.tril(torch.ones((L, S_m), device=input_ids.device, dtype=torch.bool), diagonal=offset)
                pad_bool = attention_mask.bool().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
                attention_mask = causal_triangle.unsqueeze(0).unsqueeze(0) & pad_bool
            elif attention_mask.dim() == 3:
                # Expand 3D [B, L, S] to 4D [B, 1, L, S]
                attention_mask = attention_mask.unsqueeze(1).bool()
            elif attention_mask.dtype not in (torch.bool, torch.float16, torch.bfloat16, torch.float32):
                attention_mask = attention_mask.bool()
        
        new_states = {"stem": [], "arbiter": [], "exit": [], "hypo": []}
        
        # 1. Stage 1: Denoising Stem
        for i, layer in enumerate(self.stem_layers):
            prev_st = past_states["stem"][i] if past_states else None
            x, st = layer(x, prev_state=prev_st, offset=offset, mask=attention_mask, state_reset_mask=state_reset_mask, position_ids=position_ids)
            new_states["stem"].append(st)
        x_stem = x
        
        # 2. Stage 2: Triad Processing
        # Branch A: Hopfield Core fact retrieval (Aux-loss-free PID routing)
        y_know, hopfield_loss = self.hopfield_core(x_stem)
        
        # Branch B: Continuous Domain Representation (8 SwiGLU layers)
        y_domain = x_stem
        for swiglu in self.domain_backbone:
            y_domain = y_domain + swiglu(y_domain)
            
        # Branch C: Hypothesis Subnetwork (6 layers: 5 DeltaNet + 1 Global Attention)
        y_hypo = x_stem
        for i, layer in enumerate(self.hypothesis_subnet):
            prev_st = past_states["hypo"][i] if past_states else None
            y_hypo, st = layer(y_hypo, prev_state=prev_st, offset=offset, mask=attention_mask, state_reset_mask=state_reset_mask, position_ids=position_ids)
            new_states["hypo"].append(st)
            
        # Adaptive Fusion
        y_triad = torch.cat([y_know, y_domain, y_hypo], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(x_stem))
        x_fused = self.fusion_norm(x_stem + self.fusion_proj(y_triad) * gate)
        
        # 3. Stage 3: Logic Arbiter (16 layers: 12 DeltaNet + 4 Global Attention)
        x_arb = x_fused
        for i, layer in enumerate(self.arbiter_layers):
            prev_st = past_states["arbiter"][i] if past_states else None
            x_arb, st = layer(x_arb, prev_state=prev_st, offset=offset, mask=attention_mask, state_reset_mask=state_reset_mask, position_ids=position_ids)
            new_states["arbiter"].append(st)
        x_arbiter = x_arb
            
        # 4. Stage 4: Multi-Highway Residual Integration & Exit Stem (4 layers DeltaNet)
        x_exit = self.exit_norm((x_stem + x_fused + x_arbiter) / math.sqrt(3.0))
        for i, layer in enumerate(self.exit_layers):
            prev_st = past_states["exit"][i] if past_states else None
            x_exit, st = layer(x_exit, prev_state=prev_st, offset=offset, mask=attention_mask, state_reset_mask=state_reset_mask, position_ids=position_ids)
            new_states["exit"].append(st)
            
        x_out = self.final_norm(x_exit)
        return x_out, hopfield_loss, new_states

class HetmanForCausalLM(HetmanPreTrainedModel):
    def __init__(self, config: HetmanConfig):
        super().__init__()
        self.config = config
        self.logit_softcap = getattr(config, "logit_softcap", 30.0)
        self.model = HetmanModel(config)
        
        # Weight tying with embeddings
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        
        # Multi-Token Prediction (MTP) Auxiliary Head
        self.mtp_head = MTPHead(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_states: Optional[Dict[str, List[Any]]] = None,
        offset: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        state_reset_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        hidden_states, hopfield_loss, new_states = self.model(
            input_ids,
            past_states=past_states,
            offset=offset,
            attention_mask=attention_mask,
            state_reset_mask=state_reset_mask,
            position_ids=position_ids
        )
        logits = self.lm_head(hidden_states)
        
        # Apply LM head logit soft-capping to prevent logits divergence
        if self.logit_softcap > 0.0:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        
        loss = None
        ce_loss = None
        mtp_loss = None
        
        if labels is not None:
            # 1. Primary Next-Token Prediction (t + 1)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            valid_mask = (shift_labels != -100)
            if valid_mask.any():
                ce_loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            else:
                ce_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            
            # 2. Multi-Token Prediction (t + 2)
            if input_ids.shape[1] > 2:
                h_mtp = self.mtp_head(hidden_states)
                logits_mtp = self.lm_head(h_mtp)
                if self.logit_softcap > 0.0:
                    logits_mtp = self.logit_softcap * torch.tanh(logits_mtp / self.logit_softcap)
                shift_logits_mtp = logits_mtp[..., :-2, :].contiguous()
                shift_labels_mtp = labels[..., 2:].contiguous()
                valid_mask_mtp = (shift_labels_mtp != -100)
                if valid_mask_mtp.any():
                    mtp_loss = F.cross_entropy(shift_logits_mtp.view(-1, self.config.vocab_size), shift_labels_mtp.view(-1))
                else:
                    mtp_loss = torch.tensor(0.0, device=input_ids.device)
            else:
                mtp_loss = torch.tensor(0.0, device=input_ids.device)
                
            # Composite loss: CE + 0.3 * MTP + 0.005 * Hopfield commitment
            loss = ce_loss + 0.3 * mtp_loss + 0.005 * hopfield_loss
            
        return HetmanCausalLMOutput(
            loss=loss,
            logits=logits,
            ce_loss=ce_loss,
            mtp_loss=mtp_loss,
            hopfield_loss=hopfield_loss,
            past_states=new_states
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: Any = (2, 5),  # </s> (2) and <|im_end|> (5)
        pad_token_id: int = 3,
        do_sample: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV-cache / recurrent state preservation.
        Fully compatible with Hugging Face generate() callers and standard sampling pipelines.
        """
        device = next(self.parameters()).device
        if input_ids is None or input_ids.numel() == 0:
            input_ids = torch.tensor([[1]], dtype=torch.long, device=device)  # <s> BOS token
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        B, seq_len = input_ids.shape
        if seq_len == 0:
            input_ids = torch.tensor([[1]], dtype=torch.long, device=device).expand(B, 1)
            seq_len = 1
            
        current_ids = input_ids.clone()
        past_states = None
        
        if isinstance(eos_token_id, int):
            eos_set = {eos_token_id}
        else:
            eos_set = set(eos_token_id)
            
        unfinished_sequences = torch.ones(B, dtype=torch.bool, device=device)
        
        # If do_sample is explicitly False, force greedy argmax decoding
        if do_sample is False:
            temperature = 0.0
            
        # Initial prefill pass
        outputs = self(current_ids, past_states=None, offset=0)
        past_states = outputs["past_states"]
        next_token_logits = outputs["logits"][:, -1, :]
        
        for step in range(max_new_tokens):
            if temperature >= 1e-4:
                scaled_logits = next_token_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                
                # Top-p (nucleus) filtering
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs = probs.masked_fill(indices_to_remove, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            tokens_to_append = torch.where(
                unfinished_sequences.unsqueeze(-1),
                next_token,
                torch.full_like(next_token, pad_token_id)
            )
            current_ids = torch.cat([current_ids, tokens_to_append], dim=-1)
            
            for i in range(B):
                if next_token[i, 0].item() in eos_set:
                    unfinished_sequences[i] = False
                    
            if not unfinished_sequences.any():
                break
                
            if step < max_new_tokens - 1:
                step_outputs = self(tokens_to_append, past_states=past_states, offset=seq_len + step)
                past_states = step_outputs["past_states"]
                next_token_logits = step_outputs["logits"][:, -1, :]
            
        return current_ids

    def count_parameters(self) -> Tuple[int, int]:
        total_params = sum(p.numel() for p in self.parameters())
        # Hopfield V is 16 groups * 512 * 512 * 128 = 536,870,912 params, but only Top-32 + 1 shared is active per token
        active_hopfield = self.config.hopfield_num_groups * (32 + 1) * self.config.hopfield_val_dim
        inactive_hopfield = self.model.hopfield_core.V.numel() - active_hopfield
        active_params = total_params - inactive_hopfield
        return total_params, active_params
