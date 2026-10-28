"""
Hetman-2.0B: Configuration Specification (Frontier Edition)
"""

from dataclasses import dataclass

@dataclass
class HetmanConfig:
    # Model Identity & Dimensions
    model_type: str = "hetman"
    vocab_size: int = 65536
    hidden_size: int = 1536
    intermediate_size: int = 6144
    num_hidden_layers: int = 34
    
    # Topology: 28 Gated DeltaNet + 6 Global Softmax FlashAttention
    num_deltanet_layers: int = 28
    num_global_layers: int = 6
    num_attention_heads: int = 16
    deltanet_head_dim: int = 64
    deltanet_conv_kernel: int = 4         # Causal 1D depthwise conv kernel for multi-token morphology
    global_head_dim: int = 128
    global_attention_heads: int = 12
    global_kv_heads: int = 2             # Grouped Query Attention (GQA 12:2)
    
    # Speculative Latent-Hopfield Core v2 (SLH-Core v2)
    # Tensor V: [16, 1024, 1024, 32] = 536,870,912 elements (16.78M Attractor Slots, 100% TPU MXU aligned)
    hopfield_num_groups: int = 16
    hopfield_codebook_size: int = 1024
    hopfield_val_dim: int = 32
    hopfield_expanded_dim: int = 128     # Projected to 128 via W_expand for TPU MXU 128x128 systolic alignment
    hopfield_topk_1: int = 8
    hopfield_topk_2: int = 4             # 8 * 4 = 32 active dynamic slots per group
    hopfield_shared_attractors: int = 1  # 1 shared anchor attractor slot per group
    hopfield_subquery_dim: int = 2048    # 16 groups * 2 * 64 = 2048
    hopfield_beta: float = 0.1767767     # 1 / sqrt(32) soft-attractor temperature
    hopfield_pid_ki: float = 0.01        # Aux-Loss-Free PID dynamic bias integral gain
    hopfield_pid_kd: float = 0.005       # Aux-Loss-Free PID dynamic bias derivative gain
    hopfield_ortho_weight: float = 0.005 # Declarative (Hopfield) vs Procedural (Logic) orthogonality loss weight
    
    # Blockwise Fast Walsh-Hadamard Transform (FWHT) & Outlier Suppression
    hadamard_block_size: int = 512
    swiglu_limit: float = 10.0           # SwiGLU activation clamp for numerical stability
    logit_softcap: float = 30.0          # LM Head logit soft-capping (30.0 * tanh(logits / 30.0))
    
    # Long Context & Numerical Bounds
    max_position_embeddings: int = 262144  # 256k context
    qk_norm_bound: float = 11.3137       # sqrt(128)
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0       # YaRN base
    
    # Ternary Quantization Format
    ternary_bits: float = 1.58
    use_hadamard_rotation: bool = True
