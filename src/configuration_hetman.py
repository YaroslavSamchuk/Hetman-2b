"""
Hetman-2.0B: Configuration Specification
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
    global_head_dim: int = 128
    global_attention_heads: int = 12
    global_kv_heads: int = 2  # Grouped Query Attention (GQA)
    
    # Speculative Latent-Hopfield Core (SLH-Core)
    hopfield_num_groups: int = 32
    hopfield_codebook_size: int = 512
    hopfield_val_dim: int = 64
    hopfield_topk_1: int = 8
    hopfield_topk_2: int = 4  # 8 * 4 = 32 active attractor slots per group
    hopfield_subquery_dim: int = 2048
    hopfield_beta: float = 16.0
    
    # Blockwise Fast Walsh-Hadamard Transform (FWHT)
    hadamard_block_size: int = 512
    
    # Long Context & Numerical Bounds
    max_position_embeddings: int = 262144  # 256k context
    qk_norm_bound: float = 11.3137  # sqrt(128)
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0  # YaRN base
    
    # Ternary Quantization Format
    ternary_bits: float = 1.58
    use_hadamard_rotation: bool = True
