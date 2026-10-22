"""
Hetman-2.0B: Configuration Specification (Frontier Edition)
Compatible with Hugging Face AutoConfig and PretrainedConfig
"""

try:
    from transformers import PretrainedConfig
except ImportError:
    class PretrainedConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

class HetmanConfig(PretrainedConfig):
    model_type = "hetman"
    
    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 1536,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 34,
        num_deltanet_layers: int = 28,
        num_global_layers: int = 6,
        num_attention_heads: int = 16,
        deltanet_head_dim: int = 64,
        deltanet_conv_kernel: int = 4,
        global_head_dim: int = 128,
        global_attention_heads: int = 12,
        global_kv_heads: int = 2,
        hopfield_num_groups: int = 16,
        hopfield_codebook_size: int = 1024,
        hopfield_val_dim: int = 32,
        hopfield_expanded_dim: int = 128,
        hopfield_topk_1: int = 8,
        hopfield_topk_2: int = 4,
        hopfield_shared_attractors: int = 1,
        hopfield_subquery_dim: int = 2048,
        hopfield_beta: float = 0.1767767,
        hopfield_pid_ki: float = 0.01,
        hopfield_pid_kd: float = 0.005,
        hadamard_block_size: int = 512,
        swiglu_limit: float = 10.0,
        logit_softcap: float = 30.0,
        max_position_embeddings: int = 262144,
        qk_norm_bound: float = 11.3137,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000000.0,
        ternary_bits: float = 1.58,
        use_hadamard_rotation: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_deltanet_layers = num_deltanet_layers
        self.num_global_layers = num_global_layers
        self.num_attention_heads = num_attention_heads
        self.deltanet_head_dim = deltanet_head_dim
        self.deltanet_conv_kernel = deltanet_conv_kernel
        self.global_head_dim = global_head_dim
        self.global_attention_heads = global_attention_heads
        self.global_kv_heads = global_kv_heads
        self.hopfield_num_groups = hopfield_num_groups
        self.hopfield_codebook_size = hopfield_codebook_size
        self.hopfield_val_dim = hopfield_val_dim
        self.hopfield_expanded_dim = hopfield_expanded_dim
        self.hopfield_topk_1 = hopfield_topk_1
        self.hopfield_topk_2 = hopfield_topk_2
        self.hopfield_shared_attractors = hopfield_shared_attractors
        self.hopfield_subquery_dim = hopfield_subquery_dim
        self.hopfield_beta = hopfield_beta
        self.hopfield_pid_ki = hopfield_pid_ki
        self.hopfield_pid_kd = hopfield_pid_kd
        self.hadamard_block_size = hadamard_block_size
        self.swiglu_limit = swiglu_limit
        self.logit_softcap = logit_softcap
        self.max_position_embeddings = max_position_embeddings
        self.qk_norm_bound = qk_norm_bound
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.ternary_bits = ternary_bits
        self.use_hadamard_rotation = use_hadamard_rotation
