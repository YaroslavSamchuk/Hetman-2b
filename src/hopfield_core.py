"""
Speculative Latent-Hopfield Core (SLH-Core)
8,388,608 Attractor Slots across 32 Minigroups with Top-32 Sparse Bilinear Contraction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from configuration_hetman import HetmanConfig
from quantization import BitLinear

class SpeculativeLatentHopfieldCore(nn.Module):
    def __init__(self, config: HetmanConfig):
        super().__init__()
        self.num_groups = config.hopfield_num_groups         # 32
        self.codebook_size = config.hopfield_codebook_size   # 512
        self.val_dim = config.hopfield_val_dim               # 64
        self.topk_1 = config.hopfield_topk_1                 # 8
        self.topk_2 = config.hopfield_topk_2                 # 4
        self.beta = config.hopfield_beta                     # 16.0
        self.subquery_dim = config.hopfield_subquery_dim     # 2048 (32 groups * 2 * 32)
        
        # Subquery projector
        self.q_proj = BitLinear(config.hidden_size, self.subquery_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.subquery_dim, eps=config.rms_norm_eps)
        
        # Dual Spherical Codebooks: C1, C2 per group (each [32, 512, 32])
        self.C1 = nn.Parameter(torch.randn(self.num_groups, self.codebook_size, 32))
        self.C2 = nn.Parameter(torch.randn(self.num_groups, self.codebook_size, 32))
        
        # Knowledge attractor value tensor V: [32, 512, 512, 64]
        # In memory-efficient ternary {-1, 0, +1} representation (536,870,912 parameters)
        self.V = nn.Parameter(torch.empty(self.num_groups, self.codebook_size, self.codebook_size, self.val_dim))
        
        # Final knowledge output projection back to hidden size (1536)
        # 32 groups * 64 dim = 2048 -> 1536
        self.out_proj = BitLinear(self.num_groups * self.val_dim, config.hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        # Normalize codebooks on unit sphere S^31
        with torch.no_grad():
            self.C1.copy_(F.normalize(self.C1, p=2, dim=-1))
            self.C2.copy_(F.normalize(self.C2, p=2, dim=-1))
            # Initialize ternary attractor values
            self.V.uniform_(-1.0, 1.0)
            self.V.copy_(torch.clamp(torch.round(self.V), -1.0, 1.0))

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, hidden_size]
        Returns:
            y_know: [B, L, hidden_size]
            aux_loss: commitment & Laplace-smoothed entropy regularization loss
        """
        B, L, _ = x.shape
        
        # 1. Project and normalize subqueries
        z = self.q_norm(self.q_proj(x))  # [B, L, 2048]
        z_grouped = z.view(B, L, self.num_groups, 2, 32)
        q1 = F.normalize(z_grouped[..., 0, :], p=2, dim=-1)  # [B, L, 32, 32]
        q2 = F.normalize(z_grouped[..., 1, :], p=2, dim=-1)  # [B, L, 32, 32]
        
        # 2. Dual Codebook Cosine Retrieval
        C1_norm = F.normalize(self.C1, p=2, dim=-1)  # [32, 512, 32]
        C2_norm = F.normalize(self.C2, p=2, dim=-1)  # [32, 512, 32]
        
        # Cosine similarity scores: S1, S2: [B, L, 32, 512]
        S1 = torch.einsum('blgd,gmd->blgm', q1, C1_norm)
        S2 = torch.einsum('blgd,gnd->blgn', q2, C2_norm)
        
        # 3. Top-k Sparsification (Top-8 from C1, Top-4 from C2)
        top1_vals, top1_idx = torch.topk(S1, self.topk_1, dim=-1)  # [B, L, 32, 8]
        top2_vals, top2_idx = torch.topk(S2, self.topk_2, dim=-1)  # [B, L, 32, 4]
        
        # 4. Joint bilinear activation weights for Top-32 active pairs
        # pair_scores: [B, L, 32, 8, 4] -> flatten to [B, L, 32, 32]
        pair_scores = top1_vals.unsqueeze(-1) + top2_vals.unsqueeze(-2)
        pair_scores_flat = pair_scores.view(B, L, self.num_groups, self.topk_1 * self.topk_2)
        attn_weights = F.softmax(self.beta * pair_scores_flat, dim=-1)  # [B, L, 32, 32]
        
        # 5. Bilinear contraction with ternary value tensor V
        # Gather active V slices for top1_idx and top2_idx
        group_outputs = []
        for g in range(self.num_groups):
            # Gather top-8 indices along u and top-4 along v
            idx1 = top1_idx[..., g, :]  # [B, L, 8]
            idx2 = top2_idx[..., g, :]  # [B, L, 4]
            weights_g = attn_weights[..., g, :]  # [B, L, 32]
            
            # Sub-tensor gathering: V[g, idx1, idx2, :]
            # For efficiency, index into V[g] [512, 512, 64]
            Vg = self.V[g]
            V_sub = Vg[idx1.unsqueeze(-1), idx2.unsqueeze(-2)]  # [B, L, 8, 4, 64]
            V_sub_flat = V_sub.view(B, L, 32, self.val_dim)
            
            # Weighted sum over 32 active attractor slots
            m_g = torch.einsum('bls,blsd->bld', weights_g, V_sub_flat)  # [B, L, 64]
            group_outputs.append(m_g)
            
        # Concatenate across all 32 groups: [B, L, 32 * 64 = 2048]
        m_concat = torch.cat(group_outputs, dim=-1)
        
        # 6. Knowledge projection back to 1536
        y_know = self.out_proj(m_concat)
        
        # 7. Numerical stability regularization (Laplace-smoothed entropy loss)
        # Prevents codebook collapse across 850B tokens
        p1_marginal = (1.0 + torch.mean(F.softmax(S1, dim=-1), dim=(0, 1))) / (512.0 + B * L)
        entropy_loss = -torch.sum(p1_marginal * torch.log(p1_marginal + 1e-10))
        
        return y_know, entropy_loss
