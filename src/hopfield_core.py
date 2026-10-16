"""
Speculative Latent-Hopfield Core v2 (SLH-Core v2)
16,777,216 Attractor Slots across 16 Groups (Val Dim 32 expanded to 128) with Top-32 Sparse Bilinear Contraction,
Shared Anchor Attractors, and Aux-Loss-Free PID Dynamic Bias Load Balancing.
"""

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from configuration_hetman import HetmanConfig
from quantization import BitLinear

class SpeculativeLatentHopfieldCore(nn.Module):
    def __init__(self, config: HetmanConfig):
        super().__init__()
        self.num_groups = config.hopfield_num_groups         # 16 groups
        self.codebook_size = config.hopfield_codebook_size   # 1024
        self.val_dim = config.hopfield_val_dim               # 32 (fine-grained atomic micro-facts)
        self.expanded_dim = getattr(config, "hopfield_expanded_dim", 128)  # 128 (TPU MXU systolic alignment)
        self.topk_1 = config.hopfield_topk_1                 # 8
        self.topk_2 = config.hopfield_topk_2                 # 4 (8 * 4 = 32 dynamic slots per group)
        self.beta = config.hopfield_beta                     # 1 / sqrt(32) ≈ 0.1768
        self.code_dim = 64                                   # 64-dim spherical code vectors on S^63
        self.subquery_dim = self.num_groups * 2 * self.code_dim  # 16 * 2 * 64 = 2048
        
        self.pid_ki = getattr(config, "hopfield_pid_ki", 0.01)
        self.pid_kd = getattr(config, "hopfield_pid_kd", 0.005)
        
        # Subquery projector & normalizer
        self.q_proj = BitLinear(config.hidden_size, self.subquery_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.subquery_dim, eps=config.rms_norm_eps)
        
        # Dual Spherical Codebooks: C1, C2 per group (each [16, 1024, 64])
        self.C1 = nn.Parameter(torch.randn(self.num_groups, self.codebook_size, self.code_dim))
        self.C2 = nn.Parameter(torch.randn(self.num_groups, self.codebook_size, self.code_dim))
        
        # Aux-Loss-Free PID dynamic router biases (non-differentiable state)
        self.register_buffer("b1", torch.zeros(self.num_groups, self.codebook_size))
        self.register_buffer("b2", torch.zeros(self.num_groups, self.codebook_size))
        self.register_buffer("prev_err1", torch.zeros(self.num_groups, self.codebook_size))
        self.register_buffer("prev_err2", torch.zeros(self.num_groups, self.codebook_size))
        
        # Knowledge attractor value tensor V: [16, 1024, 1024, 32]
        # In memory-efficient ternary {-1, 0, +1} representation (536,870,912 parameters, 16.78M slots)
        self.V = nn.Parameter(torch.empty(self.num_groups, self.codebook_size, self.codebook_size, self.val_dim))
        
        # Linear expansion projection W_expand: [16, 32, 128]
        # Maps 32-dim micro-fact attractor combinations to 128-dim for 100% systolic TPU MXU saturation
        self.W_expand = nn.Parameter(torch.empty(self.num_groups, self.val_dim, self.expanded_dim))
        
        # 1 Shared Anchor Attractor slot per group for baseline representation stability
        self.V_shared = nn.Parameter(torch.empty(self.num_groups, self.expanded_dim))
        
        # Final knowledge output projection back to hidden size (1536)
        # 16 groups * 128 dim = 2048 -> 1536
        self.out_proj = BitLinear(self.num_groups * self.expanded_dim, config.hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        # Normalize codebooks on unit sphere S^63
        with torch.no_grad():
            self.C1.copy_(F.normalize(self.C1, p=2, dim=-1, eps=1e-5))
            self.C2.copy_(F.normalize(self.C2, p=2, dim=-1, eps=1e-5))
            # Initialize ternary attractor values
            self.V.uniform_(-1.0, 1.0)
            self.V.copy_(torch.clamp(torch.round(self.V), -1.0, 1.0))
            # Initialize linear expansion projections semi-orthogonally to preserve norm
            for g in range(self.num_groups):
                nn.init.orthogonal_(self.W_expand[g])
            # Initialize shared anchor attractors
            self.V_shared.normal_(mean=0.0, std=1.0 / math.sqrt(self.expanded_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, L, hidden_size]
        Returns:
            y_know: [B, L, hidden_size]
            aux_loss: commitment regularization loss (with Aux-Loss-Free PID routing)
        """
        B, L, _ = x.shape
        
        # 1. Project and normalize subqueries
        z = self.q_norm(self.q_proj(x))  # [B, L, 2048]
        z_grouped = z.view(B, L, self.num_groups, 2, self.code_dim)
        q1 = F.normalize(z_grouped[..., 0, :], p=2, dim=-1, eps=1e-5)  # [B, L, 16, 64]
        q2 = F.normalize(z_grouped[..., 1, :], p=2, dim=-1, eps=1e-5)  # [B, L, 16, 64]
        
        # 2. Dual Codebook Cosine Retrieval
        C1_norm = F.normalize(self.C1, p=2, dim=-1, eps=1e-5)  # [16, 512, 64]
        C2_norm = F.normalize(self.C2, p=2, dim=-1, eps=1e-5)  # [16, 512, 64]
        
        # Raw cosine similarity scores: S1, S2: [B, L, 16, 512]
        S1_raw = torch.einsum('blgd,gmd->blgm', q1, C1_norm)
        S2_raw = torch.einsum('blgd,gnd->blgn', q2, C2_norm)
        
        # Add non-differentiable dynamic PID bias for Aux-Loss-Free load balancing
        S1 = S1_raw + self.b1.unsqueeze(0).unsqueeze(0)
        S2 = S2_raw + self.b2.unsqueeze(0).unsqueeze(0)
        
        # 3. Top-k Sparsification (Top-8 from C1, Top-4 from C2)
        top1_vals, top1_idx = torch.topk(S1, self.topk_1, dim=-1)  # [B, L, 16, 8]
        top2_vals, top2_idx = torch.topk(S2, self.topk_2, dim=-1)  # [B, L, 16, 4]
        
        # Update PID biases in training mode without backpropagating auxiliary loss gradients
        if self.training:
            with torch.no_grad():
                total_tokens = max(B * L, 1)
                # Compute empirical selection frequencies: f_u = count / total_tokens
                # One-hot aggregation over Top-k
                f1 = torch.zeros(self.num_groups, self.codebook_size, device=x.device)
                f2 = torch.zeros(self.num_groups, self.codebook_size, device=x.device)
                
                ones1 = torch.ones_like(top1_idx, dtype=torch.float32)
                f1.scatter_add_(1, top1_idx.view(-1, self.num_groups, self.topk_1).permute(1, 0, 2).reshape(self.num_groups, -1),
                                ones1.view(-1, self.num_groups, self.topk_1).permute(1, 0, 2).reshape(self.num_groups, -1))
                f1 = f1 / total_tokens
                
                ones2 = torch.ones_like(top2_idx, dtype=torch.float32)
                f2.scatter_add_(1, top2_idx.view(-1, self.num_groups, self.topk_2).permute(1, 0, 2).reshape(self.num_groups, -1),
                                ones2.view(-1, self.num_groups, self.topk_2).permute(1, 0, 2).reshape(self.num_groups, -1))
                f2 = f2 / total_tokens
                
                # Target frequency per slot
                target_f1 = self.topk_1 / float(self.codebook_size)
                target_f2 = self.topk_2 / float(self.codebook_size)
                
                # Normalized relative error avoids quadratic asymmetry and slot starvation
                err1 = (f1 - target_f1) / target_f1
                err2 = (f2 - target_f2) / target_f2
                
                d_err1 = err1 - self.prev_err1
                d_err2 = err2 - self.prev_err2
                
                # Symmetric normalized PID update
                self.b1.sub_(self.pid_ki * err1 + self.pid_kd * d_err1)
                self.b2.sub_(self.pid_ki * err2 + self.pid_kd * d_err2)
                
                # Clamp biases to prevent runaway drift
                self.b1.clamp_(-0.5, 0.5)
                self.b2.clamp_(-0.5, 0.5)
                
                self.prev_err1.copy_(err1)
                self.prev_err2.copy_(err2)
        
        # 4. Joint bilinear activation weights for Top-32 active pairs
        # Gather uncorrupted raw similarity scores for faithful associative weighting
        raw_top1_vals = torch.gather(S1_raw, dim=-1, index=top1_idx)
        raw_top2_vals = torch.gather(S2_raw, dim=-1, index=top2_idx)
        pair_scores = raw_top1_vals.unsqueeze(-1) + raw_top2_vals.unsqueeze(-2)  # [B, L, 16, 8, 4]
        pair_scores_flat = pair_scores.view(B, L, self.num_groups, self.topk_1 * self.topk_2)
        attn_weights = F.softmax(self.beta * pair_scores_flat, dim=-1)  # [B, L, 16, 32]
        
        # 5. Bilinear contraction with ternary value tensor V
        group_outputs = []
        for g in range(self.num_groups):
            idx1 = top1_idx[..., g, :]        # [B, L, 8]
            idx2 = top2_idx[..., g, :]        # [B, L, 4]
            weights_g = attn_weights[..., g, :]  # [B, L, 32]
            
            # Sub-tensor gathering: V[g, idx1, idx2, :]
            Vg = self.V[g]
            V_sub = Vg[idx1.unsqueeze(-1), idx2.unsqueeze(-2)]  # [B, L, 8, 4, 128]
            V_sub_flat = V_sub.view(B, L, self.topk_1 * self.topk_2, self.val_dim)
            
            # Weighted sum over 32 active attractor slots (32-dim micro-fact representation)
            m_dyn = torch.einsum('bls,blsd->bld', weights_g, V_sub_flat)  # [B, L, 32]
            
            # Linear expansion: 32 -> 128 for 100% systolic TPU MXU saturation
            m_expanded = torch.einsum('bld,de->ble', m_dyn, self.W_expand[g])  # [B, L, 128]
            
            # Add shared anchor attractor slot
            m_g = m_expanded + self.V_shared[g].unsqueeze(0).unsqueeze(0)
            group_outputs.append(m_g)
            
        # Concatenate across all 16 groups: [B, L, 16 * 128 = 2048]
        m_concat = torch.cat(group_outputs, dim=-1)
        
        # 6. Knowledge projection back to 1536
        y_know = self.out_proj(m_concat)
        
        # 7. Commitment loss (binds subqueries to centroids without entropy conflict)
        weights_c1 = F.softmax(top1_vals, dim=-1).unsqueeze(-1)
        c1_active = C1_norm[torch.arange(self.num_groups, device=x.device).view(1, 1, self.num_groups, 1), top1_idx]
        c1_bar = (weights_c1 * c1_active).sum(dim=-2)
        
        weights_c2 = F.softmax(top2_vals, dim=-1).unsqueeze(-1)
        c2_active = C2_norm[torch.arange(self.num_groups, device=x.device).view(1, 1, self.num_groups, 1), top2_idx]
        c2_bar = (weights_c2 * c2_active).sum(dim=-2)
        
        commit_loss = (
            F.mse_loss(q1, c1_bar.detach(), reduction='none').sum(dim=-1).mean() +
            0.25 * F.mse_loss(q1.detach(), c1_bar, reduction='none').sum(dim=-1).mean() +
            F.mse_loss(q2, c2_bar.detach(), reduction='none').sum(dim=-1).mean() +
            0.25 * F.mse_loss(q2.detach(), c2_bar, reduction='none').sum(dim=-1).mean()
        )
                      
        return y_know, commit_loss
