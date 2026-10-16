"""
Hetman-2.0B: Rigorous Mathematical & Empirical Audit (v3 - Final Verified Edition)
- Parameter breakdown: Storage (2.165B with MTP, 2.136B base) vs Active compute (1.599B base, 1.628B with MTP)
- DeltaNet: Bottleneck architecture (1536 -> 1024, 16 heads x 64 dim) & Post-update temporal convention
- Memory: Physical TPU v5e HBM (16 GB decimal = 14.90 GiB binary), KV-cache (1.50 GiB @ B=1, BF16 for 6 layers),
  and mandatory chunking for Hopfield on 256k
- Attention FLOPs (First-principles derivation): 6 * H_Q * L * d_k per layer (fwd: 2 L^2 d_k + bwd: 4 L^2 d_k)
  => Phase 2 attention overhead is +151.1% (14.50 GFLOPs/tok attention + 9.60 GFLOPs/tok dense = 24.09 GFLOPs/tok)
- Schedule sensitivity matrix for 855B tokens across 128 TPU v5e chips
"""

import math
import torch
from configuration_hetman import HetmanConfig
from modeling_hetman import HetmanForCausalLM

def run_rigorous_audit():
    print("=" * 115)
    print("  HETMAN-2.0B VERIFIED MATHEMATICAL AUDIT: PARAMETERS, MEMORY & COMPUTE SCHEDULE")
    print("=" * 115)
    
    cfg = HetmanConfig()
    with torch.device("meta"):
        model = HetmanForCausalLM(cfg)

    # 1. Parameter audit
    total_unique = sum(p.numel() for p in model.parameters())
    mtp_params = sum(p.numel() for p in model.mtp_head.parameters())
    base_model = total_unique - mtp_params
    v_total = model.model.hopfield_core.V.numel()  # 536,870,912
    v_active = cfg.hopfield_num_groups * cfg.hopfield_topk_1 * cfg.hopfield_topk_2 * cfg.hopfield_val_dim  # 16,384
    v_inactive = v_total - v_active  # 536,854,528
    base_active_params = base_model - v_inactive  # 1,599,372,672
    train_active_params = total_unique - v_inactive  # 1,627,696,512

    print("\n[1] PARAMETER ALLOCATION & ACTIVE COMPUTE DEFINITION")
    print("-" * 80)
    print(f"Total Unique Storage Parameters (with MTP):       {total_unique:>14,}")
    print(f"  - MTP Auxiliary Head (Pre-training only):       {mtp_params:>14,}")
    print(f"Base Inference Model (without MTP):               {base_model:>14,}")
    print(f"Hopfield V tensor (total storage elements):       {v_total:>14,}")
    print(f"Hopfield V active elements per token:             {v_active:>14,}")
    print(f"Hopfield V inactive elements per token:           {v_inactive:>14,}")
    print(f"Base Active Parameters (Inference definition):    {base_active_params:>14,} (~1.599B)")
    print(f"Training Active Parameters (with MTP definition): {train_active_params:>14,} (~1.628B)")

    # 2. DeltaNet architecture
    print("\n[2] GATED DELTANET RECURRENT ARCHITECTURE")
    print("-" * 80)
    print("Topology: 28 Gated DeltaNet layers (Linear recurrent memory, O(1) state)")
    print(f"Channel bottleneck: hidden_size (1536) -> total_dim (1024) [16 heads x 64 dim] -> hidden_size (1536)")
    print("Recurrent state S footprint: [16 heads, 64, 64] = 65,536 elements per layer")
    print(f"  * Per layer (B=1, FP32): 256 KiB")
    print(f"  * All 28 layers (B=1, FP32): 7.00 MiB (Strict O(1) memory)")
    print(f"  * All 28 layers (B=1, BF16): 3.50 MiB")
    print(f"  * All 28 layers (B=2, FP32): 14.00 MiB")
    print("Temporal convention: Post-update (y_t = S_t @ q_t) for immediate step-t self-querying.")

    # 3. Memory & Hardware sharding
    print("\n[3] TPU v5e MEMORY FOOTPRINT & SHARDING AUDIT")
    print("-" * 80)
    fp32_weights = total_unique * 4 / (1024**3)
    bf16_grads = total_unique * 2 / (1024**3)
    adamw_states = total_unique * 8 / (1024**3)
    total_static_gib = fp32_weights + bf16_grads + adamw_states
    
    print(f"Static Training Memory (FP32 master + BF16 grads + FP32 AdamW m, v): {total_static_gib:.2f} GiB")
    print("Hardware Sharding: 128 TPU v5e chips organized as Mesh(data=32, model=4)")
    print("Physical chip HBM: 16 GB (decimal) = 14.90 GiB (binary) per chip")
    print(f"With Model Sharding across 4 chips: {total_static_gib / 4:.2f} GiB / chip static HBM")
    print(f"Physical headroom remaining per chip: {14.90 - (total_static_gib / 4):.2f} GiB / chip (before activations/workspace)")
    
    # KV-cache calculation for 6 Global Attention layers
    kv_elements = 6 * 262144 * 2 * 128 * 2  # 805,306,368 elements
    kv_bytes = kv_elements * 2  # BF16 = 2 bytes/element => 1,610,612,736 bytes
    kv_gib = kv_bytes / (1024**3)  # 1.500 GiB
    print(f"Global Attention KV-Cache (6 layers, GQA 12:2, head_dim=128, BF16):")
    print(f"  * L=262,144 (256k), B=1: {kv_elements:,} elements = {kv_gib:.3f} GiB ({kv_bytes / 1e9:.2f} GB)")
    print(f"  * L=262,144 (256k), B=2: {kv_elements * 2:,} elements = {kv_gib * 2:.3f} GiB ({kv_bytes * 2 / 1e9:.2f} GB)")

    # 4. Compute Budget & Exact Attention FLOPs
    print("\n[4] FIRST-PRINCIPLES FLOPs DERIVATION & REVISED SCHEDULE MATRIX")
    print("-" * 80)
    dense_flops_tok = 6 * base_active_params  # 9.5962e+09 FLOPs/tok
    
    # Causal FlashAttention Training FLOPs:
    # Forward: 2 * L^2 * d_k per head (QK^T + PV for causal triangle: 2 * (L^2 / 2) * d_k = L^2 * d_k each)
    # Backward: 4 * L^2 * d_k per head (dV, dP, dQ, dK)
    # Total fwd+bwd per head: 6 * L^2 * d_k => divided by L: 6 * L * d_k per token per head
    # For 6 global layers with H_Q = 12, d_k = 128:
    # FLOPs_attn / token = 6 layers * 6 * H_Q * L * d_k = 432 * L * d_k
    H_q = cfg.global_attention_heads
    d_k = cfg.global_head_dim
    
    attn_flops_4k = 6 * 6 * H_q * 4096 * d_k   # 0.226 GFLOPs/tok (+2.36%)
    attn_flops_256k = 6 * 6 * H_q * 262144 * d_k # 14.496 GFLOPs/tok (+151.06%)
    
    phase1_tok = dense_flops_tok + attn_flops_4k
    phase2_tok = dense_flops_tok + attn_flops_256k
    
    phase1_flops = 820e9 * phase1_tok
    phase2_flops = 30e9 * phase2_tok
    phase3_flops = 5e9 * phase1_tok
    total_flops = phase1_flops + phase2_flops + phase3_flops
    
    print(f"Dense Compute FLOPs / token:                                {dense_flops_tok/1e9:.3f} GFLOPs")
    print(f"Phase 1 (4k context) Causal Attention FLOPs / token:          {attn_flops_4k/1e9:.3f} GFLOPs (+2.36%)")
    print(f"Phase 1 Total FLOPs / token:                                 {phase1_tok/1e9:.3f} GFLOPs")
    print(f"Phase 2 (256k context) Causal Attention FLOPs / token:       {attn_flops_256k/1e9:.3f} GFLOPs (+151.06%)")
    print(f"Phase 2 Total FLOPs / token:                                {phase2_tok/1e9:.3f} GFLOPs (2.51x dense)")
    print("-" * 80)
    print(f"Phase 1 Total FLOPs (820B tokens @ 4k):                      {phase1_flops:.3e} ({phase1_flops/1e21:.2f} ZettaFLOPs)")
    print(f"Phase 2 Total FLOPs (30B tokens @ 256k):                     {phase2_flops:.3e} ({phase2_flops/1e21:.2f} ZettaFLOPs)")
    print(f"Phase 3 Total FLOPs (5B tokens SFT @ 4k):                    {phase3_flops:.3e} ({phase3_flops/1e21:.2f} ZettaFLOPs)")
    print(f"TOTAL CLUSTER FLOPs (855B tokens):                           {total_flops:.3e} ({total_flops/1e21:.2f} ZettaFLOPs)")
    print("-" * 80)
    
    peak_pflops = 128 * 197.0 / 1000.0  # 25.216 PFLOPS
    sec_day = 86400
    
    print(f"{'MFU (%)':>8} | {'Effective PFLOPS':>18} | {'Phase 1 (820B)':>15} | {'Phase 2 (30B)':>15} | {'Phase 3 (5B)':>14} | {'Total Days':>11} | {'Buffer (Days)':>14} | Status")
    print("-" * 115)
    
    for mfu in [0.15, 0.18, 0.20, 0.22, 0.25, 0.265, 0.30, 0.35]:
        eff_day = (peak_pflops * 1e15 * mfu) * sec_day
        d1 = phase1_flops / eff_day
        d2 = phase2_flops / eff_day
        d3 = phase3_flops / eff_day
        dt = d1 + d2 + d3
        buf = 28.0 - dt
        st = "FITS GRANT" if buf >= 0 else "OVER BUDGET"
        print(f"{mfu*100:>7.1f}% | {peak_pflops * mfu:>17.3f} | {d1:>13.2f}d | {d2:>13.2f}d | {d3:>12.2f}d | {dt:>9.2f}d | {buf:>12.2f}d | {st}")
    print("=" * 115)

if __name__ == "__main__":
    run_rigorous_audit()
