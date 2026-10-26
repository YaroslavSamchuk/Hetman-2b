"""
Hetman-2.0B: Model Exporter to .rada Binary Container Format
Packs 1.58-bit ternary weights into 2 bits (16 weights / uint32_t) with per-tensor scales γ.
Generates 64-byte aligned Rada containers for instant mmap loading in rada.cpp.
"""

import os
import struct
import zlib
import numpy as np
import torch
from typing import Tuple, Optional
try:
    from .configuration_hetman import HetmanConfig
    from .modeling_hetman import HetmanForCausalLM
except ImportError:
    from configuration_hetman import HetmanConfig
    from modeling_hetman import HetmanForCausalLM

RADA_MAGIC = 0x41444152   # 'RADA'
RADA_VERSION = 1
RADA_ALIGNMENT = 64

# Dtypes matching rada_format.h
DTYPE_BF16 = 0
DTYPE_FP16 = 1
DTYPE_FP32 = 2
DTYPE_INT8 = 3
DTYPE_TERNARY_2BIT = 4

def pack_ternary_to_2bit(weights: torch.Tensor) -> Tuple[np.ndarray, float]:
    """
    Packs a float tensor with ternary values {-1, 0, +1} into 2-bit uint32 array.
    Guarantees row-by-row 16-element alignment for GEMV.
    Encoding:
      00 (0): 0
      01 (1): +1
      10 (2): -1
      11 (3): unused/pad
    16 weights packed per uint32.
    """
    gamma = float(torch.mean(torch.abs(weights)).item())
    if gamma < 1e-6:
        gamma = 1.0

    if weights.ndim == 2:
        N, K = weights.shape
        w_ternary = torch.clamp(torch.round(weights.detach() / gamma), -1.0, 1.0).to(torch.int8).cpu().numpy()
        mapped = np.zeros_like(w_ternary, dtype=np.uint8)
        mapped[w_ternary == 1] = 1
        mapped[w_ternary == -1] = 2
        pad_k = (16 - (K % 16)) % 16
        if pad_k > 0:
            mapped = np.pad(mapped, ((0, 0), (0, pad_k)), constant_values=0)
        reshaped = mapped.reshape(-1, 16)
        shifts = np.arange(0, 32, 2, dtype=np.uint32)
        packed = np.sum(reshaped.astype(np.uint32) << shifts, axis=1, dtype=np.uint32)
        return packed, gamma
    else:
        w_flat = weights.detach().cpu().float().numpy().flatten()
        w_ternary = np.clip(np.round(w_flat / gamma), -1.0, 1.0).astype(np.int8)
        mapped = np.zeros_like(w_ternary, dtype=np.uint8)
        mapped[w_ternary == 1] = 1
        mapped[w_ternary == -1] = 2
        pad_len = (16 - (len(mapped) % 16)) % 16
        if pad_len > 0:
            mapped = np.pad(mapped, (0, pad_len), constant_values=0)
        reshaped = mapped.reshape(-1, 16)
        shifts = np.arange(0, 32, 2, dtype=np.uint32)
        packed = np.sum(reshaped.astype(np.uint32) << shifts, axis=1, dtype=np.uint32)
        return packed, gamma

def export_model_to_rada(model: HetmanForCausalLM, output_path: str, tokenizer_path: Optional[str] = None):
    """
    Exports a HetmanForCausalLM PyTorch model into an optimized single-file .rada binary file.
    """
    cfg = model.config
    print(f"[*] Exporting Hetman-2.0B to binary container: {output_path}")
    
    tensors_to_write = []
    
    # 1. Iterate through all named parameters
    for name, param in model.named_parameters():
        param_data = param.detach().cpu()
        shape = list(param_data.shape)
        
        # Decide quantization format:
        # Note: continuous gates (alpha_proj, beta_proj, gate_proj, fusion_gate), W_expand, and V_shared must remain continuous FP16/BF16
        is_continuous_gate = any(k in name for k in ["alpha_proj", "beta_proj", "gate_proj", "fusion_gate", "W_expand"])
        is_ternary = (("proj.weight" in name and not is_continuous_gate) or 
                      ("C1" not in name and "C2" not in name and "W_expand" not in name and ".V" in name and "V_shared" not in name and "norm" not in name and "embed" not in name))
        
        if is_ternary and param_data.numel() > 1024:
            packed_data, gamma = pack_ternary_to_2bit(param_data)
            raw_bytes = packed_data.astype('<u4').tobytes()
            dtype = DTYPE_TERNARY_2BIT
            scale = gamma
        else:
            # High-precision parameters in BF16 or FP16
            if param_data.dtype == torch.bfloat16:
                raw_bytes = param_data.view(torch.int16).numpy().tobytes()
                dtype = DTYPE_BF16
            else:
                raw_bytes = param_data.half().numpy().tobytes()
                dtype = DTYPE_FP16
            scale = 1.0
            
        tensors_to_write.append({
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "scale": scale,
            "bytes": raw_bytes
        })

    # 2. Iterate through named buffers (Hopfield PID biases b1, b2, etc.)
    for name, buf in model.named_buffers():
        buf_data = buf.detach().cpu()
        shape = list(buf_data.shape)
        raw_bytes = buf_data.float().numpy().tobytes()
        tensors_to_write.append({
            "name": name,
            "shape": shape,
            "dtype": DTYPE_FP32,
            "scale": 1.0,
            "bytes": raw_bytes
        })
        
    num_tensors = len(tensors_to_write)
    header_size = 128
    desc_table_size = num_tensors * 128
    data_start_offset = (header_size + desc_table_size + (RADA_ALIGNMENT - 1)) & ~(RADA_ALIGNMENT - 1)
    
    # Check for tokenizer to embed (single-file container architecture)
    vocab_bytes = b""
    vocab_offset = 0
    vocab_size_bytes = 0
    merges_offset = 0
    merges_size_bytes = 0
    
    tok_default_path = os.path.join(os.path.dirname(__file__), "..", "tokenizer", "tokenizer.json")
    if tokenizer_path and os.path.exists(tokenizer_path):
        with open(tokenizer_path, "rb") as vf:
            vocab_bytes = vf.read()
        vocab_size_bytes = len(vocab_bytes)
    elif os.path.exists(tok_default_path):
        with open(tok_default_path, "rb") as vf:
            vocab_bytes = vf.read()
        vocab_size_bytes = len(vocab_bytes)

    # Calculate descriptors and payloads
    descriptors = []
    current_payload_offset = data_start_offset
    payload_blobs = []
    
    for t in tensors_to_write:
        name_bytes = t["name"].encode("utf-8")[:63].ljust(64, b"\x00")
        ndims = len(t["shape"])
        padded_shape = (t["shape"] + [0] * (4 - ndims))[:4]
        payload = t["bytes"]
        payload_size = len(payload)
        checksum = zlib.crc32(payload) & 0xFFFFFFFF
        
        # Align payload offset to 64 bytes
        current_payload_offset = (current_payload_offset + (RADA_ALIGNMENT - 1)) & ~(RADA_ALIGNMENT - 1)
        
        desc = struct.pack(
            "<64sIIIIIIf4sQQI12s",
            name_bytes,
            ndims,
            padded_shape[0], padded_shape[1], padded_shape[2], padded_shape[3],
            t["dtype"],
            t["scale"],
            b"\x00" * 4,
            current_payload_offset,
            payload_size,
            checksum,
            b"\x00" * 12
        )
        assert len(desc) == 128, f"Descriptor size must be 128 bytes, got {len(desc)}"
        descriptors.append(desc)
        
        payload_blobs.append((current_payload_offset, payload))
        current_payload_offset += payload_size

    # Position embedded vocab table after all tensor payloads
    if vocab_bytes:
        vocab_offset = (current_payload_offset + (RADA_ALIGNMENT - 1)) & ~(RADA_ALIGNMENT - 1)
        current_payload_offset = vocab_offset + vocab_size_bytes

    # Build Header struct (exactly 128 bytes)
    header = struct.pack(
        "<IIIIIIIIIIIIIIIIIIQQQQQQ8s",
        RADA_MAGIC,
        RADA_VERSION,
        cfg.vocab_size,
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_hidden_layers,
        cfg.num_deltanet_layers,
        cfg.num_global_layers,
        cfg.num_attention_heads,
        cfg.global_kv_heads,
        cfg.deltanet_head_dim,
        cfg.global_head_dim,
        cfg.global_attention_heads,
        cfg.hopfield_num_groups,
        cfg.hopfield_codebook_size,
        cfg.hopfield_val_dim,
        cfg.max_position_embeddings,
        num_tensors,
        header_size,           # tensor_table_offset
        data_start_offset,     # tensor_data_offset
        vocab_offset,          # vocab_offset
        vocab_size_bytes,      # vocab_size_bytes
        merges_offset,         # merges_offset
        merges_size_bytes,     # merges_size_bytes
        b"\x00" * 8            # reserved (8 bytes)
    )
    assert len(header) == 128, f"Header size must be 128 bytes, got {len(header)}"
        
    # Write container file
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header)
        for desc in descriptors:
            f.write(desc)
            
        for offset, data in payload_blobs:
            curr = f.tell()
            if curr < offset:
                f.write(b"\x00" * (offset - curr))
            f.write(data)
            
        if vocab_bytes:
            curr = f.tell()
            if curr < vocab_offset:
                f.write(b"\x00" * (vocab_offset - curr))
            f.write(vocab_bytes)
            
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[✓] Successfully exported {num_tensors} tensors to {output_path} ({file_size_mb:.2f} MB)")
    if vocab_bytes:
        print(f"    Embedded Tokenizer: {len(vocab_bytes):,} bytes at offset 0x{vocab_offset:X}")
    return output_path

if __name__ == "__main__":
    print("=== Testing Hetman Export Bridge ===")
    mini_cfg = HetmanConfig(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=34,
        num_deltanet_layers=28,
        num_global_layers=6,
        deltanet_head_dim=32,
        global_head_dim=32,
        global_attention_heads=4,
        global_kv_heads=2,
        hopfield_num_groups=2,
        hopfield_codebook_size=32,
        hopfield_val_dim=32,
        hopfield_subquery_dim=128
    )
    model = HetmanForCausalLM(mini_cfg)
    test_out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "rada.cpp", "models", "hetman_mini.rada"))
    os.makedirs(os.path.dirname(test_out), exist_ok=True)
    export_model_to_rada(model, test_out)
    print("=== Export Bridge Test Passed! ===")
