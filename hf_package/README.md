---
language:
- uk
- en
license: apache-2.0
tags:
- hetman
- bitnet
- ternary
- recurrent
- hopfield
- deltanet
- long-context
- 256k
- reasoning
- custom_code
- pytorch
- jax
pipeline_tag: text-generation
widget:
- text: "<|im_start|>system\nYou are Hetman-2.0B, a helpful, intelligent, and objective AI assistant.<|im_end|>\n<|im_start|>user\nExplain how Hopfield associative memory works in neural architectures.<|im_end|>\n<|im_start|>assistant\n<think>\n"
---

# Hetman-2.0B
### 2.14B Total Parameters (~1.603B Active) | BitNet b1.58 Ternary | 16.78M Hopfield Slots | 256k Context | TPU v5e-128

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub](https://img.shields.io/badge/GitHub-Hetman--2b-black?logo=github)](https://github.com/YaroslavSamchuk/Hetman-2b)
[![Inference Engine](https://img.shields.io/badge/Runtime-Rada.cpp-orange)](https://github.com/YaroslavSamchuk/Rada.cpp)
[![Target Hardware](https://img.shields.io/badge/Target_Edge-NVIDIA_RTX_2060+_>=6GB-green.svg)](#)
[![Context Window](https://img.shields.io/badge/Context-262k_Tokens-purple.svg)](#)

> 📢 **Model Status: Architecture & Production Tokenizer Release (Pre-Training Phase)**  
> Welcome to the official repository for **Hetman-2.0B**!  
> **Please Note:** The final pre-trained model weights are **not yet uploaded**—the 855-billion-token pre-training run on Cloud TPU v5e-128 pods is scheduled and actively being prepared.  
> 
> **What is live and fully usable in this repository today:**
> 1. **Complete Model Architecture:** Verified PyTorch code with custom modeling classes (`trust_remote_code=True`).
> 2. **Production 65,536-Token Tokenizer:** Fully functional Byte-Level BPE tokenizer with unified Cyrillic apostrophe normalization (achieving $\le 1.18$ tokens/word on Ukrainian). Ready to tokenize/encode text today!
> 3. **C++ Edge Inference Toolchain:** Native compatibility with [`Rada.cpp`](https://github.com/YaroslavSamchuk/Rada.cpp) for sub-2GB VRAM local execution.
> 
> As training milestones complete, official checkpoint weights (`model.safetensors` and `.rada` single-file containers) will be pushed directly to this model card. Feel free to clone, inspect the architecture, and test the tokenizer below!

---

## 🏛️ Architectural Highlights

**Hetman-2.0B** is an open-source, edge-native foundation model designed from first principles for high-throughput pre-training on **Cloud TPU v5e-128 Pod Slices** and ultra-efficient local inference on consumer GPUs with **$\ge 6$ GB VRAM** (RTX 2060+, RTX 3060, RTX 4060, and Apple Silicon).

* **Nominal Capacity:** 2,137,522,176 parameters (~2.138B storage footprint).
* **Active Compute per Token:** ~1,602,748,416 parameters (~1.603B active compute) via Top-32 sparse associative memory retrieval.
* **Weight Precision:** BitNet b1.58 ternary $\{-1, 0, +1\}$ augmented with orthogonal Fast Walsh-Hadamard Transforms ($B_{\text{had}} = 512$).
* **SLH-Core v2 (Speculative Latent-Hopfield Core):** **16,777,216 discrete attractor slots** across 16 groups ($16 \times 1024 \times 1024 \times 32$ trits, 107.37 MB packed). Projects 32-dim micro-fact attractor combinations into 128-dim via learnable isometry matrix $W_{\text{expand}} \in \mathbb{R}^{16 \times 32 \times 128}$, achieving **100% systolic efficiency** on TPU v5e $128 \times 128$ MXUs and cutting GPU DRAM gather traffic by **75%** (4 KB/token).
* **Aux-Loss-Free PID Dynamic Router:** Eliminates slot starvation and routing collapse without gradient conflicts.
* **Context Window:** 262,144 tokens (256k) with fixed $O(1)$ recurrent memory state (total 3.67 MB across 28 Gated DeltaNet layers) and 6 interleaved Global Softmax FlashAttention layers with GQA (12:2) and bounded QK-Norm ($|S_{ij}| \le 11.31$).
* **Multi-Token Prediction (MTP):** Auxiliary 2-token speculative drafting head driving inference speeds $> 350\text{ tok/s}$ in `Rada.cpp`.
* **Cyrillic & Code Vocabulary:** 65,536 tokens (Byte-Level BPE with unified apostrophe normalization, achieving $\le 1.18$ tokens/word on Ukrainian).

---

## ⚡ Quickstart: Testing the Tokenizer

The production Byte-Level BPE tokenizer is fully trained and ready for immediate use:

```python
from transformers import AutoTokenizer

# Load the official Hetman-2.0B tokenizer
tokenizer = AutoTokenizer.from_pretrained("YSamchuk/Hetman-2b")

# Test tokenization and encoding
text = "Hetman-2.0B is an open-source foundation model engineered with 256k context and Hopfield associative memory."
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print(f"Token count: {len(tokens)}")
print(f"Tokens: {tokens}")
print(f"Fertility: {len(tokens) / len(text.split()):.2f} tokens/word")
```

---

## 🔬 Instantiating the Architecture (`trust_remote_code=True`)

You can inspect and instantiate the full 2.14B model architecture locally:

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM

# Load configuration
config = AutoConfig.from_pretrained("YSamchuk/Hetman-2b", trust_remote_code=True)

# Instantiate model architecture (random initialization for architecture validation)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

print(f"Model Architecture: {model.__class__.__name__}")
print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Context Window: {config.max_position_embeddings:,} tokens")
print(f"Hopfield Attractor Slots: {config.hopfield_num_groups * config.hopfield_codebook_size * config.hopfield_codebook_size:,}")
```

---

## 📊 Architectural Specification

```text
┌────────────────────────────────────────────────────────────────────────┐
│               HETMAN-2.0B ARCHITECTURAL PASSPORT                       │
├────────────────────────────────┬───────────────┬───────────────────────┤
│ Parameter                      │ Value         │ Hardware Rationale    │
├────────────────────────────────┼───────────────┼───────────────────────┤
│ hidden_size                    │ 1536          │ 12 x 128 (TPU MXU)    │
│ intermediate_size (FFN)        │ 6144 (4.0x)   │ 48 x 128, 12 x 512    │
│ num_hidden_layers              │ 34 (28 + 6)   │ Hybrid ratio ~4.7:1   │
│ deltanet_head_dim              │ 64            │ 16 heads x 64 = 1024  │
│ deltanet_conv_kernel           │ 4             │ Causal 1D morphology  │
│ global_heads / global_kv_heads │ 12 / 2 (GQA)  │ 6:1 KV compression    │
│ global_head_dim                │ 128           │ TPU systolic tile     │
│ hadamard_block_size            │ 512           │ Divides 1536/6144/2048│
│ hopfield_groups                │ 16            │ 16 semantic heads     │
│ hopfield_codebook_size (C)     │ 1024          │ 2^10 dual centroids   │
│ hopfield_val_dim (d_v)         │ 32            │ uint2 64-bit transfer │
│ hopfield_expanded_dim          │ 128           │ 100% TPU MXU aligned  │
│ hopfield_attractors            │ 16,777,216    │ 16.78M micro-facts    │
│ hopfield_topk_1 / topk_2       │ 8 / 4 (32)    │ beta = 1 / sqrt(32)   │
│ vocab_size                     │ 65536         │ 2^16 tokens           │
│ max_position_embeddings        │ 262144 (256k) │ 2^18 tokens           │
└────────────────────────────────┴───────────────┴───────────────────────┘
```

---

## 🚀 Native C++20 / CUDA Inference Runtime: `Rada.cpp`

For consumer edge inference (NVIDIA RTX 2060 6GB+, RTX 3060, RTX 4060, and Apple Silicon), Hetman-2.0B is paired with its native runtime **[Rada.cpp](https://github.com/YaroslavSamchuk/Rada.cpp)**:

* **Add-Only DP4A Kernels:** Eliminates FP16 multiply-accumulate overhead in favor of 2-bit integer adds.
* **Vectorized 64-bit uint2 Memory Transfers:** Gather operations read 8 bytes per slot instead of 32 bytes, cutting DRAM bandwidth by 75% (**4 KB per token**).
* **Compact VRAM Footprint:** Full 256,000-token context requires only **~2.00 GB VRAM**, leaving **4.00 GB free** on a standard 6 GB GPU.
* **Speculative Decoding:** Auxiliary MTP head drafts 2 tokens ahead for $> 350\text{ tok/s}$ generation throughput.

---

## 📜 License & Citation

This project is released under the permissive **Apache License 2.0**.

```bibtex
@software{samchuk2026hetman,
  author = {Samchuk, Yaroslav},
  title = {Hetman-2.0B: High-Efficiency Edge-Native Sovereign Foundation Model},
  year = {2026},
  publisher = {Hugging Face / GitHub},
  url = {https://github.com/YaroslavSamchuk/Hetman-2b}
}
```
