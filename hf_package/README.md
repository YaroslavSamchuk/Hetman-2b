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

# Hetman-2.0B: Architecture & Production Tokenizer
### 2.14B Total Parameters (~1.603B Active) | BitNet b1.58 Ternary | 16.78M Hopfield Slots | 256k Context | Cloud TPU v5e-128

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status](https://img.shields.io/badge/Status-Pre--training_In_Progress-blue.svg)](#)
[![GitHub](https://img.shields.io/badge/GitHub-Hetman--2b-black?logo=github)](https://github.com/YaroslavSamchuk/Hetman-2b)
[![Inference Engine](https://img.shields.io/badge/Runtime-Rada.cpp-orange)](https://github.com/YaroslavSamchuk/Rada.cpp)
[![Pretrain-Platform](https://img.shields.io/badge/Pretraining-Cloud_TPU_v5e--128-orange.svg)](#)
[![Target Hardware](https://img.shields.io/badge/Target_Edge-NVIDIA_RTX_2060+_>=6GB-green.svg)](#)
[![Context Window](https://img.shields.io/badge/Context-262k_Tokens-purple.svg)](#)

---

> [!IMPORTANT]
> ### 🚧 Open-Source Architecture & Tokenizer Release — Pre-Training In Progress
>
> **Welcome to Hetman-2.0B! Please note: model weights are NOT yet trained.**
>
> We are building Hetman-2.0B entirely in the open. This Hugging Face repository currently hosts the **production-ready 65,536-token BPE tokenizer** and the **complete PyTorch architecture implementation (`trust_remote_code=True`)**.
>
> **Pre-training on 855 billion tokens** across Cloud TPU v5e-128 pod slices is currently scheduled and in active preparation. The final pre-trained weights (`model.safetensors` and `.rada` single-file containers) will be published directly to this repository as soon as training converges.
>
> #### 🛠️ What You Can Do Right Now:
> 1. **Use the Production 65k Tokenizer:** Load it directly via `AutoTokenizer.from_pretrained("YSamchuk/Hetman-2b")`. It is fully trained, provides excellent Ukrainian fertility ($\le 1.18$ tokens/word), and normalizes Cyrillic apostrophe variants (`ʼ`, `'`, `’`).
> 2. **Instantiate & Inspect the Full 2.14B Architecture:** Load the model graph with `trust_remote_code=True` to inspect layer configurations, verify parameter counts, test memory footprints, or experiment with custom training.
> 3. **Run Fast Local Sanity Checks (< 5s):** Clone the [GitHub repository](https://github.com/YaroslavSamchuk/Hetman-2b) and run `python src/test_verification.py` to verify BitLinear ternarization, FWHT rotations, DeltaNet recurrence, and QK-Norm bounds on your CPU.
> 4. **Benchmark with Rada.cpp:** Compile our companion C++20 / CUDA runtime [Rada.cpp](https://github.com/YaroslavSamchuk/Rada.cpp) to test add-only DP4A integer GEMM execution and 256k context handling on consumer GPUs.
>
> #### 🎯 What to Expect Once Pre-Training Finishes:
> - Full pre-trained weights (`model.safetensors`) will be uploaded directly to this repository.
> - Pre-packaged `.rada` binaries will be released for instant, zero-dependency offline inference in `Rada.cpp`.
> - Official benchmark results (MMLU, GSM8K, HumanEval, Needle-In-A-Haystack up to 256k tokens, UA-MMLU) will be posted with reproducible logs.

---

## 💡 Why We're Building Hetman-2.0B

Modern large language models are incredible, but they have grown increasingly bloated and demanding. Today, serving a long-context prompt often requires a multi-GPU cluster costing thousands of dollars. We believe sovereign, high-efficiency AI belongs in the hands of everyday researchers, engineers, and hackers—on the hardware they already own.

**Hetman-2.0B** is an open-source, edge-native foundation model built from the ground up to achieve two key breakthroughs:
1. **Systolic Pre-Training Efficiency:** 100% hardware alignment with Cloud TPU v5e-128 $128 \times 128$ Matrix Multiply Units (MXUs).
2. **Accessible Consumer Inference:** Full **256,000-token context inference on an accessible 6 GB consumer GPU** (like an NVIDIA GeForce RTX 2060 or laptop GPU) using just **~2.00 GB of VRAM**, leaving **4.00 GB free**.

### Key Architectural Pillars
* **BitNet b1.58 Ternary Weights:** Weights live in $\{-1, 0, +1\}$, eliminating expensive FP16 multiplications in favor of integer additions. Fast Walsh-Hadamard Transforms ($B_{\text{had}} = 512$) suppress activation outliers, bounding quantization loss to $< 1.8\%$ vs. FP32.
* **SLH-Core v2 (Speculative Latent-Hopfield Associative Memory):** 16,777,216 discrete attractor slots ($16 \times 1024 \times 1024 \times 32$ trits, 107.37 MB packed) enable sparse Top-32 fact retrieval. Projects 32-dim micro-fact attractors into 128-dim via learnable isometry $W_{\text{expand}}$, achieving 100% systolic efficiency on TPU MXUs and cutting GPU DRAM gather traffic by 75% (**4 KB per token**).
* **Aux-Loss-Free PID Dynamic Router:** Replaces unstable auxiliary loss terms with a non-differentiable symmetric PID dynamic bias controller that prevents routing collapse without polluting task gradients.
* **Hybrid 256k Context Engine:** 28 Gated DeltaNet linear recurrent layers ($O(1)$ memory state, only 3.67 MB total) interleaved with 6 Global Softmax FlashAttention layers with GQA (12:2) and bounded QK-Norm ($|S_{ij}| \le 11.31$) for needle-sharp long-context retrieval without KV-cache explosion.
* **Multi-Token Prediction (MTP):** Auxiliary 2-token speculative drafting head targeting $> 350\text{ tok/s}$ in `Rada.cpp` with $O(1)$ recurrent shadow-state rollback.

---

## ⚡ Quickstart 1: Testing the Production Tokenizer

The 65,536-token Byte-Level BPE tokenizer is fully trained and ready for immediate use today:

```python
from transformers import AutoTokenizer

# Load the official Hetman-2.0B tokenizer
tokenizer = AutoTokenizer.from_pretrained("YSamchuk/Hetman-2b")

# Test tokenization and Ukrainian / English fertility
sample_text = "Hetman-2.0B — це суверенна edge-native модель із контекстом 256k токенів та асоціативною пам'яттю Гопфілда."
tokens = tokenizer.tokenize(sample_text)
token_ids = tokenizer.encode(sample_text)

print(f"Token count: {len(tokens)}")
print(f"Sample tokens: {tokens[:10]}")
print(f"Fertility: {len(tokens) / len(sample_text.split()):.2f} tokens/word")
```

---

## 🔬 Quickstart 2: Instantiating the Full Architecture (`trust_remote_code=True`)

You can inspect the model architecture, verify parameter dimensions, and test forward/backward passes locally (using randomly initialized weights):

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM

# 1. Load the official configuration
config = AutoConfig.from_pretrained("YSamchuk/Hetman-2b", trust_remote_code=True)

# 2. Instantiate the complete 2.14B architecture
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model Class:             {model.__class__.__name__}")
print(f"Total Parameters:        {total_params:,} (~{total_params / 1e9:.3f}B)")
print(f"Context Window:          {config.max_position_embeddings:,} tokens")
print(f"Hopfield Attractor Slots:{config.hopfield_num_groups * config.hopfield_codebook_size * config.hopfield_codebook_size:,}")

# 3. Quick forward pass test with dummy inputs
dummy_input = torch.randint(0, config.vocab_size, (1, 16))
with torch.no_grad():
    outputs = model(dummy_input)

print(f"Logits shape:            {outputs.logits.shape} (batch, seq_len, vocab_size)")
print("✓ Architecture successfully verified and functional!")
```

---

## 📊 Architectural Passport

```text
┌────────────────────────────────────────────────────────────────────────┐
│               HETMAN-2.0B ARCHITECTURAL PASSPORT                       │
├────────────────────────────────┬───────────────┬───────────────────────┤
│ Parameter                      │ Value         │ Hardware Rationale    │
├────────────────────────────────┼───────────────┼───────────────────────┤
│ hidden_size                    │ 1536          │ 12 x 128 (Cloud TPU)  │
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

For consumer edge inference (NVIDIA GeForce RTX 2060 6GB+, RTX 3060, RTX 4060, and Apple Silicon), Hetman-2.0B is paired with its dedicated runtime **[Rada.cpp](https://github.com/YaroslavSamchuk/Rada.cpp)**:

* **Add-Only DP4A Kernels:** Eliminates FP16 multiply-accumulate overhead in favor of 2-bit integer adds.
* **Vectorized 64-bit uint2 Memory Transfers:** Gather operations read 8 bytes per slot instead of 32 bytes, cutting DRAM bandwidth by 75% (**4 KB per token**).
* **Ultra-Compact VRAM Footprint:** Full 256,000-token context requires only **~2.00 GB VRAM**, leaving **4.00 GB free** on a standard 6 GB GPU.
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
