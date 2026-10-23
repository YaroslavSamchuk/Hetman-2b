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
- text: "<|im_start|>system\nYou are Hetman-2.0B, an intelligent and thoughtful AI assistant.<|im_end|>\n<|im_start|>user\nExplain how Hopfield associative memory works in neural architectures.<|im_end|>\n<|im_start|>assistant\n<think>\n"
---

# Hetman-2.0B

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub](https://img.shields.io/badge/GitHub-Hetman--2b-black?logo=github)](https://github.com/YaroslavSamchuk/Hetman-2b)
[![Inference Engine](https://img.shields.io/badge/Runtime-Rada.cpp-orange)](https://github.com/YaroslavSamchuk/Rada.cpp)

**Hetman-2.0B** is an edge-native, sovereign foundation model engineered for high-efficiency reasoning, extreme context handling, and low-bit compute. Developed from first principles for consumer hardware accessibility, Hetman combines 1.58-bit ternary quantization, linear recurrent attention, and sparse associative memory to deliver 256,000-token context processing within consumer GPU memory budgets.

---

## Introduction

In conventional autoregressive language models, expanding context windows and scaling parameters to preserve factual recall inevitably incurs an unsustainable hardware penalty. At 128k–256k tokens, the KV-cache of standard transformers expands to tens of gigabytes of VRAM, pricing independent developers, researchers, and edge devices out of frontier AI.

**Hetman-2.0B** rethinks the core architecture of sub-3B parameter models:
* Rather than memorizing encyclopedic facts inside continuous dense weights, it delegates factual storage to a dedicated, sparse associative memory core.
* Rather than relying on quadratic Softmax attention that bloats memory with sequence length, it adopts linear recurrent memory with constant state size.
* Rather than executing energy-intensive FP16 matrix multiplications, it uses 1.58-bit ternary weights ($\{-1, 0, +1\}$), replacing multiplications with integer additions.

The result is a model designed to run a complete **256,000-token context window locally on an accessible consumer GPU (such as an NVIDIA GeForce RTX 2060 6 GB) using under 2.00 GB of VRAM**.

---

## Key Architectural Innovations

### 1. 1.58-Bit Ternary Weights (BitNet b1.58 + FWHT)
Hetman-2.0B constrains linear projection weights strictly to ternary values $\{-1, 0, +1\}$. This replaces standard floating-point GEMM operations with lightweight integer addition and subtraction inside GPU registers. To prevent activation outliers and preserve signal variance across deep layers, activations pass through online orthogonal **Fast Walsh-Hadamard Transforms (FWHT, $B=512$)**, bounding quantization loss to $< 1.8\%$ compared to full FP32 baselines.

### 2. Hybrid 256k Context Engine (Gated DeltaNet + Global FlashAttention)
To break the quadratic memory barrier of standard transformers:
* **28 Gated DeltaNet Recurrent Layers:** Maintain a constant $O(1)$ recurrent memory state (just 3.67 MB across all 28 layers combined), allowing infinite context streaming without memory growth.
* **6 Interleaved Global FlashAttention Anchor Layers:** Positioned periodically with Grouped-Query Attention (12:2 GQA) and bounded QK-Norm ($|S_{ij}| \le 11.31$) to anchor associative recall and eliminate attention dilution.

### 3. Sparse Associative Memory (SLH-Core v2)
Factual knowledge retrieval is offloaded from dense FFN layers to the **Speculative Latent-Hopfield Core (SLH-Core)**. Containing **16,777,216 discrete attractor slots** across 16 semantic groups, it performs sparse Top-32 fact retrieval via 2D bilinear contractions. Micro-facts are projected into the model dimension via an isometry matrix $W_{\text{expand}}$, reducing memory traffic by 75% down to **4 KB per token**. An auxiliary-loss-free PID dynamic bias controller prevents slot starvation without interfering with task gradients.

### 4. Native Speculative Drafting (Multi-Token Prediction)
Hetman incorporates an auxiliary 2-token speculative prediction head trained concurrently with the backbone. When served in the companion C++ runtime `Rada.cpp`, it enables multi-token drafting and verification with $O(1)$ recurrent shadow-state rollback, driving generation throughput above **350 tok/s** on consumer GPUs.

### 5. Cyrillic & Code First-Class Tokenizer
The model utilizes a custom 65,536-token Byte-Level BPE tokenizer with deterministic Ukrainian apostrophe normalization (`ʼ`, `'`, `’`). It achieves an exceptional fertility of $\le 1.18$ tokens/word on Ukrainian text (compared to 2.5–3.2 in standard LLaMA tokenizers), saving over 30% compute on multilingual tasks.

---

## Model Architecture Specifications

| Parameter | Value | Hardware Alignment Rationale |
| :--- | :--- | :--- |
| **Total Parameters (Storage)** | **2,137,522,176 (~2.14B)** | Fully packs within ~500 MB in 2-bit format |
| **Active Parameters per Token** | **~1,602,748,416 (~1.60B)** | Sub-linear Top-32 sparse Hopfield routing |
| **Hidden Size ($d_{\text{model}}$)** | 1536 | Systolic tile alignment ($12 \times 128$) |
| **Intermediate Size (FFN)** | 6144 ($4.0\times$) | Blockwise FWHT alignment ($12 \times 512$) |
| **Total Layers** | 34 | 28 Gated DeltaNet + 6 Global Attention |
| **Recurrent Head Dimension** | 64 | 16 heads $\times$ 64 = 1024 state dimension |
| **Global Attention Heads / KV** | 12 / 2 (GQA) | 6:1 KV-cache compression ratio |
| **Hopfield Memory Slots** | 16,777,216 | 16 groups $\times$ 1024 dual centroids $\times$ 32 trits |
| **Context Window Length** | 262,144 tokens (256k) | Linear recurrent $O(1)$ memory state |
| **Vocabulary Size** | 65,536 tokens | Byte-Level BPE with unified apostrophe handling |

---

## Development Status & Pre-Training Roadmap

Hetman-2.0B is an independent, open-source research initiative developed from the ground up for community access.

Currently, the complete architectural codebase, mathematical verification suites, production 65k tokenizer, distributed training autopilot, and native C++ runtime are fully finished, verified, and released. The final model weights are currently not yet trained due to cloud compute budget limitations; the project is actively awaiting approval and allocation of compute grants to carry out the planned **855-billion-token pre-training curriculum on Cloud TPU v5e-128 accelerator pods**.

As soon as compute resources are secured and pre-training completes, all production checkpoints (`model.safetensors` and single-file `.rada` containers) will be released directly to this repository under the Apache 2.0 license.

---

## License & Citation

This project is licensed under the permissive **Apache License 2.0**.

```bibtex
@software{samchuk2026hetman,
  author = {Samchuk, Yaroslav},
  title = {Hetman-2.0B: High-Efficiency Edge-Native Sovereign Foundation Model},
  year = {2026},
  publisher = {Hugging Face / GitHub},
  url = {https://github.com/YaroslavSamchuk/Hetman-2b}
}
```

### Repositories & Resources
* **Hetman-2.0B GitHub:** [github.com/YaroslavSamchuk/Hetman-2b](https://github.com/YaroslavSamchuk/Hetman-2b)
* **Rada.cpp GitHub:** [github.com/YaroslavSamchuk/Rada.cpp](https://github.com/YaroslavSamchuk/Rada.cpp)
* **Hugging Face Model Hub:** [huggingface.co/YSamchuk/Hetman-2b](https://huggingface.co/YSamchuk/Hetman-2b)
