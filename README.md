# Hetman-2.0B: High-Efficiency Open-Source Foundation Model
### 2.14B Total Capacity (~1.603B Active) | BitNet b1.58 Ternary | 256k Context | Google TPU v5e-128 Pod

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-YSamchuk%2FHetman--2b-yellow)](https://huggingface.co/YSamchuk/Hetman-2b)
[![Target-Hardware](https://img.shields.io/badge/Target_Edge-NVIDIA_RTX_2060+_>=6GB-green.svg)](#)
[![Pretrain-Platform](https://img.shields.io/badge/Pretraining-Google_TPU_v5e--128-orange.svg)](#)
[![Context-Window](https://img.shields.io/badge/Context-262k_Tokens-purple.svg)](#)

## 📌 Executive Summary
**Hetman-2.0B** («Гетьман 2б») is an open-source, edge-native foundation model engineered from first principles for high-throughput pre-training on **Google Cloud TPU v5e-128 Pod Slices** (via the Google TPU Research Cloud grant program) and cost-effective local inference on accessible consumer graphics cards with **>= 6 GB VRAM** (NVIDIA GeForce RTX 2060 6GB+, RTX 3050/3060, and RTX 4050/4060 Mobile/Desktop).

* **Nominal Capacity:** 2,137,522,176 parameters (~2.138B storage footprint).
* **Active Compute per Token:** ~1,602,748,416 parameters (~1.603B active compute) via Top-32 sparse associative memory retrieval.
* **Weight Precision:** BitNet b1.58 ternary $\{-1, 0, +1\}$ augmented with orthogonal Fast Walsh-Hadamard Transforms ($B_{\text{had}} = 512$, quantization loss bounded to $< 1.8\%$ vs. FP32).
* **Context Window:** 262,144 tokens (256k) with zero memory explosion, enabled by 28 Gated DeltaNet recurrent memory layers ($O(1)$ state) and 6 interleaved Global Softmax FlashAttention layers.
* **Vocabulary:** 65,536 tokens (Cyrillic-optimized SentencePiece BPE, achieving $\le 1.18$ tokens/word on Ukrainian and Cyrillic corpora).
* **Conversational Alignment:** Native ChatML framing with `<think>...</think>` internal reasoning tags, aligned via Phase 3 (5.0B tokens uncensored SFT) for zero-refusal, high-compliance objective counsel ready for immediate out-of-the-box serving in `Rada.cpp`.

---

## 🏛️ Architecture Overview
Hetman-2.0B breaks the monolithic memory bottleneck in sub-3B models by enforcing a 4-stage sequential Directed Acyclic Graph (DAG):

1. **Stage 1: Deep Denoising Stem (8 Layers)**  
   Contextual denoising and syntactic parsing utilizing 6 Gated DeltaNet recurrent layers ($O(1)$ memory, 128 KB/layer) interleaved with 2 Global Softmax FlashAttention layers with bounded QK-Norm ($|S_{ij}| \le \sqrt{128} \approx 11.31$).
2. **Stage 2: Triad Concurrent Processing (Single-Pass Multi-Branch)**  
   * **Branch A: Speculative Latent-Hopfield Core (SLH-Core):** 8,388,608 discrete attractor slots (536M ternary parameters packed into 105 MB) with sub-linear Top-32 associative retrieval.
   * **Branch B: Continuous Representation Backbone (8 SwiGLU Layers):** Dedicated domain syntax and continuous representation manifold.
   * **Branch C: Hypothesis Extraction Subnetwork (6 Layers):** Preliminary deductive causal traces and hypothesis generation.
   * **Adaptive Dynamic Fusion Gate:** Channel-wise gating integrating context, associative facts, and preliminary hypotheses.
3. **Stage 3: Verified Logic Arbiter (16 Layers)**  
   Deep analytical reasoning engine with 12 Gated DeltaNet layers and 4 Global Softmax layers, terminating in a mandatory Global Arbiter Exit.
4. **Stage 4: Exit Stem & Multi-Highway Decoding (4 Layers)**  
   Variance-normalized residual recombination ($1/\sqrt{3}$ RMSNorm) and output synthesis tied to the Cyrillic-optimized 65k embedding matrix.

---

## 📂 Repository Layout
```
hetman-core/
├── HETMAN_2B_TECHNICAL_SPECIFICATION.md  # Complete 30 KB audited technical specification
├── configs/
│   └── tpu_v5e_128_config.json          # WSD training schedule & TPU v5e-128 topology (850B tokens)
├── src/
│   ├── configuration_hetman.py           # HetmanConfig dataclass
│   ├── quantization.py                  # BitLinear (1.58-bit) & Fast Walsh-Hadamard Transform (FWHT 512)
│   ├── deltanet.py                      # Gated DeltaNet linear recurrent attention O(1)
│   ├── hopfield_core.py                 # Speculative Latent-Hopfield Core (SLH-Core Top-32)
│   ├── modeling_hetman.py               # Complete 4-Stage DAG model implementation
│   └── test_verification.py             # Numerical sanity & verification test suite (CPU/laptop ready)
└── docs/
    └── GOOGLE_TRC_GRANT_APPLICATION.md  # Official Google Cloud TRC compute grant documentation
```

---

## ⚡ Quick Verification (Local Sanity Test)
You can verify the numerical integrity and tensor algebra on any modern laptop CPU in under 5 seconds (requires `< 100 MB` RAM):

```bash
# Clone the repository
git clone https://github.com/YaroslavSamchuk/Hetman-2b.git
cd Hetman-2b

# Run verification suite (verifies BitLinear, FWHT, DeltaNet, QK-Norm, and Hopfield Core)
python src/test_verification.py
```

---

## 🚀 Native Inference Runtime: `Rada.cpp`
For high-performance offline inference on consumer GPUs (NVIDIA RTX 2060 6GB+, RTX 3050/3060, RTX 4050/4060 Mobile/Desktop, and Apple Silicon), use the companion runtime **[Rada.cpp](https://github.com/YaroslavSamchuk/Rada.cpp)**:
* **Add-Only DP4A Kernels:** Eliminates FP16 multiplication in favor of 2-bit integer additions.
* **Compact Memory:** 256k context requires only **~2.00 GB VRAM**, leaving **4.00 GB free** on a standard 6 GB GPU.
* **Dual Modalities:** Includes `rada_cli` for terminal power-users and `rada_web` with an embedded zero-dependency browser chat UI.

---

## 📜 License & Open Source Charter
This project is licensed under the permissive **Apache License 2.0**. All architecture specifications, model weights, training recipes, tokenizer files, and JAX/Pallas kernels will be released publicly without commercial restrictions.
