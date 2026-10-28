---
library_name: transformers
---

# Hetman-2b experimental package

This directory contains experimental model/configuration code for the Hetman research project. It does **not** contain pretrained model weights and is not a ready-to-use chatbot.

The implementation uses dense floating-point trainable tensors. Ternary fake quantization, configured long context, and associative-memory behavior are research prototypes; packed low-bit kernels, long-context performance, inference speed, factuality, and safety have not been validated.

The project has not completed full-scale pretraining or a corpus-backed run. The single-node trainer has only passed small synthetic CPU checks. The companion Rada.cpp project does not currently perform model inference.

See the [main repository](https://github.com/YaroslavSamchuk/Hetman-2b) for the current development status, data gates, and usage notes. This repository currently has no `LICENSE` file; reuse and redistribution terms are not declared.
