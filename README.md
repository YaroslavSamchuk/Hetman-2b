# Hetman-2.0B: Sovereign Open-Source Foundation Model
### 2.14B Total Capacity (~1.603B Active) | BitNet b1.58 Ternary | 256k Context | Google TPU v5e-128 Pod

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Target-Hardware](https://img.shields.io/badge/Target_Edge-NVIDIA_RTX_2060+_>=6GB-green.svg)](#)
[![Pretrain-Platform](https://img.shields.io/badge/Pretraining-Google_TPU_v5e--128-orange.svg)](#)

## 📌 Огляд проєкту
**«Гетьман 2б» (Hetman-2.0B)** — це суверенна відкрита фундаментна модель штучного інтелекту, спроєктована для високоефективного навчання на кластерах **Google Cloud TPU v5e-128 Pod Slice** (грантова програма Google TRC) та автономного локального запуску на доступних споживчих відеокартах із пам'яттю від **6 ГБ VRAM** (NVIDIA GeForce RTX 2060, RTX 3050/3060, RTX 4050/4060).

* **Повна ємність:** 2,137,522,176 параметрів (~2.138B).
* **Активний комп'ют на токен:** ~1,602,748,416 параметрів (~1.603B) завдяки розрідженій асоціативній пам'яті Хопфілда Top-32.
* **Формат ваг:** BitNet b1.58 тернарний $\{-1, 0, +1\}$ з ортогональними ротаціями Адамара ($B_{\text{had}} = 512$, втрата точності $< 1.8\%$ проти FP32).
* **Контекстне вікно:** 262,144 токени (256k) без переповнення пам'яті завдяки 28 рекурентним шарам Gated DeltaNet ($O(1)$ стан) та 6 глобальним шарам уваги.
* **Словник:** 65,536 токенів (Cyrillic-optimized SentencePiece BPE, фертильність для української мови $\le 1.18$).

## 📂 Структура репозиторію
```
hetman-core/
├── HETMAN_2B_TECHNICAL_SPECIFICATION.md  # Повна специфікація архітектури
├── configs/                              # WSD розклад, TPU топологія (850B токенів)
├── src/                                  # Претрейн-код (JAX, Flax, Pallas)
│   ├── training/                         # Цикл претрейну та Orbax-чекпоінти
│   └── rl_gagar/                         # In-Process AST & законодавство України
├── tokenizer/                            # Тренування та конфігурація 65k токенізатора
└── docs/                                 # Матеріали гранту Google TRC
```

## 🚀 Рушій інференсу
Для запуску моделі на споживчих відеокартах GeForce RTX від 6 ГБ VRAM використовуйте супутній проєкт **[Rada.cpp](../rada.cpp)** — автономний C++20 / CUDA рушій.
