# MicroGPT 1-Bit (BitNet b1.58)

A pure Python implementation of a 1-bit GPT model following the **BitNet b1.58** architecture from Microsoft Research ("The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", Ma et al., 2024).

Built on top of [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) by Andrej Karpathy — zero external dependencies, from-scratch autograd.

## What is a 1-Bit Model?

Standard neural networks use 32-bit or 16-bit floating point weights. BitNet b1.58 constrains all linear layer weights to just three values: **{-1, 0, +1}** — requiring only 1.58 bits per weight (log₂(3) = 1.58).

This means matrix multiplication becomes **pure integer addition**:

| Weight | Operation |
|--------|-----------|
| +1 | Add activation to accumulator |
| -1 | Subtract activation from accumulator |
| 0 | Skip (no-op) |

## Architecture

```
┌─────────────────────────────────────────────┐
│  Token Embedding (wte) — full precision     │
│  Position Embedding (wpe) — full precision  │
│  RMSNorm                                    │
├─────────────────────────────────────────────┤
│  Transformer Block (× n_layer)              │
│  ┌─────────────────────────────────────┐    │
│  │  RMSNorm                            │    │
│  │  Q = BitLinear(x)  ← ternary {-1,0,1}   │
│  │  K = BitLinear(x)  ← ternary        │    │
│  │  V = BitLinear(x)  ← ternary        │    │
│  │  Multi-Head Attention (standard)     │    │
│  │  O = BitLinear(attn_out) ← ternary  │    │
│  │  + Residual                          │    │
│  ├─────────────────────────────────────┤    │
│  │  RMSNorm                            │    │
│  │  SwiGLU MLP:                         │    │
│  │    gate = BitLinear(x) ← ternary     │    │
│  │    up   = BitLinear(x) ← ternary     │    │
│  │    h = SiLU(gate) * up               │    │
│  │    out  = BitLinear(h) ← ternary     │    │
│  │  + Residual                          │    │
│  └─────────────────────────────────────┘    │
├─────────────────────────────────────────────┤
│  LM Head — full precision                   │
└─────────────────────────────────────────────┘
```

### What's quantized (1.58 bits)
- All attention projections: Q, K, V, O
- All MLP projections: gate, up, down

### What stays full precision
- Token & position embeddings (lookup tables)
- Output head (lm_head)
- RMSNorm (parameter-free)
- Attention scores, softmax, residual connections

## Key Techniques

### Quantization-Aware Training (QAT)
Weights are quantized during every forward pass but full-precision "latent" copies are maintained for gradient updates.

### Straight-Through Estimator (STE)
Since quantization has zero gradient almost everywhere, the backward pass treats quantization as identity — gradients flow through as if weights were never quantized.

### Absmean Weight Quantization
```
gamma = mean(|W|)
W_ternary = clamp(round(W / gamma), -1, 1)
```

### Absmax Activation Quantization (INT8)
```
Qb = 128
gamma = max(|x|)
x_quant = clamp(round(x * Qb / gamma), -Qb, Qb-1)
```

### SwiGLU Activation
Replaces ReLU in the MLP block. Better suited for ternary weight models:
```
SwiGLU(x) = SiLU(x @ W_gate) * (x @ W_up)
```

## Results

Training configuration: 1 layer, 24 embedding dim, 4 heads, 1000 steps, lr=0.01

### Loss Comparison

| Metric | FP32 Baseline | 1-Bit BitGPT |
|---|---|---|
| Final loss | 2.6092 | 2.4709 |
| Min loss | 1.5908 | 1.5563 |
| Avg loss (last 100 steps) | 2.2811 | 2.2732 |

### Model Size Comparison

| Metric | FP32 Baseline | 1-Bit BitGPT |
|---|---|---|
| Parameters | 8,592 | 10,896 |
| Effective bits/param | 32.00 | 6.27 |
| Theoretical memory | 33.6 KB | 8.8 KB |
| Memory reduction | 1.0x | **3.8x** |

The 1-bit model has more parameters due to SwiGLU (3 matrices vs 2 in the MLP), but uses ~3.8x less memory because most weights are stored as ternary values (2 bits each) instead of 32-bit floats.

### Weight Quantization Distribution (After Training)

| Layer | Scale | -1 | 0 | +1 |
|---|---|---|---|---|
| layer0.attn_wq | 0.1341 | 30.9% | 32.3% | 36.8% |
| layer0.attn_wk | 0.1309 | 39.2% | 33.5% | 27.3% |
| layer0.attn_wv | 0.1064 | 34.7% | 30.9% | 34.4% |
| layer0.attn_wo | 0.1076 | 32.1% | 31.8% | 36.1% |
| layer0.mlp_gate | 0.1132 | 30.3% | 33.7% | 36.1% |
| layer0.mlp_up | 0.1077 | 34.8% | 33.1% | 32.2% |
| layer0.mlp_down | 0.1076 | 34.2% | 32.6% | 33.2% |

Weights are well-distributed across all three ternary states, confirming the model effectively uses the full {-1, 0, +1} codebook.

### Generated Samples

**FP32 Baseline:**
```
akakud, celen, anana, erakia, jaria, aneren, jari, iela, arsian, anelan
```

**1-Bit BitGPT:**
```
akelya, aralena, anale, alia, javia, anena, marin, jana, anria, danian
```

Both models produce plausible name-like outputs. The 1-bit model generates slightly more natural-sounding names.

### Key Takeaway

At this tiny scale (~10K params), the 1-bit model matches FP32 quality while using 3.8x less memory. BitNet b1.58 is designed for large-scale models (3B+ params) where the benefits compound — at 70B parameters, Microsoft reports 4.1x lower latency, 3.55x less memory, and 41x energy savings.

## Usage

```bash
python microgpt_1bit.py
```

This trains both a standard FP32 GPT and a 1-bit BitGPT on character-level name generation, then prints a comparison table.

## References

- [The Era of 1-bit LLMs (BitNet b1.58)](https://arxiv.org/abs/2402.17764) — Ma et al., 2024
- [BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453) — Wang et al., 2023
- [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — Andrej Karpathy
