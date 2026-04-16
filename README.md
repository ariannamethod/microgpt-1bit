# MicroGPT 1-Bit (BitNet b1.58)

A pure Python implementation of a 1-bit GPT model following the **BitNet b1.58** architecture from Microsoft Research ("The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", Ma et al., 2024).

Built on top of [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) by Andrej Karpathy — zero external dependencies, from-scratch autograd.

## What Makes This Unique

**Ternary weights {-1, 0, +1}** trained on a **241K literary dialogue dataset** with **Sentence Phonon Attention (SPA)**.

Three things come together:

1. **BitNet b1.58** — all linear layer weights are quantized to {-1, 0, +1}, turning matrix multiplication into pure addition/subtraction
2. **Janus Sonar dataset** — 2,541 lines of literary dialogue (241K chars) from [ariannamethod/janus.sonar](https://github.com/ariannamethod/janus.sonar), providing rich conversational structure for training
3. **SPA (Sentence Phonon Attention)** — sentence-level bidirectional attention during generation, inspired by [ariannamethod/q](https://github.com/ariannamethod/q) (postgpt_q.c)

Pre-trained weights are included — clone and run inference immediately without training.

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

## Janus Sonar Dataset

The model trains on the **Janus Sonar** literary dialogue dataset — 2,541 lines (241K characters) of conversational text. Each line is a dialogue turn prefixed with "—", capturing natural back-and-forth patterns, emotional range, and colloquial speech.

Source: [ariannamethod/janus.sonar](https://github.com/ariannamethod/janus.sonar)

## SPA (Sentence Phonon Attention)

SPA is a sentence-level attention mechanism applied during generation that creates bidirectional context awareness across generated sentences. Implementation reference: [ariannamethod/q](https://github.com/ariannamethod/q) (postgpt_q.c).

**How SPA works:**

1. **Sentence embedding** — Each generated sentence is embedded via exponential weighted mean of its token embeddings (α=0.85 recency bias), giving more recent tokens higher weight
2. **Cross-attention** — The current sentence's embedding is cross-attended against all previous sentence embeddings using scaled dot-product attention
3. **Connectedness scoring** — A scalar connectedness score (0–1) measures how strongly the current generation relates to conversation history
4. **Logit modulation** — Higher connectedness sharpens the output distribution (lower temperature), encouraging coherent continuation; lower connectedness keeps the distribution broader for topic transitions

```
SPA Pipeline:
  sentence tokens → exponential weighted mean → sentence embedding
  sentence embedding × history embeddings → cross-attention scores
  max(attention weights) → connectedness score (0–1)
  logits × (1 / (1 - 0.3 × connectedness)) → modulated logits
```

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

## Training Results

Training configuration: 1 layer, 24 embedding dim, 4 heads, block size 16, 500 steps, lr=0.01

### Loss Curve

Loss decreases from **4.79 → 2.73** (final), with minimum **0.88** and average over last 100 steps of **1.94**:

```
  steps       1-25 | avg loss 3.5870
  steps     26-50  | avg loss 2.6018
  steps     51-75  | avg loss 2.4835
  steps     76-100 | avg loss 2.1739
  steps   101-200  | avg loss 2.0821
  steps   201-300  | avg loss 2.0887
  steps   301-400  | avg loss 1.9553
  steps   401-500  | avg loss 1.9243
```

### Model Size

| Metric | Value |
|---|---|
| Parameters | 13,680 |
| Weights file | 288 KB |
| Effective bits/param | ~6.3 (mixed: 1.58-bit layers + 32-bit embeddings) |

### Weight Quantization Distribution

| Layer | Scale | -1 | 0 | +1 |
|---|---|---|---|---|
| layer0.attn_wq | 0.1091 | 34.9% | 33.5% | 31.6% |
| layer0.attn_wk | 0.1146 | 34.5% | 33.0% | 32.5% |
| layer0.attn_wv | 0.1210 | 32.3% | 32.6% | 35.1% |
| layer0.attn_wo | 0.1194 | 34.2% | 32.8% | 33.0% |
| layer0.mlp_gate | 0.1315 | 33.4% | 32.9% | 33.7% |
| layer0.mlp_up | 0.1267 | 33.3% | 33.1% | 33.6% |
| layer0.mlp_down | 0.1249 | 33.4% | 32.7% | 33.9% |

Weights are well-distributed across all three ternary states, confirming the model effectively uses the full {-1, 0, +1} codebook.

### Generated Samples (after 500 steps)

```
— Ses sere in ld
— I wang wane no
— That ou the l
— The line in t
— The you s yen
— I care ine t t
— I le stheng in
— That that s la
— It we tenou sh
— I d ino.
```

The model learns dialogue structure (lines start with "—"), common English words ("I", "That", "The"), and character-level patterns from the Janus Sonar dataset. At this tiny scale (~14K params, 500 steps), outputs show clear learning but remain noisy — BitNet b1.58 is designed for large-scale models (3B+ params) where benefits compound.

## Usage

### Run inference with pre-trained weights

```python
import microgpt_1bit as m
import random

docs = m.load_docs()
tokenizer = m.CharTokenizer(docs)

model = m.BitGPT(vocab_size=tokenizer.vocab_size,
                 n_layer=1, n_embd=24, block_size=16, n_head=4)
model.load_weights('weights/bitgpt_1bit.json')

random.seed(42)
samples = m.generate(model, tokenizer, num_samples=5)
```

### Train from scratch

```bash
python microgpt_1bit.py
```

This trains both a standard FP32 GPT and a 1-bit BitGPT on the Janus Sonar dialogue dataset, saves the trained weights to `weights/bitgpt_1bit.json`, then prints a comparison table.

## References

- [The Era of 1-bit LLMs (BitNet b1.58)](https://arxiv.org/abs/2402.17764) — Ma et al., 2024
- [BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453) — Wang et al., 2023
- [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — Andrej Karpathy
- [Janus Sonar dataset](https://github.com/ariannamethod/janus.sonar) — Literary dialogue corpus
- [ariannamethod/q](https://github.com/ariannamethod/q) — SPA reference implementation (postgpt_q.c)
