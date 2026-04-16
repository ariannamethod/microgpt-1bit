"""
A 1-bit (BitNet b1.58) GPT implementation in pure, dependency-free Python.

Based on microgpt by Andrej Karpathy, extended with ternary weight quantization
following "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
(Ma et al., Microsoft Research, 2024).

Key idea: all linear layer weights are quantized to {-1, 0, +1} during the
forward pass using absmean quantization, while activations are quantized to
8-bit integers using absmax quantization. Gradients flow through quantization
via the Straight-Through Estimator (STE).

Training maintains latent full-precision weights updated by Adam. Only the
forward pass uses quantized weights. Embeddings, RMSNorm, and the output head
remain in full precision.

Includes SPA (Sentence Phonon Attention) for sentence-level bidirectional
attention during generation. Trained on the Janus Sonar literary dialogue
dataset (ariannamethod/janus.sonar).
"""

import os
import json
import math
import random

# ---------------------------------------------------------------------------
# Autograd engine
# ---------------------------------------------------------------------------

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):        return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other):  return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other):  return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            v, processed = stack.pop()
            if v in visited:
                continue
            if processed:
                visited.add(v)
                topo.append(v)
            else:
                stack.append((v, True))
                for child in v._children:
                    if child not in visited:
                        stack.append((child, False))
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class CharTokenizer:
    def __init__(self, docs):
        self.chars = sorted(set(''.join(docs)))
        self.bos = len(self.chars)
        self.vocab_size = len(self.chars) + 1

    def encode(self, text):
        return [self.bos] + [self.chars.index(ch) for ch in text] + [self.bos]

    def decode_token(self, token_id):
        return self.chars[token_id]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_docs(path='input.txt', url=None, seed=42):
    if not os.path.exists(path):
        import urllib.request
        url = url or 'https://raw.githubusercontent.com/ariannamethod/janus.sonar/main/dataset.txt'
        urllib.request.urlretrieve(url, path)
    docs = [line.strip() for line in open(path, encoding='utf-8') if line.strip()]
    random.seed(seed)
    random.shuffle(docs)
    return docs

# ---------------------------------------------------------------------------
# Neural network primitives
# ---------------------------------------------------------------------------

def linear(x, w):
    """Standard full-precision linear layer: y = W @ x"""
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def sigmoid(x):
    return ((-x).exp() + 1) ** -1

# ---------------------------------------------------------------------------
# 1-bit quantization primitives (BitNet b1.58)
# ---------------------------------------------------------------------------

def weight_quantize(w):
    """
    Absmean quantization for weights -> {-1, 0, +1}.

    Given a weight matrix w (list of lists of Value), compute:
        gamma = mean(|w|)
        w_ternary = clamp(round(w / gamma), -1, 1)

    Returns (w_ternary_data, gamma) where w_ternary_data is a list of lists
    of float values in {-1, 0, 1} and gamma is a scalar float.
    """
    flat = [v.data for row in w for v in row]
    gamma = sum(abs(v) for v in flat) / len(flat) + 1e-8
    w_ternary = []
    for row in w:
        q_row = []
        for v in row:
            q = round(v.data / gamma)
            q = max(-1, min(1, q))  # clamp to {-1, 0, 1}
            q_row.append(q)
        w_ternary.append(q_row)
    return w_ternary, gamma

def activation_quantize(x, num_bits=8):
    """
    Per-token absmax quantization for activations -> INT8 range.

    Given activation vector x (list of Value), compute:
        Qb = 2^(num_bits - 1) = 128
        gamma = max(|x|)
        x_quant = clamp(round(x * Qb / gamma), -Qb, Qb-1)

    Returns (x_quant_data, scale) where x_quant_data is a list of float
    values in [-128, 127] and scale = gamma / Qb.
    """
    Qb = 2 ** (num_bits - 1)  # 128
    gamma = max(abs(v.data) for v in x) + 1e-8
    x_quant = []
    for v in x:
        q = round(v.data * Qb / gamma)
        q = max(-Qb, min(Qb - 1, q))
        x_quant.append(q)
    return x_quant, gamma / Qb

def bitlinear(x, w):
    """
    1-bit linear layer with Straight-Through Estimator (STE).

    Forward pass:
        1. Quantize weights to {-1, 0, +1} via absmean
        2. Quantize activations to INT8 via absmax
        3. Compute matmul (integer addition in principle)
        4. Rescale output by weight_scale * activation_scale

    Backward pass (STE):
        Gradients flow through the original full-precision weights as if
        quantization were the identity function. This is achieved by
        computing the output as:
            y = w_quant @ x_quant  (using quantized values for forward)
        but adding: (y_quant - y_fp).detach() + y_fp
        so that the gradient graph connects to the original weights.

    In this pure-Python autograd, we implement STE by computing the
    quantized output value but constructing the gradient path through
    the original weight Values.
    """
    # Quantize weights and activations (forward-pass values only)
    w_q, w_scale = weight_quantize(w)
    x_q, x_scale = activation_quantize(x)

    # Compute matmul using quantized values, but route gradients through
    # original weights via STE.
    #
    # For each output element:
    #   y_quant = sum(w_q[i][j] * x_q[j])  -- pure integer arithmetic
    #   y_fp    = sum(w[i][j] * x[j])       -- full precision (for grad)
    #   output  = y_fp + (y_quant * scale - y_fp).detach()
    #           = y_quant * scale  (forward value)
    #           with grad flowing through y_fp (backward)
    #
    # Implemented as: for each element, compute y_fp through the autograd
    # graph, then adjust its .data to the quantized result.

    out = []
    total_scale = w_scale * x_scale
    for i, row in enumerate(w):
        # Full-precision path through autograd (for gradient flow)
        y_fp = sum(wi * xi for wi, xi in zip(row, x))

        # Quantized forward value
        y_quant_val = sum(w_q[i][j] * x_q[j] for j in range(len(x)))
        y_scaled = y_quant_val * total_scale

        # STE: use quantized value in forward, but keep gradient graph
        # We adjust the .data of the autograd node to the quantized result
        y_fp.data = y_scaled

        out.append(y_fp)
    return out

# ---------------------------------------------------------------------------
# SwiGLU activation (recommended for BitNet b1.58)
# ---------------------------------------------------------------------------

def swiglu(x, w_gate, w_up):
    """
    SwiGLU activation: SiLU(x @ W_gate) * (x @ W_up)
    where SiLU(z) = z * sigmoid(z)

    Uses bitlinear for both projections.
    """
    gate = bitlinear(x, w_gate)
    up = bitlinear(x, w_up)
    # SiLU(gate) * up
    return [g * sigmoid(g) * u for g, u in zip(gate, up)]

# ---------------------------------------------------------------------------
# SPA (Sentence Phonon Attention)
# ---------------------------------------------------------------------------
# SPA provides sentence-level bidirectional attention between generation steps.
# Reference: ariannamethod/q (postgpt_q.c)

def spa_embed_sentence(tokens, W_embed, alpha=0.85):
    """
    Exponential weighted mean of token embeddings for a sentence.
    More recent tokens receive higher weight (recency bias).
    """
    dim = len(W_embed[0]) if W_embed else 24
    out = [0.0] * dim
    total_w = 0.0
    for i, tok in enumerate(tokens):
        if tok >= len(W_embed):
            continue
        w = alpha ** (len(tokens) - 1 - i)
        for d in range(dim):
            emb_val = W_embed[tok][d]
            out[d] += w * (emb_val.data if hasattr(emb_val, 'data') else emb_val)
        total_w += w
    return [x / (total_w + 1e-8) for x in out]


def spa_cross_attend(query_emb, sentence_embeddings):
    """
    Cross-attend between a query sentence embedding and all previous
    sentence embeddings. Returns a connectedness score (0 to 1) per
    previous sentence via scaled dot-product attention.
    """
    if not sentence_embeddings:
        return []
    dim = len(query_emb)
    scale = dim ** 0.5
    scores = []
    for emb in sentence_embeddings:
        dot = sum(q * k for q, k in zip(query_emb, emb))
        scores.append(dot / scale)
    # softmax over scores
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps) + 1e-8
    return [e / total for e in exps]


def spa_connectedness(query_emb, sentence_embeddings):
    """
    Compute a single scalar connectedness score (0 to 1) representing
    how strongly the current sentence relates to the conversation history.
    Uses the max attention weight as the connectedness measure.
    """
    if not sentence_embeddings:
        return 0.5  # neutral when no history
    weights = spa_cross_attend(query_emb, sentence_embeddings)
    return max(weights) if weights else 0.5


# ---------------------------------------------------------------------------
# 1-Bit GPT Model (BitNet b1.58)
# ---------------------------------------------------------------------------

class BitGPT:
    """
    GPT with BitLinear layers following BitNet b1.58.

    Changes from standard GPT:
    - All attention projections (Q, K, V, O) use bitlinear
    - MLP uses SwiGLU with bitlinear (3 weight matrices instead of 2)
    - No biases anywhere
    - Embeddings (wte, wpe) and output head (lm_head) remain full precision
    - RMSNorm remains full precision (no learnable parameters anyway)
    """
    def __init__(self, vocab_size, n_layer=1, n_embd=24, block_size=16, n_head=4, init_std=0.08):
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        def matrix(nout, nin):
            return [[Value(random.gauss(0, init_std)) for _ in range(nin)] for _ in range(nout)]

        # Embeddings and output head: full precision (not quantized)
        self.weights = {
            'wte': matrix(vocab_size, n_embd),
            'wpe': matrix(block_size, n_embd),
            'lm_head': matrix(vocab_size, n_embd),
        }

        for i in range(n_layer):
            # Attention projections (will use bitlinear)
            self.weights[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
            self.weights[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
            self.weights[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
            self.weights[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)

            # SwiGLU MLP (3 matrices: gate, up, down)
            # gate and up project from n_embd -> 4*n_embd
            # down projects from 4*n_embd -> n_embd
            self.weights[f'layer{i}.mlp_gate'] = matrix(4 * n_embd, n_embd)
            self.weights[f'layer{i}.mlp_up'] = matrix(4 * n_embd, n_embd)
            self.weights[f'layer{i}.mlp_down'] = matrix(n_embd, 4 * n_embd)

    def parameters(self):
        return [p for mat in self.weights.values() for row in mat for p in row]

    def forward(self, token_id, pos_id, kv_cache):
        keys, values = kv_cache
        w = self.weights

        # Embedding + positional encoding (full precision)
        x = [t + p for t, p in zip(w['wte'][token_id], w['wpe'][pos_id])]
        x = rmsnorm(x)

        for li in range(self.n_layer):
            # ----- Attention block -----
            x_res = x
            x = rmsnorm(x)

            # Q, K, V projections via bitlinear
            q = bitlinear(x, w[f'layer{li}.attn_wq'])
            k = bitlinear(x, w[f'layer{li}.attn_wk'])
            v = bitlinear(x, w[f'layer{li}.attn_wv'])

            keys[li].append(k)
            values[li].append(v)

            # Multi-head attention (standard scaled dot-product)
            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs:hs+self.head_dim]
                k_h = [ki[hs:hs+self.head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs+self.head_dim] for vi in values[li]]
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(self.head_dim)) / self.head_dim**0.5
                    for t in range(len(k_h))
                ]
                attn_weights = softmax(attn_logits)
                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(self.head_dim)
                ]
                x_attn.extend(head_out)

            # Output projection via bitlinear + residual
            x = bitlinear(x_attn, w[f'layer{li}.attn_wo'])
            x = [a + b for a, b in zip(x, x_res)]

            # ----- MLP block (SwiGLU with bitlinear) -----
            x_res = x
            x = rmsnorm(x)

            # SwiGLU: SiLU(x @ W_gate) * (x @ W_up), then project down
            x = swiglu(x, w[f'layer{li}.mlp_gate'], w[f'layer{li}.mlp_up'])
            x = bitlinear(x, w[f'layer{li}.mlp_down'])

            x = [a + b for a, b in zip(x, x_res)]

        # Output head: full precision
        return linear(x, w['lm_head'])

    def new_kv_cache(self):
        return ([[] for _ in range(self.n_layer)],
                [[] for _ in range(self.n_layer)])

    def weight_stats(self):
        """Print quantization statistics for all bitlinear layers."""
        print("\n--- weight quantization stats ---")
        for name, mat in self.weights.items():
            if name in ('wte', 'wpe', 'lm_head'):
                print(f"{name:30s}: full precision (not quantized)")
                continue
            w_q, gamma = weight_quantize(mat)
            flat = [v for row in w_q for v in row]
            n_neg = sum(1 for v in flat if v == -1)
            n_zero = sum(1 for v in flat if v == 0)
            n_pos = sum(1 for v in flat if v == 1)
            total = len(flat)
            print(f"{name:30s}: scale={gamma:.4f}  "
                  f"-1: {n_neg:4d} ({100*n_neg/total:5.1f}%)  "
                  f" 0: {n_zero:4d} ({100*n_zero/total:5.1f}%)  "
                  f"+1: {n_pos:4d} ({100*n_pos/total:5.1f}%)")

    def save_weights(self, path):
        """Save model weights to a JSON file."""
        data = {
            'config': {
                'n_layer': self.n_layer,
                'n_embd': self.n_embd,
                'block_size': self.block_size,
                'n_head': self.n_head,
            },
            'weights': {}
        }
        for name, mat in self.weights.items():
            data['weights'][name] = [[v.data for v in row] for row in mat]
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"Saved weights to {path} ({size_kb:.1f} KB)")

    def load_weights(self, path):
        """Load model weights from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        for name, mat_data in data['weights'].items():
            if name not in self.weights:
                continue
            for i, row in enumerate(mat_data):
                for j, val in enumerate(row):
                    self.weights[name][i][j].data = val
        print(f"Loaded weights from {path}")

# ---------------------------------------------------------------------------
# Standard GPT (baseline for comparison)
# ---------------------------------------------------------------------------

class GPT:
    """Standard full-precision GPT for A/B comparison."""
    def __init__(self, vocab_size, n_layer=1, n_embd=24, block_size=16, n_head=4, init_std=0.08):
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        def matrix(nout, nin):
            return [[Value(random.gauss(0, init_std)) for _ in range(nin)] for _ in range(nout)]

        self.weights = {
            'wte': matrix(vocab_size, n_embd),
            'wpe': matrix(block_size, n_embd),
            'lm_head': matrix(vocab_size, n_embd),
        }
        for i in range(n_layer):
            self.weights[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
            self.weights[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
            self.weights[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
            self.weights[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
            self.weights[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
            self.weights[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

    def parameters(self):
        return [p for mat in self.weights.values() for row in mat for p in row]

    def forward(self, token_id, pos_id, kv_cache):
        keys, values = kv_cache
        w = self.weights
        x = [t + p for t, p in zip(w['wte'][token_id], w['wpe'][pos_id])]
        x = rmsnorm(x)

        for li in range(self.n_layer):
            x_res = x
            x = rmsnorm(x)
            q = linear(x, w[f'layer{li}.attn_wq'])
            k = linear(x, w[f'layer{li}.attn_wk'])
            v = linear(x, w[f'layer{li}.attn_wv'])
            keys[li].append(k)
            values[li].append(v)

            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs:hs+self.head_dim]
                k_h = [ki[hs:hs+self.head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs+self.head_dim] for vi in values[li]]
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(self.head_dim)) / self.head_dim**0.5
                    for t in range(len(k_h))
                ]
                attn_weights = softmax(attn_logits)
                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(self.head_dim)
                ]
                x_attn.extend(head_out)

            x = linear(x_attn, w[f'layer{li}.attn_wo'])
            x = [a + b for a, b in zip(x, x_res)]

            x_res = x
            x = rmsnorm(x)
            x = linear(x, w[f'layer{li}.mlp_fc1'])
            x = [xi.relu() for xi in x]
            x = linear(x, w[f'layer{li}.mlp_fc2'])
            x = [a + b for a, b in zip(x, x_res)]

        return linear(x, w['lm_head'])

    def new_kv_cache(self):
        return ([[] for _ in range(self.n_layer)],
                [[] for _ in range(self.n_layer)])

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, params, lr=0.01, beta1=0.85, beta2=0.99, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0] * len(params)
        self.v = [0.0] * len(params)
        self.step_count = 0

    def step(self, lr_override=None):
        self.step_count += 1
        lr = lr_override if lr_override is not None else self.lr
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.step_count)
            v_hat = self.v[i] / (1 - self.beta2 ** self.step_count)
            p.data -= lr * m_hat / (v_hat ** 0.5 + self.eps)
            p.grad = 0

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, tokenizer, docs, num_steps=1000, lr=0.01, label="model"):
    params = model.parameters()
    optimizer = Adam(params, lr=lr)
    loss_history = []

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = tokenizer.encode(doc)
        n = min(model.block_size, len(tokens) - 1)

        kv_cache = model.new_kv_cache()
        losses = []
        for pos_id in range(n):
            logits = model.forward(tokens[pos_id], pos_id, kv_cache)
            probs = softmax(logits)
            losses.append(-probs[tokens[pos_id + 1]].log())
        loss = (1 / n) * sum(losses)

        loss.backward()

        lr_t = lr * (1 - step / num_steps)
        optimizer.step(lr_override=lr_t)

        loss_history.append(loss.data)
        print(f"[{label}] step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

    print()
    return loss_history

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate(model, tokenizer, num_samples=10, temperature=0.5, label="model"):
    """Generate dialogue lines with SPA-modulated logits."""
    print(f"--- {label}: inference (generated dialogue) ---")
    samples = []
    sentence_embeddings = []  # SPA: accumulated sentence embeddings

    # Get the embedding table for SPA
    W_embed = model.weights['wte']

    for i in range(num_samples):
        kv_cache = model.new_kv_cache()
        token_id = tokenizer.bos
        chars = []
        sent_tokens = []  # tokens in current sentence for SPA

        for pos_id in range(model.block_size):
            logits = model.forward(token_id, pos_id, kv_cache)

            # SPA: modulate logits using sentence-level attention
            if sentence_embeddings and sent_tokens:
                current_emb = spa_embed_sentence(sent_tokens, W_embed)
                conn = spa_connectedness(current_emb, sentence_embeddings)
                # Scale logits by connectedness: higher connection = sharper distribution
                spa_temp = 1.0 - 0.3 * conn  # range [0.7, 1.0]
                logits = [l * (1.0 / spa_temp) for l in logits]

            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(range(tokenizer.vocab_size),
                                      weights=[p.data for p in probs])[0]
            if token_id == tokenizer.bos:
                break
            chars.append(tokenizer.decode_token(token_id))
            sent_tokens.append(token_id)

        line = ''.join(chars)
        samples.append(line)

        # SPA: embed completed sentence and add to history
        if sent_tokens:
            emb = spa_embed_sentence(sent_tokens, W_embed)
            sentence_embeddings.append(emb)

        print(f"  sample {i+1:2d}: {line}")
    return samples

# ---------------------------------------------------------------------------
# Main: train both models and compare
# ---------------------------------------------------------------------------

def main():
    random.seed(42)

    docs = load_docs()
    print(f"num docs: {len(docs)}")

    tokenizer = CharTokenizer(docs)
    print(f"vocab size: {tokenizer.vocab_size}")

    # Configuration — adjusted for Janus Sonar dialogue dataset
    n_layer = 1
    n_embd = 24
    n_head = 4
    block_size = 16
    num_steps = 500
    lr = 0.01

    # --- Standard GPT (baseline) ---
    print("\n========== STANDARD GPT (FP32 baseline) ==========")
    random.seed(42)
    baseline = GPT(
        vocab_size=tokenizer.vocab_size,
        n_layer=n_layer, n_embd=n_embd, block_size=block_size, n_head=n_head
    )
    print(f"num params: {len(baseline.parameters())}")
    baseline_loss = train(baseline, tokenizer, docs, num_steps=num_steps, lr=lr, label="FP32")

    random.seed(123)
    baseline_samples = generate(baseline, tokenizer, num_samples=10, label="FP32")

    # --- 1-Bit GPT (BitNet b1.58) with SPA ---
    print("\n========== 1-BIT GPT (BitNet b1.58) + SPA ==========")
    random.seed(42)
    bitgpt = BitGPT(
        vocab_size=tokenizer.vocab_size,
        n_layer=n_layer, n_embd=n_embd, block_size=block_size, n_head=n_head
    )
    print(f"num params: {len(bitgpt.parameters())}")
    bit_loss = train(bitgpt, tokenizer, docs, num_steps=num_steps, lr=lr, label="1-bit")

    # Save trained weights
    bitgpt.save_weights('weights/bitgpt_1bit.json')

    # Show weight quantization stats
    bitgpt.weight_stats()

    random.seed(123)
    bit_samples = generate(bitgpt, tokenizer, num_samples=10, label="1-bit+SPA")

    # --- Loss curve ---
    print("\n========== LOSS CURVE (1-bit) ==========")
    n_bins = 20
    bin_size = len(bit_loss) // n_bins
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(bit_loss)
        chunk = bit_loss[start:end]
        avg = sum(chunk) / len(chunk)
        bar = '#' * int(avg * 10)
        step_label = f"{start + 1}-{end}"
        print(f"  steps {step_label:>10s} | avg loss {avg:.4f} | {bar}")

    # --- Comparison ---
    print("\n========== COMPARISON ==========")
    print(f"{'Metric':<25s} {'FP32 Baseline':>15s} {'1-Bit + SPA':>15s}")
    print("-" * 57)
    print(f"{'Final loss':<25s} {baseline_loss[-1]:>15.4f} {bit_loss[-1]:>15.4f}")
    print(f"{'Min loss':<25s} {min(baseline_loss):>15.4f} {min(bit_loss):>15.4f}")
    avg_bl = sum(baseline_loss[-100:]) / 100
    avg_bt = sum(bit_loss[-100:]) / 100
    print(f"{'Avg loss (last 100)':<25s} {avg_bl:>15.4f} {avg_bt:>15.4f}")
    print(f"{'Num parameters':<25s} {len(baseline.parameters()):>15d} {len(bitgpt.parameters()):>15d}")

    # Compute effective bits for BitGPT
    fp_params = 0
    bit_params = 0
    for name, mat in bitgpt.weights.items():
        count = sum(len(row) for row in mat)
        if name in ('wte', 'wpe', 'lm_head'):
            fp_params += count
        else:
            bit_params += count

    total = fp_params + bit_params
    effective_bits = (fp_params * 32 + bit_params * 1.58) / total
    baseline_bits = 32.0

    print(f"{'Effective bits/param':<25s} {baseline_bits:>15.2f} {effective_bits:>15.2f}")
    print(f"{'Theoretical memory':<25s} "
          f"{len(baseline.parameters()) * 32 / 8 / 1024:>14.1f}K "
          f"{(fp_params * 32 + bit_params * 2) / 8 / 1024:>14.1f}K")
    print(f"\nDataset: Janus Sonar ({len(docs)} dialogue lines)")
    print(f"SPA: Sentence Phonon Attention (dim={n_embd})")
    print(f"Note: BitNet b1.58 shines at larger scales (3B+ params).")


if __name__ == '__main__':
    main()
