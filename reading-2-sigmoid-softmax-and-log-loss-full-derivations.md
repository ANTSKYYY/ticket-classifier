# Reading 2 — Sigmoid, Softmax, and Log-Loss: Full Derivations

Mathematical reference deriving sigmoid and softmax (with Jacobians), binary and categorical cross-entropy (with MLE interpretation), numerical-stability tricks, and the (a − y) gradient identity.

---

# Sigmoid, Softmax, and Log-Loss — Full Derivations

> **Module 3 Reading 2** — The mathematical reference for the three functions you will use almost every day this term. Each derivation is from scratch, with the numerical-stability tricks and the connections back to logistic regression (Reading 1) and forward to Modules 4 (deep MLP), 11 (Transformer output heads), and 12 (label smoothing).

This is a "look-up reading" — read it once carefully, then come back any time you need a derivation.

---

## 1. Why bother deriving them?

`sigmoid`, `softmax`, and `log-loss` (cross-entropy) are three lines of PyTorch you will paste hundreds of times. So why derive them by hand?

Because three of the most common bugs in early deep-learning practice come from *not* understanding what is inside those three functions:

1. Applying softmax *then* `cross_entropy` (double softmax → broken gradient).
2. Computing `log(sigmoid(z))` directly when `z` is large negative (numerical overflow / underflow).
3. Picking sigmoid + categorical-cross-entropy when you should have picked softmax + categorical-cross-entropy (or vice versa).

The derivations below also unify with one of the most surprising results in classical ML: the gradient of cross-entropy through softmax (and sigmoid) is always `ŷ − y`. Knowing *why* — i.e., that the link function and the loss are *conjugate* — separates a senior practitioner from a junior one.

---

## 2. Notation

Throughout this reading:

| Symbol | Meaning |
|--------|---------|
| `z ∈ ℝ` | pre-activation scalar (a single neuron output before activation) |
| `z ∈ ℝ^K` | pre-activation vector (K-class output before softmax) |
| `a, ŷ` | post-activation (probability) — scalar in binary, vector in multi-class |
| `y` | true label — scalar (0/1) in binary, one-hot vector in multi-class |
| `K` | number of classes |
| `m` | number of training examples |
| `σ(·)` | sigmoid |
| `softmax(·)` | softmax |
| `L` | scalar loss for one example |
| `J` | mean loss over the batch |

---

## 3. Sigmoid

### 3.1 Definition
The (logistic) sigmoid is:

```
σ(z) = 1 / (1 + e^{−z})
```

**Properties at a glance:**
- Range: `(0, 1)` — useful as a probability.
- `σ(0) = 0.5` (calibrated middle).
- Saturates: `σ(z) → 1` as `z → +∞`, `σ(z) → 0` as `z → −∞`.
- Smooth and differentiable everywhere.
- Mirror symmetric about the origin: `σ(−z) = 1 − σ(z)`.

### 3.2 The derivative — a derivation you should memorise

We want `dσ/dz`. Differentiate:

```
σ(z) = 1 / (1 + e^{−z}) = (1 + e^{−z})^{−1}

dσ/dz = −1 · (1 + e^{−z})^{−2} · d/dz (1 + e^{−z})
      = −1 · (1 + e^{−z})^{−2} · (−e^{−z})
      = e^{−z} / (1 + e^{−z})^2
```

Now write `e^{−z} = (1 + e^{−z}) − 1`:

```
dσ/dz = [(1 + e^{−z}) − 1] / (1 + e^{−z})^2
      = 1/(1 + e^{−z})  −  1/(1 + e^{−z})^2
      = σ(z) − σ(z)^2
      = σ(z) (1 − σ(z))
```

**Key result.** `σ'(z) = σ(z) · (1 − σ(z))`. This is what makes backprop through a sigmoid neuron a single multiplication.

### 3.3 Why sigmoid saturates — and what that costs deep networks

The maximum of `σ'(z)` is at `z = 0`, where it equals `0.25`. Far from `z = 0`, `σ'(z) → 0`. **Stacked sigmoids in a deep network multiply many small numbers together** → the gradient signal vanishes. This is the **vanishing-gradient problem** and the reason Module 4 will tell you not to put sigmoid in hidden layers.

```
Gradient magnitude through a deep stack:
   |∂L/∂w_1|  ≈  |σ'(z_L)| · |σ'(z_{L-1})| · … · |σ'(z_1)|  ≤  0.25^L
For L = 10, this is ≤ 9.5e-7.
```

That is why ReLU exists.

### 3.4 Numerically stable sigmoid

`σ(z) = 1 / (1 + e^{−z})` overflows when `z` is very negative (`e^{−z}` blows up). The standard fix:

```python
def stable_sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    # for z >= 0: σ(z) = 1 / (1 + exp(-z))  — exp small, safe
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    # for z < 0:  σ(z) = exp(z) / (1 + exp(z))  — exp small, safe
    e = np.exp(z[neg])
    out[neg] = e / (1.0 + e)
    return out
```

In PyTorch, `torch.sigmoid` already does this; you only need to know about it when you write CUDA kernels or NumPy code.

### 3.5 The cleanest fact about sigmoid

`σ(z)` is the **logistic link function**: it is the inverse of the *log-odds*, `logit(p) = log(p / (1 − p))`. Logistic regression is "linear regression on the log-odds" — exactly the line `z = w · x + b` interpreted as the log-odds of `y = 1`. You will need this idea to reason about calibration in Module 12.

---

## 4. Log-loss (binary cross-entropy)

### 4.1 Definition
The per-example log-loss (a.k.a. binary cross-entropy) is:

```
L(a, y) = − [ y · log a  +  (1 − y) · log(1 − a) ]
```

For a batch:

```
J = (1/m) Σ_i L(a^{(i)}, y^{(i)})
```

### 4.2 Where it comes from — maximum likelihood

If we model `y ∈ {0, 1}` as Bernoulli with parameter `a`:

```
P(y | a) = a^y · (1 − a)^{1 − y}
```

Negative log-likelihood for one example:

```
−log P(y | a) = − [ y log a + (1 − y) log(1 − a) ]
```

This is *exactly* `L(a, y)`. **Minimising binary cross-entropy = maximising likelihood under a Bernoulli model.** Cross-entropy is not an arbitrary choice — it is the natural loss for binary classification.

### 4.3 Why not MSE?

For sigmoid output + MSE loss, the loss surface is **non-convex** in `w` and **plateaus** in the saturated regions (where `σ'(z) ≈ 0`). For sigmoid + BCE, the loss is **convex** in `w` (for a single linear layer) and has a much steeper gradient when the model is confidently wrong. Both properties make optimisation dramatically easier.

A short empirical demonstration: train logistic regression on MNIST 4-vs-9 with MSE vs. BCE — BCE converges in 1/5 the iterations and reaches lower error.

### 4.4 The derivative — the most-quoted equation in early deep learning

We want `∂L/∂z`. By the chain rule:

```
∂L/∂a = − [ y/a − (1 − y) / (1 − a) ]
      = (a − y) / [ a (1 − a) ]                  ← combine over common denominator
∂a/∂z = a (1 − a)                                ← from §3.2
∂L/∂z = (∂L/∂a) · (∂a/∂z) = (a − y) / [a(1−a)] · a(1−a)
      = a − y
```

**Result.** `∂L/∂z = a − y`. The `a(1 − a)` cancels — *that* is why sigmoid + BCE is the conjugate pair.

The downstream weight gradients (single example):

```
∂L/∂w = (a − y) · x
∂L/∂b = (a − y)
```

For a batch (vectorised, with `X ∈ ℝ^{m×n}`, `a, y ∈ ℝ^m`):

```
∂J/∂w = (1/m) · Xᵀ (a − y)
∂J/∂b = (1/m) · 1ᵀ (a − y)
```

You will paste this into Lab 3.

### 4.5 Numerically stable log-loss

Three places to lose precision:

1. **`log(0)`** — when `a = 0` or `a = 1` exactly. Clamp `a` to `[ε, 1 − ε]` with `ε = 1e-7`, *or* use a fused implementation.
2. **`log(sigmoid(z))`** — for large negative `z`. Use the identity:
   ```
   log σ(z) = − softplus(−z) = − log(1 + e^{−z})
   ```
   `softplus` has a numerically stable implementation:
   ```
   softplus(x) = max(x, 0) + log(1 + exp(−|x|))
   ```
3. **Fused `BCEWithLogitsLoss`** — PyTorch's `nn.BCEWithLogitsLoss(logits, y)` combines sigmoid + log-loss in one numerically stable op. **Use it.** Do not pass `sigmoid(z)` to `BCELoss` — that is the unstable path.

```python
# GOOD
loss = F.binary_cross_entropy_with_logits(logits, y)

# AVOID
loss = F.binary_cross_entropy(torch.sigmoid(logits), y)
```

### 4.6 Properties of binary cross-entropy

- Always ≥ 0; zero only when `a = y` exactly.
- Convex in `w` for a linear model + sigmoid + BCE (this is the optimisation reason it's loved).
- Punishes confident-wrong predictions disproportionately (recall `log 0 = −∞`).
- Has a probabilistic interpretation as KL-divergence: `BCE(a, y) = KL(y || a)` for Bernoulli `y`.

---

## 5. Softmax

### 5.1 Definition (K classes)

For a pre-activation vector `z ∈ ℝ^K`:

```
softmax(z)_i = e^{z_i} / Σ_{j=1}^{K} e^{z_j}
```

The output is a probability distribution over the K classes: each `a_i ∈ (0, 1)`, `Σ_i a_i = 1`.

### 5.2 Sigmoid as a special case

For `K = 2`, softmax reduces to sigmoid (up to a re-parameterisation):

```
softmax([z_1, z_2])_1 = e^{z_1} / (e^{z_1} + e^{z_2})
                      = 1 / (1 + e^{z_2 − z_1})
                      = σ(z_1 − z_2)
```

So **sigmoid is the K=2 special case of softmax**. In practice we still write the binary case as sigmoid because it uses one output unit instead of two.

### 5.3 Why subtract the max — numerical stability

For large `z_i`, `e^{z_i}` overflows. The fix: softmax is **shift-invariant** — subtracting a constant from every entry does not change the output:

```
softmax(z)_i = e^{z_i − c} / Σ_j e^{z_j − c}     for any constant c
```

Standard choice: `c = max_j z_j`. Now the largest exponent is `e^0 = 1`, and everything stays in the safe range.

```python
def stable_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
```

PyTorch's `torch.softmax` already does this.

### 5.4 The Jacobian of softmax

For `a = softmax(z)`, the Jacobian `∂a_i / ∂z_k` is:

```
∂a_i / ∂z_k = a_i (δ_{ik} − a_k)
```

where `δ_{ik}` is the Kronecker delta (1 if `i = k`, else 0). Equivalently:

```
∂a_i / ∂z_i = a_i (1 − a_i)              ← when k = i, this echoes sigmoid's derivative
∂a_i / ∂z_k = − a_i · a_k                 ← when k ≠ i
```

The matrix form: if `a` is the column vector of softmax outputs, the Jacobian is `diag(a) − a aᵀ`. Beautifully compact.

You almost never need this in code — because (as in §6.4 below) the categorical-cross-entropy + softmax composition collapses *the entire Jacobian* into a single `ŷ − y` term. But knowing the Jacobian exists matters for Module 11 (attention) and Module 12 (label smoothing).

---

## 6. Categorical cross-entropy + softmax

### 6.1 Definition

With one-hot label `y ∈ {0, 1}^K` (exactly one entry = 1) and softmax output `a ∈ (0, 1)^K`:

```
L(a, y) = − Σ_i y_i log a_i
```

Since `y` is one-hot, only the *correct-class* term survives:

```
L = − log a_{y*}                where y* is the correct class index
```

For a batch:

```
J = − (1/m) Σ_n log a^{(n)}_{y*^{(n)}}
```

This is the loss used in 90 % of the classification problems in this course.

### 6.2 As maximum likelihood

`a` is the model's posterior `P(class = i | x)`. Maximising the joint likelihood over `m` examples = minimising the average of `− log a_{y*}`, exactly the loss above. Cross-entropy is, again, *the natural loss*.

### 6.3 The most elegant derivation in deep learning — `∂L/∂z = a − y`

We want `∂L/∂z_k`. Two cases:

**(a) k = y\* (the correct class):**

```
L = − log a_{y*}
∂L/∂a_{y*} = − 1 / a_{y*}
∂a_{y*}/∂z_{y*} = a_{y*} (1 − a_{y*})           ← from softmax Jacobian
∂L/∂z_{y*} = − 1/a_{y*} · a_{y*} (1 − a_{y*})
           = − (1 − a_{y*})
           = a_{y*} − 1
```

**(b) k ≠ y\* (any other class):**

```
∂L/∂a_{y*} = − 1/a_{y*}
∂a_{y*}/∂z_k = − a_{y*} · a_k                    ← softmax Jacobian, k ≠ i
∂L/∂z_k = − 1/a_{y*} · (− a_{y*} · a_k)
        = a_k
```

Combining both:

```
∂L/∂z_k = a_k − y_k    for every k
```

where `y` is one-hot. Vectorially:

```
∂L/∂z = a − y                    ← exactly the binary-case result, generalised
```

**This is the single most important fact about cross-entropy + softmax.** It is why backprop through the output layer is one subtraction.

### 6.4 Downstream gradients

For a softmax output layer on top of a `W ∈ ℝ^{K×h}` and bias `b ∈ ℝ^K` over a hidden activation `h_prev ∈ ℝ^h`:

```
∂L/∂W = (a − y) · h_prevᵀ          # outer product, ℝ^{K×h}
∂L/∂b = (a − y)
∂L/∂h_prev = Wᵀ (a − y)            # propagated to the previous layer
```

This pattern appears at the output of every classifier we train in DL2026.

### 6.5 Numerically stable categorical CE

The same warning as binary: never compute `log(softmax(z))` directly. Use **log-softmax**:

```
log_softmax(z)_i = z_i − logsumexp(z)
```

with a numerically stable `logsumexp`:

```
logsumexp(z) = c + log Σ_j e^{z_j − c}        with  c = max z
```

PyTorch's `F.cross_entropy(logits, y_class)` is the fused, stable implementation — it expects:
- **raw logits** (not softmax outputs),
- **integer class indices** (not one-hot vectors).

```python
# GOOD
loss = F.cross_entropy(logits, target)              # target = (B,) integer indices

# AVOID
loss = -(target_one_hot * torch.log(torch.softmax(logits, -1))).sum(-1).mean()
```

### 6.6 Properties of categorical cross-entropy

- Always ≥ 0; zero only when the model is confident and correct.
- Convex (with respect to the logits) for a linear classifier; non-convex once a hidden layer is added (Module 4).
- Equivalent to minimising the KL-divergence between the one-hot label and the model distribution.
- The basis for **label smoothing** (Module 12): replace `y` with `(1 − ε) y + ε / K · 1`. This trades a tiny amount of accuracy for substantially better calibration.

---

## 7. Sigmoid vs. softmax — quick decision rules

| Use sigmoid + binary cross-entropy when… | Use softmax + categorical cross-entropy when… |
|------------------------------------------|----------------------------------------------|
| Output is a single binary label (cat / not-cat). | Output is one of K mutually exclusive classes (cat / dog / horse). |
| Output is *multi-label* (each label independent: "image contains cat", "image contains dog", both can be true). Use one sigmoid per label. | Labels are mutually exclusive (exactly one is correct). |
| K = 2 and you only want one output unit. | K ≥ 2 and you want a calibrated probability per class. |

**Common mistake:** softmax on a multi-label task (the softmax forces the probabilities to sum to 1, but the true labels may have multiple 1s).

---

## 8. Connections to the rest of DL2026

| Concept | Where it returns |
|---------|------------------|
| `∂L/∂z = a − y` | The output-layer gradient of every classifier in this course. |
| Sigmoid saturation | Module 4: vanishing gradients in deep MLPs. |
| Softmax Jacobian | Module 11: attention weights are *softmax outputs*; their Jacobian matters for gradient flow in Transformers. |
| Cross-entropy as MLE | Module 8: VAE ELBO has a cross-entropy reconstruction term. |
| Label smoothing | Module 12: a regulariser that trades calibration for accuracy. |

---

## 9. Worked NumPy reference — copy into your notebook

```python
import numpy as np

# Sigmoid (numerically stable)
def sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[neg]); out[neg] = e / (1.0 + e)
    return out

# Softmax (numerically stable)
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

# Binary cross-entropy (with logits — preferred)
def bce_with_logits(z, y, eps=1e-12):
    # − log σ(z) when y=1, − log(1−σ(z)) when y=0; stable form:
    #   max(z, 0) − z*y + log(1 + exp(−|z|))
    return (np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))).mean()

# Categorical cross-entropy (with logits)
def ce_with_logits(z, y_idx):                 # y_idx: integer class labels
    log_probs = z - np.log(np.exp(z - z.max(-1, keepdims=True)).sum(-1, keepdims=True))
    return -log_probs[np.arange(len(y_idx)), y_idx].mean()
```

Keep this snippet — Lab 3 expects from-scratch versions of these three functions.

---

## 10. Key takeaways

1. **Sigmoid** maps real numbers to (0, 1); its derivative is `σ(1 − σ)`. It saturates at the tails, which causes vanishing gradients in deep stacks.
2. **Softmax** is the K-class generalisation of sigmoid; its Jacobian is `diag(a) − a aᵀ`. Subtract the max before exponentiating.
3. **Binary cross-entropy** is the negative log-likelihood of a Bernoulli model. Combined with sigmoid, the loss gradient at the pre-activation is `a − y`.
4. **Categorical cross-entropy** is the negative log-likelihood of a categorical model. Combined with softmax, the loss gradient at the pre-activation is again `a − y` — the most elegant identity in early DL.
5. **Always use the fused log-sigmoid / log-softmax / cross-entropy implementations** (`BCEWithLogitsLoss`, `CrossEntropyLoss`); don't combine the pieces manually.
6. The vanishing-gradient problem of sigmoid is the practical reason ReLU exists. Save sigmoid for output layers.

---

## 11. Recommended resources

- Goodfellow, Bengio, Courville. *Deep Learning*, §6.2.2 (output units), §6.2.1 (cost functions).
- Bishop. *Pattern Recognition and Machine Learning*, §4.3 (logistic regression), §4.3.4 (softmax).
- Michael Nielsen, *Neural Networks and Deep Learning*, Ch. 3 (cross-entropy).
- Olah. *Calculus on Computational Graphs: Backpropagation.* colah.github.io.
- PyTorch docs: `F.binary_cross_entropy_with_logits`, `F.cross_entropy`, `torch.softmax`.

> If you can produce the derivation of `∂L/∂z = a − y` for both sigmoid + BCE and softmax + categorical CE on a whiteboard from memory, you have absorbed Module 3.