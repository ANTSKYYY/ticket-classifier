# Reading 1 — Neurons, Layers, Activations, and the Forward/Backward Pass

Comprehensive walkthrough of the four building blocks of every neural network, plus a fully worked backprop derivation.

---

# Neurons, Layers, Activations, and the Forward/Backward Pass

> **Module 2 Reading 1** — A from-scratch walkthrough of the four building blocks every deep-learning model is assembled from.

---

## 1. Why these four ideas matter

A neural network is just a function `f_θ(x)` parameterised by weights `θ`. Training a network is just solving an optimisation problem: find `θ` that minimises a loss `L(f_θ(x), y)` averaged over data. **The four ideas in this reading — neuron, layer, activation, and the forward/backward pass — are the entire mechanism behind every deep model**, from a 10-line MLP to a 200-billion-parameter Transformer. Everything you study later (CNNs, RNNs, attention, GANs) is *a special way of arranging neurons into layers and connecting their gradients*. Master this reading and you will never feel that a new architecture is "magic" — you will see only a different graph of the same primitives.

---

## 2. The artificial neuron

### 2.1 Biological inspiration (briefly)
The artificial neuron is a *very loose* abstraction of a biological neuron: dendrites receive signals, the soma integrates them, and if the integrated signal exceeds a threshold the neuron "fires". McCulloch & Pitts (1943) and Rosenblatt (1958) translated this into a mathematical object. Do not over-extend the analogy — modern neurons share almost nothing with biology beyond the cartoon.

### 2.2 The mathematical definition
A single neuron is the composition of **a linear transformation** and **a nonlinear activation**:

```
z = w · x + b           # linear combination (scalar)
a = φ(z)                # activation (scalar)
```

where:
- `x ∈ ℝ^n` is the input vector,
- `w ∈ ℝ^n` is the weight vector,
- `b ∈ ℝ` is the bias (scalar),
- `φ` is a non-linear activation function (defined in §4),
- `a ∈ ℝ` is the neuron's output.

The dot product `w · x` is geometrically a **projection of `x` onto direction `w`**, scaled by `‖w‖`. The bias `b` shifts the decision boundary off the origin. Without `b`, every hyperplane the neuron can represent passes through the origin — a crippling restriction.

### 2.3 Worked example
Suppose `x = [0.5, -1.2, 2.0]`, `w = [0.1, 0.4, -0.3]`, `b = 0.2`, and `φ = ReLU`.

```
z = 0.1·0.5 + 0.4·(-1.2) + (-0.3)·2.0 + 0.2
  = 0.05 − 0.48 − 0.60 + 0.20
  = −0.83
a = ReLU(−0.83) = 0
```

This neuron is silent on this input. Change `b` to `+1.0` and the neuron fires with `a = 0.17`. **The bias is what lets a neuron be selective.**

### 2.4 Common misconceptions
- "A neuron *stores* information." No — only the weights and bias do. The neuron is a computation, not a memory cell.
- "A neuron is intelligent." No — a single neuron with sigmoid is *exactly* logistic regression. The intelligence is emergent across many neurons.

---

## 3. Layers — composing neurons in parallel

### 3.1 Why we stack neurons
One neuron defines one hyperplane. Real problems need many hyperplanes (e.g., to carve up a curved decision boundary). We achieve this by running `m` neurons **on the same input** in parallel:

```
z = W x + b              # W ∈ ℝ^{m×n},  b ∈ ℝ^m,  x ∈ ℝ^n
a = φ(z)                 # elementwise activation, a ∈ ℝ^m
```

This is the canonical **fully-connected (dense) layer**. The row `W[i,:]` is the weight vector of the i-th neuron in this layer; `b[i]` is its bias.

### 3.2 Layer types you will meet in this course
| Layer | Linear operation | Notes |
|-------|------------------|-------|
| **Dense / Fully-connected** | `Wx + b` | Default building block; used in MLP heads. |
| **Convolutional** | Cross-correlation with a small kernel | Module 5 — exploits spatial locality. |
| **Recurrent** | `h_t = φ(W_h h_{t-1} + W_x x_t + b)` | Module 7 — shares weights across time. |
| **Attention** | `softmax(QKᵀ/√d) V` | Module 11 — content-based mixing. |
| **Normalisation** | `(x − μ) / σ · γ + β` | Module 12 — stabilises training. |
| **Dropout** | Random zero-out with prob. p | Module 12 — regularisation. |

All of them are still just *linear-then-nonlinear* in disguise. Only the linear part changes.

### 3.3 Depth — stacking layers in series
A *deep* network is a layer applied to the output of another:

```
h^(0) = x
h^(1) = φ(W^(1) h^(0) + b^(1))
h^(2) = φ(W^(2) h^(1) + b^(2))
...
ŷ     = φ(W^(L) h^(L−1) + b^(L))
```

We call `L` the **depth**. The intermediate vectors `h^(ℓ)` are **hidden representations**: each layer reshapes the input into a representation in which the next layer's job is easier. Empirically, depth lets a network represent *compositional* features cheaply: edges → textures → parts → objects.

### 3.4 The Universal Approximation Theorem (and what it doesn't say)
**A single hidden layer with enough neurons can approximate any continuous function on a compact set to arbitrary accuracy** (Cybenko 1989; Hornik 1991). This is *existence*, not *learnability*. In practice, depth is usually exponentially more parameter-efficient than width for the functions we actually care about (Telgarsky 2016). **Translation:** "wide enough" is a theorem; "deep enough" is the engineering reality.

---

## 4. Activation functions — where the nonlinearity lives

Without `φ`, a stack of dense layers collapses to a single linear map: `W^(L)·...·W^(1) x` is still linear. **The activation is the only thing that makes depth meaningful.**

### 4.1 The classic three

| Name | Formula | Range | Derivative |
|------|---------|-------|------------|
| **Sigmoid** | `1 / (1 + e^{−z})` | (0, 1) | `σ(z)(1−σ(z))` |
| **Tanh** | `(e^z − e^{−z}) / (e^z + e^{−z})` | (−1, 1) | `1 − tanh²(z)` |
| **ReLU** | `max(0, z)` | [0, ∞) | `1 if z > 0 else 0` |

### 4.2 ReLU and its descendants

- **ReLU** (Krizhevsky 2012): dead-simple, computationally cheap, gradient is exactly 1 on the positive side — no vanishing gradient. The downside is **dead neurons**: a unit can get stuck always-zero if its pre-activation never becomes positive again.
- **Leaky ReLU**: `max(αz, z)` with small `α` (e.g. 0.01) keeps a small gradient on the negative side.
- **GELU** (`z · Φ(z)`): smoother variant used in Transformers (Module 11).
- **Swish / SiLU** (`z · σ(z)`): popularised in modern vision and NLP backbones.

### 4.3 Picking an activation — quick rules
- Hidden layers in an MLP/CNN: **start with ReLU**.
- Hidden layers in a Transformer: **GELU** (it's what BERT/GPT use).
- Binary-classification output: **sigmoid** (followed by binary cross-entropy).
- Multi-class output: **softmax** (followed by categorical cross-entropy):
  ```
  softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
  ```
- Regression output: **identity** (no activation).
- *Never* put sigmoid or tanh in the *hidden* layers of a deep network — they cause vanishing gradients (Module 4 discusses this in detail).

### 4.4 Why nonlinearity is the secret sauce
Two linear layers `W_2 (W_1 x)` = `(W_2 W_1) x`, a single linear map. Sandwich a ReLU between them and the function becomes **piecewise linear with many pieces** — and that is exactly what we need to fold a space into a useful representation. Each ReLU draws a hyperplane and toggles a half-space on/off; stacking them defines arbitrarily complex polytopes.

---

## 5. The forward pass

The forward pass is the act of computing `ŷ = f_θ(x)` by evaluating the layers in order. Concretely:

```python
def forward(x, params):
    h = x
    for (W, b, phi) in params:        # layer-by-layer
        z = W @ h + b                 # linear
        h = phi(z)                    # nonlinear
    return h                          # ŷ
```

During the forward pass, every intermediate tensor (`z`, `h`) is **cached**, because the backward pass needs them.

### 5.1 The loss
After we have `ŷ`, we compare it to the target `y` using a **loss function** `L(ŷ, y)`. Two losses cover most of this course:

- **Mean squared error** (regression): `L = ½ ‖ŷ − y‖²`
- **Cross-entropy** (classification): `L = −Σ_i y_i log ŷ_i`

The loss is a *scalar*. Its job is to give us a single number to differentiate.

### 5.2 Why this is just function evaluation
Conceptually, the forward pass is no different from running `f(x) = sin(2x + 1)`. We are walking a computation graph that happens to have millions of nodes. The depth and width don't change the principle.

---

## 6. The backward pass (a.k.a. backpropagation)

### 6.1 The goal
We want `∂L / ∂θ` for every parameter `θ ∈ {W^(ℓ), b^(ℓ)}` so we can do gradient descent:

```
θ ← θ − η · ∂L/∂θ
```

For a deep network with millions of parameters, computing each gradient independently is hopeless. Backprop solves it by **reusing intermediate results** via the chain rule.

### 6.2 The chain rule, the only formula you need
For composed functions `L = h(g(f(x)))`:

```
dL/dx = (dL/dh) · (dh/dg) · (dg/df) · (df/dx)
```

Right-to-left, this is **forward-mode differentiation**. Left-to-right, this is **reverse-mode differentiation** — which is *exactly* backprop. Reverse mode is cheap when there are many inputs and one output, which is precisely our setting (millions of weights, one scalar loss).

### 6.3 Backprop on a 2-layer MLP — fully worked

Setup: `x ∈ ℝ^n`, weights `W_1 ∈ ℝ^{h×n}, W_2 ∈ ℝ^{1×h}`, biases `b_1, b_2`, ReLU hidden activation, MSE loss with target `y`.

**Forward:**
```
z_1 = W_1 x + b_1                   # ℝ^h
a_1 = ReLU(z_1)                     # ℝ^h
z_2 = W_2 a_1 + b_2                 # ℝ
ŷ   = z_2                           # identity output
L   = ½ (ŷ − y)²
```

**Backward (each step is one chain-rule application):**
```
∂L/∂ŷ   = (ŷ − y)                    # scalar
∂L/∂z_2 = ∂L/∂ŷ · 1     = (ŷ − y)
∂L/∂W_2 = ∂L/∂z_2 · a_1ᵀ            # ℝ^{1×h}
∂L/∂b_2 = ∂L/∂z_2                   # scalar
∂L/∂a_1 = W_2ᵀ · ∂L/∂z_2            # ℝ^h
∂L/∂z_1 = ∂L/∂a_1 ⊙ ReLU'(z_1)      # ℝ^h, ⊙ is elementwise
∂L/∂W_1 = ∂L/∂z_1 · xᵀ              # ℝ^{h×n}
∂L/∂b_1 = ∂L/∂z_1                   # ℝ^h
```

The whole backward pass uses *only matrix-vector products and one elementwise multiply*. The same algorithm scales to 100 layers.

### 6.4 In code (NumPy)
```python
# Forward
z1 = W1 @ x + b1
a1 = np.maximum(0, z1)
z2 = W2 @ a1 + b2
y_hat = z2
loss = 0.5 * (y_hat - y) ** 2

# Backward
dz2 = y_hat - y
dW2 = dz2 * a1.T          # outer product
db2 = dz2
da1 = W2.T * dz2
dz1 = da1 * (z1 > 0)
dW1 = np.outer(dz1, x)
db1 = dz1
```

`torch.autograd` will do this for you automatically (Module 4 lab), but **you must be able to do it by hand** to debug numerical issues, custom layers, and exotic losses.

### 6.5 Computation-graph view
Modern frameworks (PyTorch, JAX, TensorFlow) treat every tensor operation as a node in a **directed acyclic graph (DAG)**. The forward pass builds the DAG; the backward pass walks it in reverse topological order, accumulating local gradients via the chain rule. This is **reverse-mode automatic differentiation**, and it is the engine of every deep-learning library you will use this term.

### 6.6 The pitfalls of backprop
- **Vanishing gradients**: when activations saturate (sigmoid in deep stacks), `φ'(z) ≈ 0`, and gradient signals die. Cure: ReLU/GELU + careful init + skip connections.
- **Exploding gradients**: gradients grow without bound in deep RNNs or unnormalised nets. Cure: gradient clipping, layer normalisation.
- **Numerical instability in softmax + cross-entropy**: always use the framework's fused `log_softmax` / `cross_entropy_with_logits` instead of computing them separately.

---

## 7. Putting it all together — one training step

```python
for batch_x, batch_y in dataloader:
    y_hat = forward(batch_x, params)              # §5
    loss = loss_fn(y_hat, batch_y)                # §5.1
    grads = backward(loss, params)                # §6
    params = [(W - η*dW, b - η*db) for ...]       # SGD update
```

That is the inner loop of essentially every supervised deep-learning algorithm. Every later innovation (Adam, batch norm, dropout, residual connections, attention) is *a refinement of this loop, not a replacement.*

---

## 8. Key takeaways
1. A neuron is `φ(w·x + b)`. A layer is many neurons in parallel. A network is many layers in series.
2. The activation is the only nonlinear ingredient — strip it out and depth becomes meaningless.
3. The forward pass evaluates the network and caches every intermediate tensor.
4. The backward pass is reverse-mode automatic differentiation: one walk of the computation graph, applying the chain rule.
5. Backprop is *not* a separate algorithm — it is just the chain rule applied to a DAG.
6. Vanishing/exploding gradients, dead ReLUs, and unstable softmax are the practical hazards you must recognise.

---

## 9. Recommended resources
- Goodfellow, Bengio, Courville. *Deep Learning*. Chapter 6 (feedforward networks), Chapter 8.1 (gradient descent), free online: deeplearningbook.org.
- Karpathy. *Yes you should understand backprop.* Medium post.
- 3Blue1Brown. *But what is a neural network?* (4-part YouTube series — best visual intuition you can get in an hour).
- Olah. *Calculus on Computational Graphs: Backpropagation.* colah.github.io.
- Nielsen. *Neural Networks and Deep Learning*, Chapter 2 — derives backprop step-by-step.

> When in doubt, derive backprop by hand on a 2-layer MLP. Most "I don't understand X" moments in deep learning dissolve once that derivation is automatic.