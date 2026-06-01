# Reading 1 — Companion Notes for Goodfellow §6.3–6.5 (Hidden Units, Architecture, Backprop)

Structured companion to Goodfellow Ch. 6 §§6.3–6.5: activations (ReLU/sigmoid/tanh/GELU), initialisation (Xavier/Kaiming), depth vs. width, and reverse-mode backprop. Recast in DL2026 notation with study questions and forward references.

---

# Companion Notes — Goodfellow et al., *Deep Learning*, Chapter 6 §6.3–6.5

> **Module 4 Reading 1** — A guided study companion for **§6.3 (Hidden Units)**, **§6.4 (Architecture Design)**, and **§6.5 (Back-Propagation and Other Differentiation Algorithms)** of Goodfellow, Bengio, and Courville's *Deep Learning* (MIT Press, 2016). The book is freely available at **https://www.deeplearningbook.org**. These notes are a structured *second pass*, not a substitute — read the original first.

---

## 0. How to use this reading

1. **Open the original** at deeplearningbook.org → Chapter 6. The three sections cover roughly pages 168–223. They are dense — budget ~3 hours for the first read, with pen and paper.
2. **Then come back to these notes.** They paraphrase and re-organise the chapter in DL2026 notation, point out where the textbook is now slightly dated (e.g., on activations), and connect each idea to a later module.
3. **Answer the study questions in §10** before Session 4 lab.

Throughout, equations use the DL2026 convention (vectors lowercase, matrices uppercase, batch dimension first — the same as Module 2 Reading 1 and Module 3 Readings 1–2).

---

## 1. Why these three sections matter

§6.3, §6.4, and §6.5 together answer the three engineering questions you will face every time you sit down to design a deep feedforward network:

1. **What activation should I use?** → §6.3
2. **How deep / how wide should the network be?** → §6.4
3. **How are the gradients actually computed?** → §6.5

The rest of the course is variations on these answers. CNNs (Module 5), RNNs (Module 7), Transformers (Module 11) — each one redefines what counts as a "layer", but the *activation, depth, and gradient* decisions are still made by you, the engineer.

---

## 2. §6.3 — Hidden Units (activations)

### 2.1 The role of an activation
A hidden unit is `a = φ(W h_prev + b)` for some non-linear `φ`. Without `φ`, a stack of `L` linear layers collapses to **one** linear map: `W^(L) ... W^(1) x = (W^(L) … W^(1)) · x`, which has no more expressive power than logistic regression. **Every gain from depth in this course comes from the non-linearity between layers.**

Goodfellow's framing: an activation introduces non-linear *features* that the next layer can exploit. The choice of `φ` shapes:
- the *expressive power* of the network,
- the *gradient flow* (saturation behaviour, dead-unit risk),
- the *numerical conditioning* (range of outputs, magnitude of derivatives).

### 2.2 The activations the textbook treats in detail

#### 2.2.1 Rectified linear unit (ReLU) — the modern default
```
ReLU(z) = max(0, z)
ReLU'(z) = 1 if z > 0 else 0
```
**Why ReLU dominates** (Goodfellow §6.3.1):
- Computationally trivial — one comparison per element.
- Gradient is exactly 1 on the active side — no vanishing for positive activations.
- Easy to *initialise* (see §2.3 below).
- Empirically beats sigmoid/tanh in deep networks by large margins (proven by AlexNet, 2012).

**The cost** — the **dying-ReLU** problem. A unit whose pre-activation `z` is negative for *every* training example gets gradient 0 forever and never recovers. Cures:
- **Leaky ReLU**: `max(αz, z)` with `α ∈ {0.01, 0.1}`.
- **Parametric ReLU (PReLU)**: learn `α` per unit.
- **ELU**: `z if z > 0 else α(exp(z) − 1)` — smooth and negative-allowed.
- Good initialisation + careful learning rate often suffice (Modules 12 lab demonstrates).

#### 2.2.2 Sigmoid and tanh — what the textbook calls "softer" alternatives
```
σ(z)   = 1 / (1 + e^{−z})        # output ∈ (0, 1)
tanh(z) = 2σ(2z) − 1              # output ∈ (−1, 1)
```
The textbook treats these for completeness but is candid about their pitfalls in deep networks:
- Both saturate; their derivatives shrink toward 0 for large `|z|`.
- Multiplied through L layers, the gradient vanishes (Reading 2 §3.3 of Module 3 worked this out).
- Tanh is preferred to sigmoid when used at all, because it is **zero-centred** — but ReLU is preferred to both.
- **Where they survive**: output layers (sigmoid for binary, softmax for multi-class), gating in LSTMs/GRUs (Module 7), and attention weights (Module 11).

#### 2.2.3 Other functions worth knowing (and where the textbook is now dated)
The 2016 edition pre-dates a few activations now standard:
- **GELU** (Hendrycks & Gimpel, 2016) — used in BERT and GPT. Smooth approximation: `z · Φ(z)`.
- **Swish / SiLU** (Ramachandran et al., 2017) — `z · σ(z)`. Used in EfficientNet, modern vision backbones.
- **GLU and gated variants** (Dauphin et al., 2017) — used in Transformer feed-forward blocks.

In Module 11 you will see GELU in production for the first time.

### 2.3 Initialisation — the cure for "deep networks just don't train"
Goodfellow §6.3 ends with a long discussion of why naïve initialisation breaks deep nets. The short version:

- **All-zero init**: every unit computes the same thing → no symmetry breaking → no learning.
- **Too-large init**: pre-activations explode; for sigmoid/tanh, units saturate; for ReLU, exploding gradients.
- **Too-small init**: signal dies as it propagates; gradients vanish.

The two named "good" inits:
- **Xavier / Glorot** (for sigmoid/tanh): `var(W) = 1 / fan_in` (or the symmetric average of fan_in and fan_out).
- **He / Kaiming** (for ReLU): `var(W) = 2 / fan_in` — accounts for ReLU killing half the variance.

**Rule of thumb you will paste 1000 times this term:** *use Kaiming init for ReLU networks, Xavier for tanh/sigmoid, the framework default for nearly everything else.*

PyTorch's `nn.Linear` already initialises sensibly; you only need to override when you do something exotic (Module 12 demonstrates an explicit override).

### 2.4 Quick reference table

| Activation | Range | `φ'` peak | Vanishing risk | Dead-unit risk | Modern use |
|------------|-------|-----------|----------------|----------------|------------|
| Sigmoid    | (0,1) | 0.25      | High           | No             | Binary output, gating |
| Tanh       | (−1,1)| 1.00      | Medium         | No             | RNN cells |
| ReLU       | [0,∞) | 1 (a.s.)  | Low            | **Yes**        | Default hidden |
| Leaky ReLU | ℝ     | 1 (a.s.)  | Low            | No             | Hidden, deep CNNs |
| GELU       | ≈ ℝ   | ~1.13     | Low            | No             | Transformers |
| Swish/SiLU | ≈ ℝ   | ~1.10     | Low            | No             | Modern vision |

---

## 3. §6.4 — Architecture Design

### 3.1 The Universal Approximation Theorem — what it says, what it doesn't
**Statement** (Cybenko 1989; Hornik 1991): a single-hidden-layer feedforward network with a "squashing" activation (sigmoid, tanh) and enough hidden units can approximate any continuous function on a compact subset of `ℝ^n` to arbitrary accuracy.

**What this is NOT:**
- It is *existence*, not *learnability*. The theorem promises a `W, b` exists; it doesn't promise gradient descent finds it.
- It is not a *constructive* result. The required width can be astronomical.
- It does not say *shallow is enough in practice*. The opposite is true:

### 3.2 Depth vs. width — Goodfellow's empirical case
A core message of §6.4: **depth is exponentially more parameter-efficient than width for most functions of practical interest.**

- Telgarsky (2016) — showed there exist functions requiring `O(2^L)` width at depth `L'<L`, but `O(L)` parameters at depth `L`.
- Empirical scaling: GPT-3 didn't get to 175 B parameters by widening 3 layers; it got there by stacking 96.

The trade-off is real, however:
| If you go too shallow | If you go too deep |
|-----------------------|---------------------|
| Need exponentially many neurons. | Vanishing / exploding gradients. |
| Training is fast but accuracy plateaus. | Need normalisation, skip connections, careful init. |
| Easy to interpret. | Easier to overfit; opaque. |

### 3.3 Practical guidance from §6.4
Goodfellow distils architecture choice into three rules:
1. **Pick a known-working family for your data**: CNNs for grids, RNNs/Transformers for sequences, MLPs for tabular. Don't reinvent.
2. **Start small.** Most projects in this course should start with the smallest model the problem plausibly needs. Lab 2 already showed 256-wide MLP > 95 % on MNIST.
3. **Scale up via depth before width**, holding the parameter budget fixed.

You will see this in your project proposal review — every cohort, two-thirds of teams over-spec their first architecture. Don't be those teams.

### 3.4 Other architecture knobs the chapter touches
- **Skip / residual connections** (introduced by Goodfellow but expanded by ResNet, He et al. 2015) — `h^(ℓ+1) = h^(ℓ) + F(h^(ℓ))`. The cure for deep-network optimisation.
- **Layer normalisation / batch normalisation** (Module 12) — discussed briefly here, fully in Goodfellow Ch. 8.
- **Parameter sharing** (CNNs, RNNs) — Modules 5 and 7.

---

## 4. §6.5 — Back-Propagation

§6.5 is the longest section of Chapter 6. Goodfellow's pedagogical move is to *separate three ideas* that practitioners often conflate:

| Idea | What it is |
|------|------------|
| **Forward pass** | Compute the network's output by evaluating the computation graph node-by-node. |
| **Backward pass / backprop** | Compute `∂L/∂θ` by reverse-mode automatic differentiation. |
| **Optimisation (e.g., SGD)** | *Use* the gradients to update parameters. |

People say "backprop" when they often mean "training". The textbook is strict: **backprop is the gradient computation only**; optimisation is a separate step.

### 4.1 The chain rule — one formula, many uses
For `L = h(g(f(x)))`:
```
dL/dx = (dL/dh) · (dh/dg) · (dg/df) · (df/dx)
```
- **Forward-mode AD** evaluates right-to-left: cheap when *few inputs, many outputs*.
- **Reverse-mode AD** evaluates left-to-right: cheap when *many inputs, one output*.

Deep learning has **one scalar loss** and **millions of parameters** — reverse mode wins by orders of magnitude. That is backprop.

### 4.2 The computation graph
Every tensor op (`@`, `+`, `relu`, `softmax`, ...) is a node. Edges encode "operand of" relationships. Forward pass builds the graph and caches the values; backward pass walks the same graph in reverse topological order, applying the chain rule and accumulating partial derivatives into each parameter's `.grad` slot.

PyTorch's `autograd`, TensorFlow's `GradientTape`, and JAX's `jit`-traced VJP are all implementations of this idea. **Reverse-mode AD is the engine of every framework you use this term.**

### 4.3 The algorithm in skeleton form
The chapter pseudo-code, paraphrased:

```
forward(x):
    for layer in network.layers:
        h = layer.forward(h, cache=True)    # store inputs needed for backward
    return h

backward(loss):
    grad = 1
    for layer in reversed(network.layers):
        grad = layer.backward(grad)         # local Jacobian × upstream grad
        accumulate(grad into layer.parameters)
```

Each layer's `.backward` only needs:
- the **upstream gradient** (`grad` from the layer above),
- the **cached forward inputs** (to evaluate the local Jacobian),
- the **chain-rule combination** (one matrix-vector multiply for `Linear`; element-wise multiply for activations).

### 4.4 Worked example — backprop on a 2-layer MLP (the canonical exercise)
You did this in Reading 1 of Module 2. Re-do it now from memory:

```
z_1 = W_1 x + b_1          # (h,)
a_1 = ReLU(z_1)
z_2 = W_2 a_1 + b_2        # (K,)
a_2 = softmax(z_2)
L   = cross_entropy(a_2, y)

# Backward
dz_2 = a_2 - y                          # output gradient (Module 3 Reading 2 §6)
dW_2 = dz_2 · a_1.T
db_2 = dz_2
da_1 = W_2.T · dz_2
dz_1 = da_1 ⊙ ReLU'(z_1)                # element-wise; 1 where z_1 > 0
dW_1 = dz_1 · x.T
db_1 = dz_1
```

Every later network in DL2026 is a **bigger version of this exact computation**. If you can derive this on a whiteboard cold, you can debug any deeper network.

### 4.5 Pitfalls the chapter warns about
- **Vanishing gradients** — sigmoid + depth + naive init. Cure: ReLU + Kaiming init + skip connections.
- **Exploding gradients** — deep RNNs or unnormalised nets. Cure: gradient clipping (Module 7).
- **Numerical instability** — `log(softmax(z))` blows up. Use fused `log_softmax` (Module 3 Reading 2 §5.5).
- **Cache memory** — backward needs every intermediate tensor. Long sequences and big batches blow GPU memory. Cure: gradient *checkpointing* (Module 13).
- **Higher-order derivatives** — for second-order methods, you backprop *through* the backward pass. PyTorch supports it via `create_graph=True`; rarely used in this course.

### 4.6 Where automatic differentiation came from
A short historical aside (textbook §6.5.10): reverse-mode AD predates deep learning by ~40 years (Linnainmaa 1970, Werbos 1974, Rumelhart-Hinton-Williams 1986 in the connectionist context). What changed in 2010s is *frameworks*: PyTorch, TensorFlow, JAX let you express the model in plain code and get the gradient for free. **You are inheriting a polished tool — but you should still be able to derive it by hand.**

---

## 5. The "compute-then-derive" mental model

A useful crystallisation of §6.5: every layer you ever write should support **four operations**:
1. **Forward** — given inputs, produce outputs.
2. **Cache** — store whatever the backward needs.
3. **Backward (input grad)** — given upstream grad, produce the gradient with respect to inputs.
4. **Backward (parameter grad)** — given upstream grad, produce the gradient with respect to parameters.

PyTorch's `nn.Module` automates 2–4 via autograd, but when you write a *custom* op (Module 12 lab) or debug an autograd graph (Module 11), you'll define them by hand. Learn the structure now.

---

## 6. Where each idea is picked up later in DL2026

| Section | Returns in |
|---------|------------|
| §6.3 activations | Module 11 (GELU in Transformers), Module 12 (regularisation interactions). |
| §6.3 initialisation | Lab 4 (init ablation), Module 12 (BatchNorm interaction). |
| §6.4 depth vs. width | Module 5 (ResNet skip connections), Module 11 (depth of Transformer stacks). |
| §6.5 backprop | Every later module. The single most foundational idea in the field. |
| Reverse-mode AD | Module 13 (mixed precision, gradient checkpointing, DDP). |

---

## 7. Connections to Modules 2 and 3

This chapter is the natural sequel to:
- **Module 2 Reading 1** (Neurons, layers, activations, forward/backward pass) — §6.3 turns "use a non-linearity" into a menu, and §6.5 turns the cartoon backprop into the formal algorithm.
- **Module 3 Reading 1** (Logistic regression as a NN) — §6.5 generalises the `(a − y)` identity to arbitrary computation graphs.
- **Module 3 Reading 2** (Sigmoid, softmax, log-loss derivations) — §6.3 explains why sigmoid is *not* a hidden activation in modern nets, even though it remains the right output activation for binary classification.

If you skipped any of those, read them before sitting Lab 4 — the assumptions stack.

---

## 8. Common confusions the chapter resolves
- **"Backprop computes gradients of weights only."** — No. Backprop computes gradients with respect to whatever you ask. In Module 5 you will backprop into the *input* (for adversarial examples); in Module 8 you will backprop into a *random latent*.
- **"Backprop = SGD."** — No (and Goodfellow is firm on this). Backprop *gives* you gradients; SGD, Adam, AdamW, etc. *consume* them.
- **"PyTorch does backprop for me, so I don't need to understand it."** — Wrong. Half of DL debugging is reading PyTorch tracebacks that point at gradient flow.

---

## 9. Vocabulary introduced or formalised in this chapter

You should be able to define each in one sentence:

- **Universal Approximation Theorem** — single hidden layer + enough units can approximate any continuous function on a compact set.
- **Symmetry breaking** — random initialisation so that hidden units start in different states.
- **Xavier / Glorot init** — `var(W) = 1/fan_in` (for tanh/sigmoid).
- **Kaiming / He init** — `var(W) = 2/fan_in` (for ReLU).
- **Dead ReLU** — a unit stuck at zero pre-activation with no gradient flow.
- **Reverse-mode automatic differentiation** — the cheap way to compute the gradient of a scalar loss with respect to many parameters.
- **Forward pass / backward pass / parameter update** — three distinct stages of a training step.

---

## 10. Study questions — answer in your own words

1. State the Universal Approximation Theorem precisely. Why is it usually misquoted as "a single hidden layer suffices"?
2. Why does Kaiming init use `2/fan_in` while Xavier uses `1/fan_in`? Where does the factor of 2 come from?
3. A ReLU network is trained, and you observe that 30 % of units have output exactly 0 for every example in the validation set. Diagnose the issue and propose two fixes.
4. Show that for a stack of 10 sigmoid layers initialised with `W ∼ N(0, 1/n)`, the expected gradient at the bottom of the stack is at most `(1/4)^10 ≈ 1e-6` of the gradient at the top. Why does the same argument fail for ReLU networks?
5. Compute the FLOP cost of one *forward pass* and one *backward pass* through `nn.Linear(d_in, d_out)`. Why is the backward roughly 2× the forward?
6. Why is reverse-mode AD cheaper than forward-mode AD for neural networks? Under what (very rare) condition would forward-mode be cheaper?
7. In your own words, why is "backprop" not the same thing as "training a neural network"?

Bring written answers (1 page) to Session 4.

---

## 11. Key takeaways

1. **Activations:** ReLU is the default for hidden units in 2026. Sigmoid and tanh survive at output layers and in gating.
2. **Initialisation:** match the init to the activation — Kaiming for ReLU, Xavier for tanh/sigmoid. Don't override framework defaults without a reason.
3. **Depth > width** for nearly all functions you care about, *given enough optimisation tooling* (ReLU, init, skip connections).
4. **Backprop is reverse-mode AD.** Every framework you use this term implements the same algorithm.
5. **You should be able to derive backprop on a 2-layer MLP from memory** — Reading 2 will train this until it's automatic.

---

## 12. Recommended next steps

- Read **Module 4 Reading 2** — Karpathy's "Backprop Is Just the Chain Rule" companion notes. They cover the *intuition* the textbook is too formal to deliver.
- Watch **3Blue1Brown's neural-network series** (4 episodes on YouTube) — visual companion to §6.5.
- Skim **PyTorch autograd docs** (`https://pytorch.org/docs/stable/notes/autograd.html`).
- Re-derive the 2-layer MLP gradient on a whiteboard, three times. The third time it will be automatic.