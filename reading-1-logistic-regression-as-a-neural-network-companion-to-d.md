# Reading 1 — Logistic Regression as a Neural Network (Companion to deeplearning.ai)

Companion notes to deeplearning.ai's 'Logistic Regression as a NN' lectures, re-cast in DL2026 notation with study questions and a bridge to Module 4.

---

# Logistic Regression as a Neural Network — Companion Notes

> **Module 3 Reading 1** — A guided study companion to the *deeplearning.ai* lectures on **"Logistic Regression as a Neural Network"** (Andrew Ng, *Neural Networks and Deep Learning*, Coursera, Week 2). The original lectures and notes are at **https://www.coursera.org/learn/neural-networks-deep-learning** and the linked transcript / lecture-notes PDFs on the course website. **These notes summarise and extend that material — they do not replace it.** Watch (or skim the transcript of) Week 2 first; this document is the second pass.

---

## 0. How to use this reading

1. Watch the *deeplearning.ai* Week 2 videos in order (or read the linked transcript / notes):
   - "Binary Classification"
   - "Logistic Regression"
   - "Logistic Regression Cost Function"
   - "Gradient Descent"
   - "Derivatives", "More Derivative Examples", "Computation Graph"
   - "Logistic Regression Gradient Descent"
   - "Vectorization", "Vectorizing Logistic Regression"
2. Then walk through this document section-by-section. Where Ng's notation differs from ours, this reading uses the **DL2026 notation** (matrices uppercase, vectors lowercase, batch dimension first) so it lines up with Module 2 Reading 1 and Module 3 Reading 2.
3. Answer the study questions in §10 before Session 3 lab.

The single most important reframing here is **"logistic regression *is* a one-neuron neural network"** — once you absorb that, every later module is a stack of these primitives.

---

## 1. The pitch — why this reading matters

Logistic regression is the **smallest object** that contains every idea you need to build deep networks:

- a **linear transformation** `z = w · x + b`,
- a **nonlinear activation** `a = σ(z)`,
- a **loss function** `L = − [y log a + (1 − y) log(1 − a)]`,
- a **gradient update** `w ← w − η · ∂L/∂w`.

A 200-billion-parameter Transformer is *the same four ideas* arranged in a deeper graph. If you understand logistic regression as a NN, you can derive backprop for any deeper network by composition.

---

## 2. Binary classification — the setup

A binary classification problem has:
- inputs `x ∈ ℝ^n` (e.g. flattened image pixels),
- labels `y ∈ {0, 1}` (cat vs. not-cat).

The model predicts a *probability* `ŷ = P(y = 1 | x)`. We want `ŷ` close to 1 when `y = 1`, close to 0 when `y = 0`.

Notation we will use throughout DL2026:

| Symbol | Meaning |
|--------|---------|
| `n` | input dimension |
| `m` | number of training examples |
| `X ∈ ℝ^{m×n}` | training matrix (one row per example) |
| `y ∈ {0, 1}^m` | label vector |
| `w ∈ ℝ^n` | weight vector |
| `b ∈ ℝ` | bias scalar |
| `z ∈ ℝ^m` | pre-activation, `z = X w + b` |
| `a = ŷ ∈ (0, 1)^m` | post-activation, `a = σ(z)` |

> *Note.* Andrew Ng's Coursera course uses `x ∈ ℝ^{n×m}` (features as rows, examples as columns). DL2026 uses the PyTorch convention — examples as rows. The math is the same; only the shapes transpose. Use whichever convention matches your code.

---

## 3. The model — one perceptron with a sigmoid

For a single example `x`:

```
z = w · x + b           # scalar
a = σ(z) = 1 / (1 + e^{−z})   # in (0, 1)
ŷ = a                    # the predicted P(y = 1 | x)
```

That is **logistic regression**. It is also `nn.Linear(n, 1)` followed by `sigmoid` in PyTorch — exactly the model you'll build in Lab 3.

Geometrically, the model defines a **hyperplane** `w · x + b = 0` in input space. Points on the `+w` side become probability > 0.5, points on the `−w` side become probability < 0.5. The classifier is purely *linear* — that's both its strength (interpretability) and its limit (Lab 3 plots will show what curved data does to it).

---

## 4. The loss — binary cross-entropy / log-loss

Why don't we use squared error `½ (a − y)²` for classification? Two reasons:

1. **Non-convexity.** With sigmoid output, MSE is *non-convex* in `w`. Gradient descent gets stuck.
2. **Probabilistic interpretation.** We want to maximise the likelihood that the data came from a Bernoulli distribution with parameter `a`. Maximising likelihood ⇔ minimising **negative log-likelihood** = cross-entropy.

The per-example loss:

```
L(a, y) = − [ y · log a  +  (1 − y) · log(1 − a) ]
```

Interpret it intuitively:
- If `y = 1`: `L = −log a`. This punishes the model for under-confident "yes" predictions. As `a → 0`, `L → ∞`.
- If `y = 0`: `L = −log(1 − a)`. Symmetrically punishes confident-wrong "yes" predictions.

The full cost across `m` examples is the average:

```
J(w, b) = (1/m) Σ_i L(a^{(i)}, y^{(i)})
```

This is what gradient descent will minimise.

> **Sanity check.** `L` is always ≥ 0, with equality only when `a = y` exactly. It is unbounded above when the model is confidently wrong. This is the right "shape" for classification — confident mistakes are punished disproportionately.

(The full mathematical derivation — including the maximum-likelihood interpretation — is in **Module 3 Reading 2 §4**. Treat this section as the verbal version.)

---

## 5. Gradient descent — one step

Gradient descent updates `w` and `b` in the direction that reduces `J`:

```
w ← w − η · ∂J/∂w
b ← b − η · ∂J/∂b
```

`η` is the **learning rate** (small positive scalar; typical first try `0.01`).

What does `∂J/∂w` look like? After applying the chain rule (full derivation in Reading 2 §4.4), it collapses to a strikingly simple form:

```
∂L/∂z = a − y
∂L/∂w = (a − y) · x       (per-example)
∂L/∂b = (a − y)
```

Average over the batch and you get:

```
∂J/∂w = (1/m) Xᵀ (a − y)
∂J/∂b = (1/m) Σ_i (a^{(i)} − y^{(i)})
```

**This is the single most-cited equation in early deep learning courses.** Memorise it. Every later loss / activation pair you meet in DL2026 either matches this form or differs from it in a way you should be able to explain.

> Why so simple? Sigmoid + binary cross-entropy were *designed* to compose into `(a − y)`. It's not coincidence — it's the magic of conjugate likelihood / link functions. Reading 2 §4 derives it.

---

## 6. Vectorisation — the leap from "for-loop" to "matrix"

The biggest performance jump in classical ML happens when you stop writing `for i in range(m):` and start writing matrix products. For logistic regression:

```python
z = X @ w + b                       # (m,)
a = 1 / (1 + np.exp(-z))            # (m,)
J = -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))

# Backward
dz = a - y                          # (m,)
dw = X.T @ dz / m                   # (n,)
db = dz.mean()
```

On a 50,000-row dataset this is ~100× faster than a Python loop and gives identical numbers. GPUs amplify the gap to 10,000×. Every later module assumes you have internalised this — see Lab 3 task 2.

A subtle but important note: **the same vectorised code is the inner step of training every model in this course**. The only thing that changes between this and a 12-layer Transformer is the depth of `X @ w` and the choice of `a = …`.

---

## 7. The "computation graph" view — bridging to backprop

Ng's Week 2 lectures introduce the **computation graph** as a tool. For logistic regression with one example and two features `x_1, x_2`:

```
x_1 ──┐
       ├──> z = w_1 x_1 + w_2 x_2 + b ──> a = σ(z) ──> L(a, y)
x_2 ──┘
```

Backprop walks this graph **right-to-left**, accumulating partial derivatives:

```
∂L/∂a = (a − y) / [a (1 − a)]      # from the BCE formula
∂a/∂z = a(1 − a)                    # from σ'(z) = σ(z)(1−σ(z))
∂L/∂z = ∂L/∂a · ∂a/∂z = a − y      # ← the cancellation that makes life easy
∂L/∂w_i = ∂L/∂z · x_i
∂L/∂b   = ∂L/∂z
```

The graph view is exactly what PyTorch / TensorFlow / JAX build automatically when you write `loss.backward()`. **A logistic-regression neuron is the smallest non-trivial computation graph in the universe of neural networks**, and it is the one you should be able to draw and differentiate without notes.

(Reading 2 §3.2 derives `σ'(z) = σ(z)(1 − σ(z))` from scratch.)

---

## 8. From logistic regression to a deep network — what changes?

A deep network is logistic regression with one extra move: stack one or more *hidden layers* between input and output.

| Object | Logistic regression | Deep network |
|--------|--------------------|---------------|
| Linear part | one row vector `w` | many matrices `W^{(ℓ)}` |
| Nonlinearity | sigmoid at the output | ReLU/GELU at every hidden layer + sigmoid/softmax at the output |
| Loss | binary cross-entropy | binary or categorical cross-entropy |
| Backprop | one chain-rule step | many chain-rule steps composed (Module 4) |
| Code length in PyTorch | 1 line | 10–50 lines |

Notice what does **not** change: the **per-layer backprop step is still `(a − y)` at the loss, then `Wᵀ · δ` propagated backwards.** That's why Reading 1 of Module 2 emphasised "deep learning is composition of simple primitives" — logistic regression is the *atomic* primitive.

---

## 9. Pitfalls Ng calls out (and that you will meet in Lab 3)

1. **Numerically unstable `log a`.** When `a → 0`, `log a → −∞`. In practice, use `log_sigmoid(z)` or `logsumexp` tricks — never compute `log(sigmoid(z))` directly. (Reading 2 §3.4 has the stable version.)
2. **Initialisation.** For logistic regression you can initialise `w = 0` and `b = 0`. For *deep* networks you can't (it breaks symmetry). Module 4 explains why.
3. **Feature scaling.** Logistic regression's convergence rate is sensitive to feature scale; un-normalised features can make `η` very hard to tune. Normalise inputs or standardise (mean-0, std-1).
4. **Class imbalance.** If 99 % of examples are negative, a "predict 0 always" classifier hits 99 % accuracy and 0 % recall. Look at AUC or PR-AUC before celebrating.
5. **Overfitting with too many features.** Even logistic regression overfits when `n > m`. Use L2 regularisation (`+ λ ‖w‖²`) — covered in detail in Module 12.

---

## 10. Study questions — answer in your own words

1. State logistic regression as a one-line `nn.Sequential` definition in PyTorch.
2. Why is binary cross-entropy preferred to MSE for binary classification? (Two reasons.)
3. Show that `σ'(z) = σ(z)(1 − σ(z))`. (Compute the derivative from `1 / (1 + e^{−z})`.) Verify your derivation against Reading 2 §3.2.
4. Why does `∂L/∂z = a − y` come out so clean? (Hint: chain rule + the `a(1 − a)` cancellation.)
5. Suppose you have 1,000 features but only 200 examples. What goes wrong with un-regularised logistic regression? Name two cures.
6. The deeplearning.ai lectures show a `for i in range(m)` version *and* a vectorised version. Why is the vectorised version often *more numerically stable* on top of being faster?
7. Modify the model to do **multi-class** classification with 10 classes. What changes in: (a) the model's last layer, (b) the loss function, (c) the gradient at the output? See Reading 2 §6 for the answers.

Bring written answers (1 page) to Session 3.

---

## 11. Key takeaways

1. Logistic regression *is* a one-neuron neural network. Sigmoid output + binary cross-entropy.
2. The loss is binary cross-entropy because it is **convex** in the linear weights and is the **negative log-likelihood** of the Bernoulli model.
3. The gradient at the pre-activation is `a − y` — the simplest, most-used gradient in early DL.
4. Vectorise: matrix-vector products replace loops; this is non-optional in practice.
5. Numerical stability matters from day one — never compute `log(sigmoid(z))` directly.
6. Everything you do in Modules 4–13 is a deeper, wider, or fancier version of this single neuron.

---

## 12. Recommended next steps

- Re-derive `∂L/∂z = a − y` with pen and paper. Three minutes well spent.
- Read **Module 3 Reading 2** for the full derivations and the softmax/categorical-CE extension.
- Skim Andrew Ng's *Heroes of Deep Learning* interview with Geoffrey Hinton for the historical motivation of cross-entropy as a loss.
- Bookmark Karpathy's "*Yes you should understand backprop*" — you will return to it before Module 4.
- Lab 3 is where you implement everything in this reading from scratch in NumPy. Do not skip the from-scratch version; libraries make this look easy but the gradient code is what you need to internalise.

> A student who can derive logistic regression's gradient by hand, vectorise it, and explain why `(a − y)` falls out cleanly is half-way to understanding the rest of DL2026.