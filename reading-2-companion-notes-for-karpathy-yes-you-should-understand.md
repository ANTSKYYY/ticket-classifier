# Reading 2 — Companion Notes for Karpathy, 'Yes you should understand backprop'

Companion to Karpathy's classic essay. Walks the four leaks of the backprop abstraction (sigmoid saturation, dead ReLUs, exploding gradients, data-pipeline bugs) + a fifth on mixed precision, with PyTorch demos and a diagnostic cheat sheet.

---

# Companion Notes — Karpathy, "Yes you should understand backprop"

> **Module 4 Reading 2** — A guided study companion to Andrej Karpathy's essay **"Yes you should understand backprop"** (the canonical version of *"Backprop Is Just the Chain Rule"*), originally published at **https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b**. The essay is freely available; read it once before working through these notes.

Karpathy's argument is short and sharp: **backprop is a leaky abstraction**, and pretending it isn't is the fastest way to ship a broken model. This reading walks through the four canonical "leaks" he names, expands each with worked examples, and ties them to DL2026 modules where the leak bites you for real.

---

## 0. The thesis in one paragraph

Modern frameworks (PyTorch, TensorFlow, JAX) let you call `loss.backward()` and get gradients "for free". Karpathy's claim — built from his experience teaching Stanford's CS231n — is that *the abstraction breaks in four specific places* often enough that any practitioner who can't reason about backprop will spend days debugging. Understanding backprop is not an academic nicety; it is a workplace skill.

This reading covers each of those four leaks, plus a fifth that has emerged since the essay was written (mixed precision / autocast).

---

## 1. Why this essay survives 8 years later

Most "should-I-care-about-the-internals" essays in deep learning age badly because the internals change. Karpathy's essay survives because **the chain rule does not change**. Frameworks abstract more and more, but the gradient still has to flow through every multiplication, every saturation, every clipping operation in your graph. Where the gradient *can't* flow, your model *can't* learn. That hasn't gone away in 8 years; it has only become *more* relevant as models grew taller and deeper.

If you only remember one sentence from this entire module, remember:
> **A neural network learns where the gradient can flow. Everything else is decoration.**

---

## 2. The chain rule, in 90 seconds (Karpathy's tone)

For `y = f(g(x))`:
```
dy/dx = f'(g(x)) · g'(x)
```

For a *graph* (many `f`s and `g`s),  every node computes its **local Jacobian** during the forward pass, and the chain rule strings these together during the backward pass. *That's the whole algorithm.* Every framework's `backward()` is `for each node in reverse topological order, multiply by the local Jacobian and add into the input gradients`.

Karpathy's point isn't that the chain rule is hard. It's that **the chain rule has failure modes**, and those failure modes show up as model bugs.

---

## 3. Leak 1 — Sigmoids saturate (the classical pitfall)

### 3.1 The setup
The sigmoid's derivative is `σ'(z) = σ(z)(1 − σ(z))`, peaking at **0.25** when `z = 0` and decaying toward 0 as `|z| → ∞`.

### 3.2 The leak
Stack 10 sigmoid layers and the gradient at the bottom is multiplied by *at most* `0.25^10 ≈ 9.5e-7`. Gradient vanishes; weights at the bottom of the stack never move; *the bottom of your network never learns.* Karpathy puts this baldly: "your deeper layers will be just as useful as if you had randomly initialised them, because they never train".

### 3.3 The cure
- Use **ReLU** in hidden layers (gradient is exactly 1 when active).
- Use **skip / residual connections** (a gradient highway around the saturating region).
- Use **layer normalisation** or **batch normalisation** to keep pre-activations near 0 (Module 12).

### 3.4 Where this still bites you in 2026
- **LSTM gating** — the input/forget/output gates use sigmoid. Training a deep stack of LSTMs is hard for exactly this reason. (Module 7 covers the mitigations.)
- **Attention weights** in Transformers use softmax, which is a *generalised* sigmoid. The same saturation can stop gradients from flowing through attention heads that "decide" early. Module 11 discusses fixes (scaled dot-product, attention dropout).

### 3.5 Debugging recipe
If your loss plateaus at the chance level for the first few epochs and your network has > 5 hidden sigmoid layers, **assume vanishing gradients first**. Plot the per-layer gradient norm — it should be reasonably flat. If it decays exponentially with depth, you have this leak.

---

## 4. Leak 2 — Dead ReLUs

### 4.1 The setup
ReLU's derivative is **0** for `z < 0`. If a unit's pre-activation is negative for *every* training example, it has zero gradient *forever* — no learning signal, ever.

### 4.2 The leak
A common failure mode: you set the learning rate too high; one SGD step shifts the bias of a ReLU unit far negative; the unit becomes inactive on every example; it stays inactive. Karpathy's measurement: with a poorly-tuned LR, *40% of ReLU units in a small CNN died within 10 steps*.

### 4.3 The cure
- **Lower the learning rate** (the cheapest fix).
- Use **Leaky ReLU** or **ELU** — they let a small gradient through on the negative side.
- Use **Kaiming initialisation** — designed precisely so that, at init, roughly half the units are active and the variance of pre-activations is preserved.
- **Use a warm-up schedule** for the LR (Module 12).

### 4.4 Debugging recipe
After training a few epochs, count how many ReLU units have *zero variance* across the training batch. If it's > 10 %, you have dead-ReLU disease.

```python
with torch.no_grad():
    a = relu_layer(x_batch)
    dead_frac = (a.std(dim=0) == 0).float().mean().item()
    print(f"dead ReLU fraction = {dead_frac:.2%}")
```

### 4.5 Where this still bites you
- **Quantised models** (Module 13 deployment): when activations are clipped to small integers, "dead unit" gets even easier to trigger.
- **Sparse attention** layers — a head that never fires never trains.

---

## 5. Leak 3 — Exploding gradients in deep / recurrent networks

### 5.1 The setup
Forward through a deep network: each layer's local Jacobian `J_ℓ` multiplies the upstream gradient. If `‖J_ℓ‖ > 1` consistently, the gradient explodes; if `< 1`, it vanishes. Sigmoid networks vanish; **unnormalised RNNs and deep CNNs without skip connections explode**.

### 5.2 The leak
A single exploded gradient can blow your weights into NaN territory in one step. You see `loss = nan` and the whole model is unrecoverable from the most recent checkpoint.

### 5.3 The cure
- **Gradient clipping**: `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`. Module 7 lab makes this concrete.
- **Skip connections** to provide an identity gradient path.
- **Layer / batch normalisation** to control activation statistics → controls Jacobian spectra.
- **Lower learning rate**.

### 5.4 Where this still bites you
- **RNNs / GRUs / LSTMs** trained on long sequences — gradients across 1000 time-steps are the textbook exploding-gradient case (Module 7).
- **Diffusion models** (training across many denoising steps) — implicit "depth" in time.
- **Reinforcement learning** with policy gradients — value-function bootstrapping can amplify gradients across long trajectories (Module 9).

### 5.5 Debugging recipe
Log `‖grad‖` per step. If you see a spike followed by `nan`, clip the gradient. If clipping at norm 1 stops the spike but the model still doesn't learn, your learning rate is the upstream problem.

---

## 6. Leak 4 — DataLoader and label-smoothing bugs that *look* like gradient bugs

Karpathy lists a fourth leak that he calls "the silent killers": **bugs in data preprocessing or label encoding that look like the model is broken**, because the optimiser is being fed garbage gradients.

### 6.1 Examples
- One-hot labels mis-shaped: model receives `(B, 1, K)` instead of `(B, K)`. Cross-entropy silently broadcasts and the loss is computed against the wrong target.
- A `Dataset.__getitem__` returns a slice of *one* example duplicated 32 times. The batch looks normal in shape, but the gradient signal is the same example repeated.
- A normalisation pipeline computes mean/std on the full dataset (including test) — see Assignment 3 — the model train loss looks great, test accuracy is mysteriously low.

### 6.2 Cure
- Add a `assert` on the shape and dtype of every input/output at the boundary of every component.
- Use a tiny "overfit-one-batch" sanity check: if your model cannot get *training* loss to ~0 on a single batch, the model or the data is broken, not the optimisation.
- Visualise random samples *as the model sees them* after augmentation, normalisation, and label encoding.

This is the single highest-yield debugging recipe in the essay.

---

## 7. Leak 5 — Mixed precision (post-Karpathy update)

Karpathy's essay predates the widespread use of **mixed-precision training** (fp16 / bf16). Since 2018 it has introduced one new leak worth knowing:

- In fp16, `1e-7` underflows to 0 silently.
- Small gradients can vanish numerically even when the math says they shouldn't.
- The standard cure is a **dynamic loss scaler** (`torch.cuda.amp.GradScaler`) — multiply the loss by a large constant before backward, divide gradients back after, undo NaN-poisoned steps.

You will meet this for real in Module 13. The takeaway: **the chain rule's failure modes are not only about the math — they're about the numerical representation that math is computed in.**

---

## 8. The "compute one gradient by hand" exercise (Karpathy's prescription)

Karpathy says the most useful exercise for understanding backprop is, *for any new layer you ever write*:

1. Sketch the computation graph.
2. Write the forward pass in NumPy.
3. Write the analytical backward by hand.
4. Run a finite-difference gradient check against your analytical backward.

You did this in **Assignment 3** for the logistic-regression gradient. Module 4's lab will extend it to an MLP. Module 11 (Transformers) will extend it to attention. **In every case the exercise is the same**: understand the local Jacobian of every node in your graph, then trust the chain rule to do the rest.

---

## 9. Worked example — vanishing-gradient demonstration

Karpathy supports his vanishing-gradient claim with a 20-line NumPy demo. Here is a slightly modernised PyTorch version you should run while reading.

```python
import torch, torch.nn as nn
torch.manual_seed(0)

def measure_grad_norms(activation, depth=10, dim=64):
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(dim, dim), activation()]
    net = nn.Sequential(*layers, nn.Linear(dim, 1))
    x = torch.randn(8, dim)
    y = torch.randn(8, 1)
    nn.MSELoss()(net(x), y).backward()
    norms = []
    for m in net.modules():
        if isinstance(m, nn.Linear):
            norms.append(m.weight.grad.norm().item())
    return norms

print("Sigmoid grad norms by depth:", measure_grad_norms(nn.Sigmoid))
print("ReLU    grad norms by depth:", measure_grad_norms(nn.ReLU))
```

Expected output (orders of magnitude):
- Sigmoid: grad norm at the **first** linear is ~`1e-8`, climbing to ~`1e-1` at the last linear. **6 orders of magnitude difference across 10 layers.**
- ReLU: grad norm is roughly the same order of magnitude across all 10 layers — *flat*. This is what you want.

Run it. Look at the numbers. They are the empirical proof of Karpathy's thesis.

---

## 10. A cheat sheet for diagnosing gradient pathologies

When training a network and something looks wrong, walk this list in order:

1. **Overfit one batch.** Does training loss reach ~0 on a single batch? If no → model/data bug, not optimisation.
2. **Per-layer gradient norm.** Compute `‖grad‖` at every parameter group. Should be roughly flat by depth.
3. **Activation statistics.** Compute mean and variance of activations per layer. Drift toward 0 or ±∞ → vanishing or exploding.
4. **Dead-unit fraction.** For ReLU layers, count units with zero variance.
5. **Learning-rate sweep.** Try LRs across three orders of magnitude. If only one tiny LR doesn't NaN, the problem is too-large updates → look at init or clip.
6. **Numerical precision.** Repeat the best LR run in fp32 — does it still NaN? If no, the bug was fp16.
7. **Data sanity.** Visualise the network's input *after all preprocessing*. Are the labels what you think they are?

If you find yourself running PyTorch's `register_hook` on every layer, you are doing this list right.

---

## 11. Connections to DL2026

| Leak | First fixed in | Bites again in |
|------|---------------|-----------------|
| Sigmoid saturation | Module 4 (ReLU + Kaiming) | Module 7 (LSTM gates), Module 11 (attention) |
| Dead ReLU | Module 12 (init + LR schedules) | Module 13 (quantisation) |
| Exploding gradients | Module 7 (clipping) | Module 9 (RL), Module 11 (deep Transformer) |
| Data-pipeline bug | Module 2 (proposal: overfit-one-batch sanity) | Every module |
| Mixed-precision | Module 13 | Production deployment |

---

## 12. Study questions

1. **Sigmoid math.** Show that the maximum of `σ'(z)` over all `z` is `0.25`. Derive it.
2. **Dead ReLU forensics.** A teammate trained an MLP on tabular data; the loss got stuck at ~the chance level after 1 epoch. Sketch the three measurements you would take to determine whether the cause is (a) vanishing gradients, (b) dead ReLUs, or (c) data shuffling bug.
3. **Gradient clipping.** Write the two-line PyTorch idiom that clips the gradient to a max global L2 norm of 1.0. Where in the training loop does it go — before or after `opt.step()`?
4. **Why ReLU networks need Kaiming init and not Xavier.** Explain in two sentences referencing the variance preservation argument.
5. **Mixed-precision intuition.** Why does multiplying the loss by 1024 before backward keep small gradients alive in fp16? Why must we divide them back after? Why does PyTorch's `GradScaler` skip the optimiser step when a NaN is detected?
6. **Karpathy's overfit-one-batch test.** State the test, explain why it isolates "is the model architecture broken?" from "are the data/labels broken?", and write the 6-line PyTorch snippet that performs it.

Bring written answers (1 page max) to Session 4. The lab assumes you have run the §9 demo and have working intuition for the gradient norms it produces.

---

## 13. Key takeaways

1. Backprop is the chain rule applied to a computation graph — nothing more, nothing less.
2. The chain rule has *four* canonical failure modes Karpathy named, plus a fifth from mixed precision.
3. **Sigmoid saturation, dead ReLUs, exploding gradients, data-pipeline bugs, fp16 underflow** — every one of them looks like "my model doesn't learn" until you look at gradients.
4. The single most useful diagnostic is **overfit-one-batch**.
5. Frameworks abstract backprop, but they cannot abstract its failure modes — only an engineer who understands the chain rule can.

---

## 14. Recommended next steps

- **Run the §9 demo** in a notebook. Save the numbers in a markdown cell — Lab 4 will refer to them.
- **Skim PyTorch autograd docs**, especially `register_full_backward_hook` — Lab 4 may use it.
- **Re-read Goodfellow §6.5** with Karpathy's eyes — you'll see why the textbook is so careful about pseudocode.
- Bookmark Karpathy's *Neural Networks: Zero to Hero* lecture series. Episode 1 ("micrograd") is the half-hour version of this entire reading.

> If you can name the four leaks, recognise each in a stack trace, and write the fix without looking it up, you have absorbed Module 4.