# Reading 2 — Companion Notes for Olah, 'Understanding LSTM Networks'

Line-by-line companion to Olah's classic LSTM post: the four equations of an LSTM cell, why cell state defeats vanishing gradients, engineering tips Olah leaves implicit, where the post is now dated.

---

# Companion Notes — Olah, *Understanding LSTM Networks*

> **Module 7 Reading 2** — A guided study companion to Christopher Olah's classic post **"Understanding LSTM Networks"** (August 2015). The original is freely available at **https://colah.github.io/posts/2015-08-Understanding-LSTMs/**. Olah's diagrams are the gold standard for understanding gating; read the original first with your notebook open.

If Goodfellow Chapter 10 (Reading 1) is *why* RNNs and LSTMs exist, Olah's post is *how the gates actually work*. Together they cover both halves of "I understand LSTMs."

---

## 0. How to use this reading

1. **Read Olah's original.** ~25 minutes. The diagrams do most of the teaching.
2. **Come back here** for the line-by-line walk-through, the connections to Goodfellow §10.10, and the engineering tips Olah elides.
3. **Answer the study questions in §9** before Assignment 7.

The post itself is a series of diagrams + equations + intuition. We will not redraw them — open the post in a second window.

---

## 1. The pitch

Olah's three-part argument:

1. **Vanilla RNNs forget.** Information from step 1 is, at best, a faint echo by step 100.
2. **LSTMs add a "highway" — the cell state** — through which information can flow without distortion.
3. **Three gates** decide what to write to, read from, and forget on this highway.

Once those three sentences click, the rest is implementation detail.

---

## 2. The diagram you must internalise

Olah draws each RNN cell as a horizontal box with:
- **An input arrow from below** carrying `x^(t)`.
- **An arrow from the previous cell on the left** carrying `h^(t-1)` (and for LSTM, `c^(t-1)`).
- **An arrow to the next cell on the right** carrying `h^(t)` (and for LSTM, `c^(t)`).
- **An arrow out the top** producing `h^(t)` for any downstream layers.

For an LSTM, the box contains four internal operations stacked vertically — the forget gate, input gate, candidate cell, and output gate. The cell state `c^(t)` runs along the top of the box, modified only by the forget multiplication and the input addition.

Open Olah's Figure 4 (the LSTM diagram) and trace each arrow until you can re-draw it from memory. **You will be asked to do this in Assignment 7.**

---

## 3. The four equations of an LSTM cell

Reading 1 §8 already gave these. Olah's value-add is mapping each equation to a region of the diagram:

```
f^(t) = σ(W_f [h^(t-1), x^(t)] + b_f)         # FORGET   — gate at the left of c^(t-1)
i^(t) = σ(W_i [h^(t-1), x^(t)] + b_i)         # INPUT    — gate next to forget
g^(t) = tanh(W_g [h^(t-1), x^(t)] + b_g)      # CANDIDATE — content to potentially write
c^(t) = f^(t) ⊙ c^(t-1) + i^(t) ⊙ g^(t)       # CELL UPDATE — the highway
o^(t) = σ(W_o [h^(t-1), x^(t)] + b_o)         # OUTPUT   — gate that picks what to expose
h^(t) = o^(t) ⊙ tanh(c^(t))                    # HIDDEN   — what's seen by the next layer
```

Olah's diagrams show this in two passes:
- *First* the cell-state update (forget · input · candidate · sum).
- *Then* the hidden-state update (output gate · tanh-of-cell).

The most subtle point is that `h^(t)` and `c^(t)` are **different**. `c^(t)` is the *memory*; `h^(t)` is the *output exposed to the rest of the network*.

---

## 4. Why this defeats vanishing gradients

Olah ends the post with the punch line. The gradient flowing back through the cell state involves repeated multiplications by the **forget gate** `f^(t)`. When `f^(t) ≈ 1`, the gradient survives. When `f^(t)` is exactly 1 and `i^(t)` is exactly 0, the cell state is **untouched** — gradients pass through unchanged.

Contrast this with a vanilla RNN, where the Jacobian at each step is `diag(1 − h²) · W_h^T` — saturated tanh + a matrix product that almost always shrinks.

Two consequences:
- An LSTM can learn dependencies across hundreds of time steps. A vanilla RNN can manage maybe 20.
- The forget gate is the *most important* parameter in an LSTM. Initialising its bias to `+1` (so it starts mostly open) is a well-known training trick (Jozefowicz et al., 2015).

---

## 5. Variants Olah briefly mentions

Olah doesn't dwell on these, but you should know them by name:

- **Peephole connections** — gates see `c^(t-1)` directly. Marginal benefit.
- **Coupled forget/input gates** — `i^(t) = 1 − f^(t)`. Smaller, comparable performance.
- **GRU (Cho et al., 2014)** — combines forget and input into one "update gate". Drops the explicit cell state. Lab 7 measures this against vanilla LSTM.

Olah's diagram of the GRU is the cleanest in print; check it before Lab 7.

---

## 6. The engineering tips Olah leaves implicit

Things every LSTM practitioner learns the hard way:

1. **Initialise the forget-gate bias to 1.** PyTorch's `nn.LSTM` does *not* do this by default. Worth a manual override.
2. **Gradient-clip at every step.** Even with LSTM gating, exploding gradients can hit. `clip_grad_norm_(params, 1.0)` is the canonical default.
3. **Layer-norm the recurrence.** Variant: *Layer-Normalized LSTM* (Ba et al., 2016). Improves training stability noticeably.
4. **Use `pack_padded_sequence`** for variable-length batches in PyTorch. Saves a lot of compute on padded tokens.
5. **Bidirectional only when you have full sequence.** For real-time / streaming inference, you can't go bidirectional.
6. **Mixed precision can underflow** at LSTM gate values close to 0 or 1. Use bf16 or fp32 if you see NaNs.

You will use 1–4 in Lab 7. We'll note 5 and 6 when they come up later.

---

## 7. Where Olah's post is dated

Olah's post was the canonical LSTM explainer in 2015. The two things it predates:

- **Attention** (Bahdanau et al., 2015; Vaswani et al., 2017). Olah hints at attention at the end of the post — *that's* what made Transformers possible.
- **The Transformer** (Module 11). Once attention removed the encoder-decoder bottleneck *and* removed recurrence entirely, LSTMs ceded most of NLP.

The diagrams still teach gating beautifully. The framing — "you need LSTMs to handle long sequences" — is no longer how 2026 builds models.

---

## 8. Connections to DL2026

| Olah idea | Returns in |
|---|---|
| Cell state as gradient highway | Module 5 (ResNet identity shortcuts), Module 11 (residual connections in Transformer blocks). |
| Gating | Module 11 (gated MLP variants, GLU, Mixture of Experts routers). |
| Forget-gate bias = 1 init | Module 12 (init choices). |
| GRU as parameter-efficient variant | Lab 7 (you'll measure this directly). |

---

## 9. Study questions

1. Draw the LSTM cell from memory. Label `c^(t-1)`, `c^(t)`, `h^(t-1)`, `h^(t)`, `f`, `i`, `g`, `o`. Compare to Olah's Figure 4.
2. The cell-state update is `c^(t) = f^(t) ⊙ c^(t-1) + i^(t) ⊙ g^(t)`. Why is this addition (rather than a `tanh` of a sum) the load-bearing trick? What would happen if it were `c^(t) = tanh(f^(t) ⊙ c^(t-1) + i^(t) ⊙ g^(t))`?
3. In a forget-gate-bias = 0 init, what is the expected value of `f^(t)` at step 0? At a +1 init?
4. State the GRU's update equations and identify the *single* gate that does double duty (replacing both forget and input).
5. Why does Olah call attention "an extension" at the end of the post? What problem with LSTMs is attention trying to fix?
6. A 2-layer LSTM has roughly how many parameters when `h = 256`, `vocab = 65`, sequence length is unlimited? (Hint: the answer doesn't depend on `T`.)
7. Olah doesn't discuss truncated BPTT. Why is it necessary in practice, and what's the standard truncation length for char-RNNs on a long corpus?

Bring written answers (≤ 1 page) to Assignment 7.

---

## 10. Key takeaways

1. The LSTM is a *cell state* + *three gates* + *one output gate*. Everything else is bookkeeping.
2. The cell state is a **gradient highway** — gradients flow back through it without the per-step shrinkage that kills vanilla RNNs.
3. **Forget-gate bias = +1 init** is a free win. Many frameworks don't apply it; do it manually.
4. **GRU** is the leaner alternative — usually within 1–2 % of LSTM accuracy at ⅔ the parameter count.
5. The Olah post predates attention; it still teaches gating perfectly, but the broader narrative has moved to Transformers (Module 11).

---

## 11. Recommended next steps

- Re-skim Olah's diagrams *without the equations*. If the diagrams alone explain the model to you, you've absorbed Module 7.
- Open the **PyTorch `nn.LSTM` source** and find the forget-gate bias init. Note that it doesn't add +1. Override in Assignment 7.
- Karpathy's *The Unreasonable Effectiveness of Recurrent Neural Networks* gives the exact char-RNN setup you'll build in Assignment 7.
- After Module 11, re-read Olah's last paragraph (the one mentioning attention). You'll suddenly understand why it was on the path to the Transformer.

> When you can draw Olah's LSTM diagram from memory and explain in one sentence why the cell state defeats vanishing gradients, you have absorbed Module 7 Reading 2.