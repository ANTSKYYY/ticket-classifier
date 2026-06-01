# Reading 1 — Companion Notes for Goodfellow Ch. 10 (Sequence Modeling: RNNs)

Structured companion to Goodfellow Ch. 10: temporal priors, unrolling, BPTT, vanishing/exploding gradients, LSTM gating, where the 2016 textbook is now dated.

---

# Companion Notes — Goodfellow et al., *Deep Learning*, Chapter 10 (Sequence Modeling: Recurrent and Recursive Nets)

> **Module 7 Reading 1** — A guided study companion for **Chapter 10** of Goodfellow, Bengio, and Courville's *Deep Learning* (MIT Press, 2016). The book is free at **https://www.deeplearningbook.org**. These notes summarise, paraphrase, and connect Chapter 10 to the rest of DL2026 — they are *not* a substitute for reading the original.

---

## 0. How to use this reading

1. **Open the original** at deeplearningbook.org → Chapter 10. ~50 pages, dense.
2. **Read it once.** Budget ~2 hours with pen and paper.
3. **Come back here** for the structured second pass, the *2016-is-now-old* commentary, and the connections to Modules 11 (Transformers) and 9 (deep RL).
4. **Answer the study questions in §10** before Lab 7.

Notation reminder for this reading:
- Inputs `x^(1), ..., x^(T)` are a sequence of vectors.
- Hidden state `h^(t) ∈ ℝ^h`.
- Outputs `ŷ^(t) ∈ ℝ^K`.

---

## 1. Why sequences need their own chapter

Up through Module 5 the world was *static images*. Module 7 introduces the **time / order** axis: text, audio, sensor readings, decision sequences, source code, DNA. Static models don't know what to do with these — they treat position 1 the same as position 1000.

Chapter 10's pitch is that **recurrence is the most natural prior for sequences**:
- *Sparse interactions across time* — `h^(t)` only depends on `h^(t−1)` and `x^(t)`, not on the entire history.
- *Parameter sharing across time* — the *same* weights `W_h, W_x` apply at every step.
- *Variable-length input handling* — the same model runs on a 10-token sequence and a 10 000-token sequence.

These are the temporal analogues of the spatial priors in Module 5. Recognising the parallel makes the rest of Chapter 10 click.

---

## 2. §10.1 — Unfolding the computation graph

A recurrent net `h^(t) = f(h^(t−1), x^(t); θ)` is, equivalently, a *deep feedforward network* whose depth equals the sequence length `T`. **Unrolling** is the visualisation trick that turns the loop into a chain you can backprop through.

Two consequences:

1. The graph depth grows with `T` — gradients propagate through `T` matrix multiplications. This is the *single most important fact* about RNN training.
2. The parameter count does *not* grow with `T`. Parameters are shared, not duplicated. This is what makes the architecture viable on long sequences.

Goodfellow emphasises that "unrolled RNN" is a *visualisation*, not a different model. Same parameters, same forward, same backward — drawn differently for our benefit.

---

## 3. §10.2 — Recurrent network designs

The textbook lists three canonical designs:

### 3.1 Output-at-every-step
```
h^(t) = tanh(W_h h^(t-1) + W_x x^(t) + b)
y^(t) = W_y h^(t) + b_y
```
Used for **language modelling** (predict next char/word) and **sequence labelling** (POS tagging, NER).

### 3.2 Output-only-at-the-end
```
h^(t) = tanh(W_h h^(t-1) + W_x x^(t) + b)
y     = W_y h^(T)
```
Used for **sequence classification** (sentiment, intent detection) — Lab 7 builds this.

### 3.3 Encoder–decoder (seq2seq)
```
encoder:  reads x^(1..T) → context c = h^(T)
decoder:  generates y^(1..T') conditioned on c
```
Used for **machine translation**, **summarisation**, **dialogue**. Bahdanau et al. (2015) extended it with **attention** — the seed of Module 11.

---

## 4. §10.3 — Bidirectional RNNs

A vanilla RNN sees only the *past*. A **bidirectional RNN** runs two RNNs:
- Forward: left-to-right, producing `h_→^(t)`.
- Backward: right-to-left, producing `h_←^(t)`.

The final representation per position is `[h_→^(t); h_←^(t)]`. Useful for tasks where the answer at position `t` depends on context from both sides — e.g., word-sense disambiguation.

Used in BERT-style models (Module 11) and many sequence taggers.

---

## 5. §10.4 — Encoder-decoder / sequence-to-sequence

The encoder–decoder is the model that brought RNNs to global attention via Google Translate (2014–2016). Two halves:

1. **Encoder** reads the source sequence and compresses it into a fixed-size vector `c`.
2. **Decoder** generates the target sequence one token at a time, conditioned on `c` and previously-emitted tokens.

The big weakness Goodfellow names: `c` is a **fixed-size bottleneck**. For a 100-word sentence, packing everything into a 1024-dim vector loses information. Bahdanau et al.'s **attention** removed this bottleneck — and a few years later, Transformers (Module 11) removed the recurrence entirely.

---

## 6. §10.5–6 — Deep RNNs and Recursive Networks

### 6.1 Deep RNNs (stacking)
You can stack RNNs by feeding `h_layer1^(t)` into `RNN_layer2`. Most modern RNN-based systems use 2–4 stacked LSTMs.

### 6.2 Recursive networks (tree-structured)
For inputs that are *trees*, not chains (parse trees, abstract syntax trees), the recurrence walks the tree bottom-up. Less common now — Transformers handle these via attention.

---

## 7. §10.7 — The challenge of long-term dependencies

This is the most important section of the chapter. The gradient of the loss with respect to early hidden states involves a *product* of `T − 1` Jacobian matrices `J = ∂h^(t) / ∂h^(t−1)`. Two failure modes:

- **Vanishing gradients.** If the largest singular value of `J` is `< 1`, the gradient shrinks geometrically with `T`. After 50 steps it's effectively zero.
- **Exploding gradients.** If the largest singular value of `J` is `> 1`, the gradient grows geometrically. After 50 steps you get NaN.

This is the temporal version of Karpathy's "leak 1" from Module 4 Reading 2. The cures Goodfellow lists:

1. **Gradient clipping** for explosion (`torch.nn.utils.clip_grad_norm_`).
2. **Gated architectures** for vanishing — LSTM (§10.10), GRU.
3. **Better initialisation** — orthogonal init of recurrent weights keeps singular values near 1.
4. **Skip / residual connections across time** (used in some modern variants).

> The 1990s killed RNN research because of vanishing gradients. The 2014–2016 resurgence happened because LSTMs and Adam together made training viable. The 2018–present **decline** of RNNs happened because attention is faster and learns longer dependencies.

---

## 8. §10.10 — LSTMs (and the gating idea)

Goodfellow's §10.10 introduces the **Long Short-Term Memory** cell (Hochreiter & Schmidhuber, 1997). The mechanism is built around a **cell state** `c^(t)` that flows mostly *linearly* through time — gradients can travel along it without saturating.

Three gates manipulate `c^(t)` at each step:
- **Forget gate** `f^(t) = σ(W_f [h^(t-1), x^(t)])` — decides what to drop from `c^(t-1)`.
- **Input gate** `i^(t)` and **candidate** `g^(t) = tanh(...)` — decides what new info to add.
- **Output gate** `o^(t)` — decides what part of `c^(t)` to expose as `h^(t)`.

The update:
```
c^(t) = f^(t) ⊙ c^(t-1) + i^(t) ⊙ g^(t)
h^(t) = o^(t) ⊙ tanh(c^(t))
```

**The genius**: when the forget gate is open and the input gate is closed, `c^(t) = c^(t-1)` — the gradient flows back unchanged. This is the cure for vanishing gradients that vanilla RNNs lack.

Reading 2 (Olah) is *the* visual explanation of this. Read it before Lab 7.

---

## 9. Other gated variants worth knowing

The chapter touches briefly on:
- **GRU** (Cho et al., 2014) — combines forget and input gates into one "update" gate, drops the explicit cell state. Roughly ⅔ the parameters of LSTM with similar performance on many tasks. Lab 7 lets you measure this directly.
- **Peephole LSTM** — gates see `c^(t-1)` too. Marginal gains.
- **Recurrent Highway Networks**, **clockwork RNNs**, **leaky units** — academic curiosities; rarely used in 2026.

---

## 10. Where the chapter is now (2026) somewhat dated

Chapter 10 was written in 2016, when LSTMs were *the* default for NLP, speech, and translation. The world has moved on:

| 2016 conventional wisdom | 2026 reality |
|---|---|
| LSTMs power Google Translate | Transformer-based models do |
| RNNs dominate language modelling | Transformers (BERT/GPT) do |
| RNNs are the default for audio | Conformer / Wav2Vec (still partly recurrent, but with attention) |
| RNNs win at moderate-data tasks | True, but for short sequences; Transformers catch up with enough data |
| RNN beat MLPs on time series | True; but Temporal Convolutional Networks and Transformers often beat both |

**Where RNNs still win in 2026:**
- **Strictly causal real-time prediction** — RNN's O(T) inference is faster than Transformer's O(T²) on long streams.
- **Edge / on-device inference** — LSTMs are small and fast on phone CPUs.
- **Tiny-data regimes** — fewer parameters → less overfitting.
- **Time-series forecasting with strong autoregressive structure** — LSTMs and GRUs hold their own.

When you see "we used an LSTM" in a 2026 paper, it's *usually* in one of these niches, not as a default.

---

## 11. Vocabulary

You should be able to define each in one sentence by Session 7:

- **Recurrent connection** — a weight that ties `h^(t)` to `h^(t−1)`.
- **Unrolling** — the visualisation of an RNN as a chain of `T` copies of the same cell.
- **BPTT (Back-Propagation Through Time)** — backprop applied to the unrolled graph.
- **Truncated BPTT** — BPTT on a window of `k` steps to bound memory and compute.
- **Vanishing / exploding gradients** — failure modes from the product of `T − 1` Jacobians.
- **Gradient clipping** — capping the gradient norm before the optimiser step.
- **LSTM cell state** — the linear-flow memory channel in an LSTM that mitigates vanishing.
- **Gate** — a sigmoid-output multiplicative mask that controls information flow.
- **GRU** — gated unit without an explicit cell state; lighter than LSTM.
- **Encoder–decoder bottleneck** — the loss of information when a long source is compressed into one vector.

---

## 12. Connections table

| Chapter 10 idea | Returns in |
|---|---|
| Recurrence + parameter sharing | Module 5 (parallel: spatial parameter sharing in CNNs). |
| Vanishing/exploding gradients | Module 4 Reading 2 (Karpathy leak 1 + 3), Module 12 (clipping). |
| LSTM cell state as gradient highway | Module 5 (ResNet identity shortcuts — same idea spatially). |
| Encoder–decoder bottleneck | Module 11 (attention removes the bottleneck; Transformer removes recurrence entirely). |
| Truncated BPTT | Module 13 (memory-budget tricks at scale). |
| BPTT | Module 9 (policy-gradient RL also computes gradients across sequences of decisions). |

---

## 13. Study questions

1. State the exact formula for `h^(t)` in a vanilla RNN. How many learnable matrices are there? How does the parameter count scale with sequence length `T`?
2. Derive `∂h^(t) / ∂h^(0)` for a vanilla RNN with linear activation. Why does this product blow up or shrink geometrically?
3. Why does gradient clipping help with *exploding* gradients but not *vanishing* gradients?
4. Show that when the LSTM forget gate is exactly 1 and the input gate is exactly 0, the cell state is preserved exactly. Why does this address vanishing gradients?
5. A GRU has roughly 3 weight matrices, an LSTM has 4. What does the GRU lose by dropping the explicit cell state?
6. The encoder-decoder model uses a fixed-size `c`. Sketch a sentence where this bottleneck would lose important information.
7. In 2026, when would you *still* reach for an LSTM over a Transformer? Name two specific use cases.

Bring written answers (≤ 1 page) to Lab 7.

---

## 14. Key takeaways

1. RNNs encode the three temporal priors: **sparse interactions across time, parameter sharing across time, variable-length handling.**
2. Their crippling weakness is the **vanishing/exploding gradient** problem from products of Jacobians across long sequences.
3. **Gating** (LSTM, GRU) created a gradient highway through the cell state — *the* cure for vanishing gradients in the RNN era.
4. **Encoder-decoder** with attention removed the fixed-context bottleneck; the Transformer removed the recurrence entirely.
5. In 2026, RNNs survive in **streaming, on-device, and tiny-data** niches. They are no longer the default for language.

---

## 15. Recommended next steps

- **Module 7 Reading 2** — Olah, *Understanding LSTM Networks*. The single best visual explanation of the gating equations in §8.
- Skim **Bahdanau et al. (2015), *Neural Machine Translation by Jointly Learning to Align and Translate*** — the attention paper that fixed the encoder-decoder bottleneck.
- Watch **Karpathy, *The Unreasonable Effectiveness of Recurrent Neural Networks*** (blog post + video) — the canonical char-RNN intuition you'll use in Assignment 7.

> When you can derive `∂h^(t)/∂h^(0)` for a vanilla RNN and explain why an LSTM's cell state escapes that product, you have absorbed Module 7 Reading 1.