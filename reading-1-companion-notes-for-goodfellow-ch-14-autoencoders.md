# Reading 1 — Companion Notes for Goodfellow Ch. 14 (Autoencoders)

Companion to Goodfellow Ch. 14: undercomplete AEs, regularised variants (sparse / denoising / contractive), stochastic encoders bridging to VAEs, manifold framing, denoising-AE → score connection.

---

# Companion Notes — Goodfellow et al., *Deep Learning*, Chapter 14 (Autoencoders)

> **Module 8 Reading 1** — A guided study companion for **Chapter 14** of Goodfellow, Bengio, Courville's *Deep Learning* (MIT Press, 2016), free at **https://www.deeplearningbook.org**. Read the original first; this reading is the structured second pass.

---

## 0. How to use this reading

1. **Open the original** at deeplearningbook.org → Chapter 14. ~25 pages.
2. **Read it once** with a pen and a notebook.
3. **Come back** to these notes for the DL2026 framing, the connections to Modules 10/11, and the engineering tips Chapter 14 leaves implicit.
4. **Answer the study questions in §10** before Assignment 8.

---

## 1. Why autoencoders deserve their own chapter

An **autoencoder** is a neural network trained to copy its input to its output. That sentence sounds either trivial or pointless until you add the trick: *force the copy to pass through a bottleneck*.

```
x  →  encoder  →  z  →  decoder  →  x̂   (loss = ‖x − x̂‖² or similar)
```

The bottleneck `z` is much smaller than `x`. The model is forced to discard the noise and keep only what the decoder needs to reconstruct. **`z` becomes a compressed, learnt representation of `x`.** That's the whole pitch.

Three families of payoff matter for DL2026:

1. **Dimensionality reduction.** PCA on steroids — non-linear, learnt jointly with the task.
2. **Anomaly detection.** Train on "normal" data → "abnormal" data reconstructs poorly. Lab 8 builds this.
3. **Generative modelling.** Variational autoencoders (Reading 2) turn the deterministic compressor into a probabilistic *sampler* — the bridge from Module 8 to Module 10.

---

## 2. §14.1 — Undercomplete autoencoders

The simplest autoencoder. `dim(z) < dim(x)` — the bottleneck is hard-baked into the architecture.

```
encoder:  z = f_θ(x)             (e.g., 784 → 256 → 32)
decoder:  x̂ = g_φ(z)             (e.g., 32 → 256 → 784)
loss:     L(x, x̂) = ½ ‖x − x̂‖²
```

If the encoder/decoder are **linear** and the loss is **MSE**, the optimum is exactly PCA — the encoder learns the top-`d` principal components. Chapter 14 §14.1 proves this. (Useful intuition: deep AE = "non-linear PCA".)

The trade-off: too small a bottleneck → can't reconstruct; too large → identity learnt (no compression). Modern AEs solve this by adding *regularisation* (§4) instead of relying purely on dimension.

---

## 3. §14.2 — Regularised autoencoders

Three knobs to force the autoencoder to *learn something* even when the bottleneck is large enough to memorise.

### 3.1 Sparse autoencoders
Add a sparsity penalty on `z`:
```
L = reconstruction_loss + λ Σ_i |z_i|
```
or a KL divergence between the average activation and a small target (`ρ ≈ 0.05`). The encoder is forced to use *few* units per example.

### 3.2 Denoising autoencoders
Corrupt the input on the way in (`x → x̃` with Gaussian or masking noise), then reconstruct the *clean* `x`:
```
L = ‖x − g_φ(f_θ(x̃))‖²
```
The AE must learn a *robust* manifold representation — features that survive noise. This trick prefigures modern **masked autoencoders** (MAE, BERT).

### 3.3 Contractive autoencoders
Penalise the encoder's sensitivity to small perturbations:
```
L = reconstruction_loss + λ ‖∂z/∂x‖²_F
```
This is exotic and rarely used today, but the *idea* (penalise the encoder's Jacobian) reappears in modern self-supervised methods.

---

## 4. §14.3 — Representational power, layer size, depth

Two empirical facts the chapter highlights:

1. **A single-layer linear AE = PCA** (proved in §14.1).
2. **Stacking many small layers > one huge layer**, even at matched parameter count. Same depth-vs-width story as Module 4 Reading 1.

In 2026, a typical AE encoder is 4–8 conv layers (for images) or 4–6 Transformer blocks (for text/multimodal). Depth lets the encoder build *compositional* features.

---

## 5. §14.4 — Stochastic encoders and decoders

The set-up for the VAE you'll meet in Reading 2. A *stochastic* encoder doesn't output a vector `z`; it outputs **parameters of a distribution** `q(z | x)`. We then sample `z ∼ q(z | x)` before decoding.

```
encoder:   x → (μ(x), σ(x))      # parameters of a Gaussian
sample:    z ∼ N(μ(x), σ²(x))
decoder:   z → x̂
```

Why stochastic? Two reasons:

1. **Regularisation.** Forcing the encoder to commit to a *region* of latent space, not a single point, prevents memorisation.
2. **Generative modelling.** Once `z` is sampled rather than computed, you can *sample new z's from a prior* and decode them → new examples.

The technical machinery (the reparameterisation trick, the KL term in the ELBO) is the subject of Reading 2.

---

## 6. §14.5 — Denoising autoencoders, in depth

A second pass on §14.2.2: denoising AEs implicitly learn the **gradient of the data density**.

Concretely, if you train an AE to recover `x` from `x + ε`, the optimal decoder satisfies (in the limit of small noise):
```
g_φ(f_θ(x + ε)) ≈ x + σ² ∇_x log p(x)
```
which means the AE estimates `∇_x log p(x)` — the **score function**. This is the seed of *score-based generative models* (Song & Ermon 2019) and *diffusion models* (Ho et al. 2020). When you study Module 10's diffusion variant, you'll come back to this section.

---

## 7. §14.6 — Manifold learning

The chapter's most elegant framing: autoencoders learn the **manifold structure** of the data. Real-world data (images of digits, audio of speech, sentences in English) lies on a low-dimensional manifold embedded in a high-dimensional space. The AE's bottleneck `z` is a coordinate system on that manifold.

Two practical consequences:
- **Linear interpolation in `z` ⇒ smooth changes in `x`.** Walking between two points in latent space produces a smooth morph between the two corresponding images. (Assignment 8's interpolation panel.)
- **Off-manifold inputs reconstruct badly.** This is why AEs work for anomaly detection — anomalies are *off the learnt manifold*.

---

## 8. §14.7 — Applications

The chapter lists six application classes; we care about three in DL2026:

| Application | DL2026 reference |
|---|---|
| Dimensionality reduction / visualisation | Lab 8 (we use it for anomaly detection instead, same machinery) |
| Information retrieval (semantic hashing) | Module 11 (sentence embeddings descend from here) |
| Generative modelling (via VAE / diffusion) | Module 10 + Assignment 8 |

Plus three Goodfellow lists but DL2026 mostly skips: feature learning, image denoising, semi-supervised learning. The first two are *implicit* in our labs; the third is a research topic of its own.

---

## 9. Where Chapter 14 is now dated (2026)

The 2016 chapter pre-dates two huge developments:

1. **Diffusion models.** The score-based view of denoising AEs (§5 above) was the seed; DDPM, DDIM, and Stable Diffusion took it to the production-quality generative modelling of 2022+. Chapter 14 does not anticipate this.
2. **Self-supervised pre-training.** MAE (He et al., 2021), BERT (Devlin et al., 2018), SimCLR (Chen et al., 2020) — these are all *autoencoders with structural priors*. The chapter treats AEs as a tool; modern self-supervision turns them into the dominant **pre-training paradigm**.

When you build a VAE in Assignment 8 and read about diffusion in Module 10, you're walking the path the chapter pointed at.

---

## 10. Vocabulary

- **Autoencoder (AE)** — a neural network trained to reconstruct its input through a bottleneck.
- **Undercomplete AE** — bottleneck dimension < input dimension.
- **Sparse / denoising / contractive AE** — three regularised variants from §14.2.
- **Bottleneck (`z`)** — the compressed latent code.
- **Stochastic encoder** — outputs *parameters* of `q(z | x)` rather than a single `z`.
- **Variational autoencoder (VAE)** — the prototype of §14.4; Reading 2 derives it.
- **Manifold** — the low-dimensional surface in input space on which real data lies.
- **Reparameterisation trick** — the differentiable sampler you'll meet in Reading 2.

---

## 11. Study questions

1. Show that a linear AE with MSE loss and `dim(z) = d` recovers the top-`d` PCA components of the data. Why does adding a non-linearity break the equivalence?
2. A practitioner trains an AE with no regularisation and `dim(z) = dim(x)`. What does the model trivially learn, and why is the result useless?
3. State the optimisation objective of a denoising AE and explain in one sentence why it implicitly learns the score `∇_x log p(x)`.
4. Why does interpolating in latent space `z` produce smooth morphs in `x`-space? When does the trick fail?
5. Sketch the architecture of an autoencoder for anomaly detection on tabular fraud data. Where does the *reconstruction error* enter the scoring rule?
6. Give one task in 2026 where a *Transformer-based masked autoencoder* (MAE / BERT) is preferred over a vanilla AE.

Bring written answers (≤ 1 page) to Session 8 lab.

---

## 12. Key takeaways

1. An autoencoder is `x → bottleneck → x̂`; the trick is the bottleneck.
2. Three regularisation styles — sparse, denoising, contractive — make AEs useful even when the bottleneck is generous.
3. A linear AE recovers PCA; non-linear depth lets it learn a *non-linear manifold*.
4. **Denoising AEs implicitly estimate the score** — the bridge from AEs to diffusion models (Module 10).
5. The chapter's stochastic-encoder section (§14.4) is the doorway to **VAEs** — read it carefully before Reading 2.

---

## 13. Recommended next steps

- **Module 8 Reading 2** — the Kingma & Welling VAE paper.
- Skim **He et al. (2021), *Masked Autoencoders Are Scalable Vision Learners*** — modern AE for self-supervised pre-training.
- Skim **Karpathy's *micrograd*** for the simplest possible AE example in NumPy.

> When you can sketch the four AE variants (vanilla / sparse / denoising / VAE) on a whiteboard and name *what they regularise*, you have absorbed Module 8 Reading 1.