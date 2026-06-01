# Reading 2 — Companion Notes for Kingma & Welling, 'Auto-Encoding Variational Bayes' §1–3

Companion to the VAE paper §1–3: amortised variational inference, ELBO derivation, reparameterisation trick, closed-form Gaussian KL, VAE-vs-GAN-vs-diffusion comparison.

---

# Companion Notes — Kingma & Welling, *Auto-Encoding Variational Bayes* (§1–3)

> **Module 8 Reading 2** — A guided study companion to **§1–3** of Kingma and Welling, *Auto-Encoding Variational Bayes* (arXiv 1312.6114, 2013). The paper is freely available at **https://arxiv.org/abs/1312.6114**. These notes are the structured second pass — read the original first.

The VAE paper is dense (8 pages, heavy math). Reading 2's job is to make the algorithm operational: by the end you should be able to derive the ELBO, identify the reparameterisation trick, and write the loss in 5 lines of PyTorch.

---

## 0. How to use this reading

1. **Open the paper.** Read §1 (Introduction), §2 (Method), §3 (Example: VAE) carefully. Skip §4 (related work) and §5 (experiments) for now.
2. **Re-derive the ELBO** on paper. Do not just trust the equations.
3. **Then read these notes** for the DL2026 framing and the connections to Module 10 (GANs / diffusion).
4. **Answer the study questions in §10** before Assignment 8.

---

## 1. Why this paper is foundational

Before 2013, generative modelling with neural networks was hard. Two paths existed: **directed latent-variable models** (Restricted Boltzmann Machines, Helmholtz machines) were probabilistically principled but slow to train; **autoregressive models** were trainable but slow to sample.

Kingma & Welling's contribution: combine **variational inference** (a classical Bayesian technique) with **neural-network amortisation** (use a neural net to *output the variational parameters*). The result is the **VAE** — a model that is fast to train *and* fast to sample.

Two ideas they introduced that endure in 2026:
- **The reparameterisation trick** — used in every modern stochastic-gradient-with-Gaussian-noise method.
- **Amortised inference** — used in every modern probabilistic deep model.

This 8-page paper is one of the highest-impact-per-page papers in deep learning history.

---

## 2. §1 — The set-up

We assume a **latent-variable model**:
```
z ∼ p(z) = N(0, I)                # prior on the latent
x ∼ p(x | z) = N(g_θ(z), σ² I)    # likelihood, mean given by a neural net
```

The model has parameters `θ` (the decoder's weights). Two intractable problems we'd like to solve:

1. **Inference**: given an observed `x`, what's the posterior `p(z | x)`? This is intractable because the marginal `p(x) = ∫ p(x | z) p(z) dz` has no closed form.
2. **Training**: how do we update `θ` to maximise `log p(x)` given a dataset of `x`'s?

Classical variational inference attacks (1) by introducing a *variational posterior* `q(z | x)` that approximates the true posterior, and *iteratively optimising* over `q` per data point. The VAE's move: **let `q(z | x)` be a neural network**, the *encoder*, so the same `q` works for any `x` without re-optimisation. This is **amortised** inference.

---

## 3. §2 — The ELBO

The Evidence Lower BOund. The most important inequality in modern probabilistic deep learning.

Start from `log p(x)`. Multiply and divide by `q(z | x)` inside the integral:

```
log p(x) = log ∫ p(x, z) dz
         = log ∫ q(z | x) · (p(x, z) / q(z | x)) dz
         ≥ ∫ q(z | x) log (p(x, z) / q(z | x)) dz             (Jensen's inequality)
         = E_{z ∼ q} [log p(x, z) − log q(z | x)]
         = E_{z ∼ q} [log p(x | z)] − KL(q(z | x) || p(z))
         =: ELBO(x)
```

**The ELBO has two parts:**
- **Reconstruction term** — `E_{z ∼ q}[log p(x | z)]`. The decoder's job: reconstruct `x` well from the latent.
- **KL term** — `−KL(q(z | x) || p(z))`. The encoder's regulariser: don't drift too far from the prior `N(0, I)`.

> Practical reading: **decoder fits, encoder regularises.** When the KL term dominates, the model under-fits (posterior collapse). When the reconstruction term dominates, the latent space loses smoothness. Modern VAEs (β-VAE, free-bits, KL warm-up) tune this trade-off.

---

## 4. The reparameterisation trick — the load-bearing idea

We want to backprop through `z ∼ q(z | x)`. The problem: sampling is non-differentiable. The trick: rewrite the sample as a deterministic function of (a) the encoder's output and (b) auxiliary noise.

For a Gaussian encoder `q(z | x) = N(μ(x), σ²(x))`, the sample is equivalent to:
```
ε ∼ N(0, I)
z = μ(x) + σ(x) ⊙ ε
```

Now `z` is a **differentiable function** of `μ(x), σ(x)`, and `ε` is just noise (no parameters to backprop through). The gradient of the ELBO flows through `μ` and `σ` cleanly.

**This trick is what made variational deep learning fast.** Before 2013, score-function estimators (REINFORCE) were used; they have much higher variance. The reparameterisation trick reduces variance by orders of magnitude.

---

## 5. §3 — The VAE in code

The paper's §3 is the algorithm. In modern PyTorch:

```python
class VAE(nn.Module):
    def __init__(self, x_dim=784, z_dim=20, hidden=400):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(x_dim, hidden), nn.ReLU())
        self.mu     = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, x_dim))                # logits for Bernoulli x

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mu + std * eps                            # reparameterisation
        x_logits = self.dec(z)
        return x_logits, mu, logvar

def vae_loss(x_logits, x, mu, logvar):
    recon = F.binary_cross_entropy_with_logits(x_logits, x, reduction="sum")
    kld   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon + kld) / x.shape[0]                 # ELBO is maximised → loss is its negative
```

Five new lines (the encoder splits into `mu` and `logvar`, the reparameterised sample, the closed-form KL) is the entire delta from a regular autoencoder. That's it. That's the VAE.

The closed-form KL of a Gaussian against `N(0, I)`:
```
KL(N(μ, σ²) || N(0, 1)) = -½ Σ (1 + log σ² − μ² − σ²)
```
Derive it from the definition; it appears in every VAE notebook you'll ever see.

---

## 6. What the VAE does well — and where it fails

**Strengths:**
- Fast to train, fast to sample. The decoder runs in `O(forward-pass time)`.
- Learnt latent space supports **smooth interpolation** — walk between two points in `z`, decode each, get a morph.
- Probabilistic — every quantity has an interpretation.

**Weaknesses:**
- **Blurry samples.** The Gaussian reconstruction term penalises mean-squared error → averaging over multiple equally-likely outputs → blur. GANs (Module 10) and diffusion models do not have this pathology.
- **Posterior collapse.** When the decoder is too powerful, the model can ignore `z` entirely and use the prior alone. Manifest as `KL ≈ 0` in training logs.
- **Mode mismatch.** A unimodal Gaussian `q(z | x)` cannot capture a truly multimodal posterior.

Variants in the wild:
- **β-VAE** (Higgins et al., 2017) — multiply KL by `β > 1` for *disentanglement*.
- **VQ-VAE** (van den Oord et al., 2017) — discrete codes; basis of Dall-E 1 and many speech-generation models.
- **Conditional VAE** (Sohn et al., 2015) — condition on a label `y` to control what's generated.

---

## 7. Connections to DL2026

| Idea from VAE paper | Returns in |
|---|---|
| ELBO | Reused as the training objective in diffusion models (Module 10). |
| Reparameterisation trick | Module 9 (policy-gradient continuous-control), Module 11 (Gumbel-softmax). |
| Amortised inference | Modern self-supervised models — the encoder is amortised in every BERT-style pre-training. |
| Decoder as `p(x | z)` | The "generator" half of a GAN (Module 10) and the "denoiser" of a diffusion model. |

---

## 8. The two equations you must memorise

Print them on a sticky note:

```
ELBO(x) = E_{z ∼ q(z|x)}[log p(x | z)] − KL(q(z | x) || p(z))

KL(N(μ, σ²) || N(0, 1)) = − ½ Σ (1 + log σ² − μ² − σ²)
```

Every other line of VAE code is derived from these two.

---

## 9. Common confusions Kingma & Welling resolve

- **"The VAE is an autoencoder with sampling at the bottleneck."** True but incomplete. It's an autoencoder *whose loss includes a KL regulariser* — the sampling is just how the regulariser becomes differentiable.
- **"The encoder outputs `z`."** No — it outputs the *parameters* `μ(x), σ(x)` of `q(z | x)`. We then *sample* `z` from this distribution.
- **"The KL term is optional."** Without it, you have a regular AE with random noise injection — the model can ignore the noise and you lose the smooth latent structure.
- **"Posterior collapse means the encoder broke."** No — it means the model found a shortcut. The decoder is *too powerful relative to the data*. Cures: KL warm-up, free bits.

---

## 10. Study questions

1. Derive the ELBO from `log p(x)` using Jensen's inequality. (One page, no shortcuts.)
2. Derive the closed-form KL `KL(N(μ, σ²) || N(0, 1))`. Verify the answer in §8.
3. Explain in 3 sentences why the reparameterisation trick is preferred to a score-function estimator (REINFORCE).
4. The VAE produces blurry images. Why does the Gaussian reconstruction term cause this? What loss change would mitigate it?
5. A practitioner reports `KL → 0` during training. Diagnose (posterior collapse). Propose two cures.
6. Sketch how you would extend the VAE to condition on a class label (turning it into a conditional VAE). What changes in §5's code?
7. The reparameterisation trick assumes `q` is reparameterisable. Why doesn't it work for *discrete* `z`? (Hint: this motivates VQ-VAE and Gumbel-softmax.)

Bring written answers (≤ 1 page) to Assignment 8.

---

## 11. Key takeaways

1. **Variational inference + amortisation = VAE.** The encoder outputs the parameters of `q(z | x)`; the decoder reconstructs from a sampled `z`.
2. **ELBO = reconstruction − KL.** Two terms in tension; the balance is what makes VAEs work or fail.
3. **The reparameterisation trick** is the load-bearing innovation. It made deep variational learning fast.
4. The VAE produces **smooth latent spaces** at the cost of **blurry samples** — explaining the rise of GANs (Module 10) and diffusion models.
5. The two equations of §8 underlie every probabilistic deep generative model in this course.

---

## 12. Recommended next steps

- **Doersch (2016), *Tutorial on Variational Autoencoders*** — the canonical 20-page derivation walk-through.
- **Lilian Weng's blog, "*From Autoencoder to β-VAE*"** — the most readable visual essay on the family.
- **Module 10 readings** — DDPM and GAN papers — both pick up the threads VAEs leave open.
- Re-derive the ELBO from `log p(x)` on a whiteboard. Three minutes well spent.

> When you can write the ELBO, the KL, and the reparameterisation sample from memory, you have absorbed Module 8 Reading 2.