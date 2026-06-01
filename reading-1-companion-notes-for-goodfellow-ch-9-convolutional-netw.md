# Reading 1 — Companion Notes for Goodfellow Ch. 9 (Convolutional Networks)

Structured companion to Goodfellow Ch. 9: the three structural priors (sparse interactions, parameter sharing, equivariance), pooling, conv variants, efficient algorithms, history, with study questions and forward references.

---

# Companion Notes — Goodfellow et al., *Deep Learning*, Chapter 9 (Convolutional Networks)

> **Module 5 Reading 1** — A structured study companion for **Chapter 9** ("Convolutional Networks") of Goodfellow, Bengio, and Courville's *Deep Learning* (MIT Press, 2016). The book is free at **https://www.deeplearningbook.org**. These notes summarise, paraphrase, and connect Chapter 9 to the rest of DL2026 — they are **not** a substitute for reading the original.

---

## 0. How to use this reading

1. **Read the original** at deeplearningbook.org → Chapter 9. It is ~40 pages and unusually dense; budget 2 hours for the first pass with a pen.
2. **Then come back to these notes.** They re-organise the chapter in DL2026 notation (NCHW tensors, examples-batch-first), point out where the textbook is slightly dated, and connect each idea to a later module.
3. **Answer the study questions in §10** before Lab 5.

Notation reminder for this chapter:
- Input tensor `X ∈ ℝ^{B × C_in × H × W}` (batch, channels, height, width).
- Conv kernel `K ∈ ℝ^{C_out × C_in × k_h × k_w}`.
- Output tensor `Y ∈ ℝ^{B × C_out × H' × W'}` with `H', W'` from the arithmetic in Reading 2.

---

## 1. Why CNNs deserve their own chapter

Up through Module 4, you were free to treat your network as "a stack of `nn.Linear` layers". For images, that is brutally wasteful:
- A 224×224 RGB image flattened is `150,528` inputs.
- A single dense hidden layer of width 1,000 has `150 M` parameters.
- A CNN does the *same* job with **roughly 60,000 parameters** in its first layer.

Goodfellow opens Chapter 9 by naming the three structural priors that buy that ~2500× efficiency:

1. **Sparse interactions** — each output depends on a small *local* patch of inputs, not all of them.
2. **Parameter sharing** — the same kernel slides across all positions.
3. **Equivariant representations** — translating the input translates the feature map by the same amount.

Together, these three turn "a feedforward network" into "an image-aware feedforward network". Everything else in Chapter 9 — pooling, padding, strides, transposed convolutions — is implementation detail of these three priors.

---

## 2. §9.1 — The Convolution Operation (and why it's really cross-correlation)

The textbook starts with the continuous formula:
```
(X * K)(t) = ∫ X(τ) K(t − τ) dτ
```
…then switches to the discrete 2-D form that you actually use in PyTorch:
```
Y[i, j] = Σ_m Σ_n  X[i + m, j + n] · K[m, n]
```

**Important honesty Goodfellow makes explicit:** what PyTorch calls "convolution" is mathematically **cross-correlation** — the kernel is *not* flipped. True convolution flips the kernel. For learnable kernels this distinction doesn't matter (the learnt weights absorb the flip), but it explains the confusion when you compare PyTorch to a signal-processing textbook.

### 2.1 The two "free axes" — channels and batch
Two practical extensions of the 1-page formula:

- **Channels.** Conv layers operate on `(C_in, H, W)` tensors with kernels of shape `(C_out, C_in, k_h, k_w)`. Each output channel sums over *all* input channels:
  ```
  Y[c_out, i, j] = Σ_{c_in} Σ_m Σ_n  X[c_in, i+m, j+n] · K[c_out, c_in, m, n] + b[c_out]
  ```
- **Batch.** Add a leading batch dimension; the same kernel applies independently to every image in the batch.

If you can write that sum from memory, you can debug nearly any conv-layer shape error.

---

## 3. §9.2 — The Three Structural Priors (the heart of the chapter)

### 3.1 Sparse interactions
A fully-connected layer connects *every* input to *every* output. A convolution connects an output only to the inputs within its **receptive field** — typically 3×3 or 5×5 in the first layer.

Why this is right for images: a pixel is influenced *mostly* by its neighbours. Long-range structure is built up by stacking many small-receptive-field convolutions: the **effective receptive field grows linearly with depth**.

Practical consequence: the first conv layer of a CNN learns *edge detectors*; the second learns *corners* and *textures*; the fifth learns *part fragments*; the deepest learns *object-level* features. (Module 5 lab visualises this.)

### 3.2 Parameter sharing
The same `3×3` kernel is applied at every spatial position. This is the move that drops parameter count from `O(H W · H W · C)` (dense) to `O(k_h k_w · C_in · C_out)` (conv) — independent of image size. The textbook calls this **tied weights**.

Pedagogically: imagine teaching a dense MLP that "an edge is an edge no matter where it appears". You can't — you'd have to show it every example at every position. The CNN's parameter sharing **bakes that fact into the architecture for free**.

### 3.3 Equivariance to translation
Formally: if `T_v` translates the image by vector `v`, then `Conv(T_v X) = T_v Conv(X)`. The feature map of a translated cat is a translated feature map of the original cat.

This is *not* the same as **invariance**: pooling adds (approximate) invariance, but conv layers alone are only equivariant. Distinguishing equivariance from invariance is one of the most-asked-but-rarely-understood points; the chapter is meticulous about it.

### 3.4 What CNNs are *not* invariant to (a Goodfellow caveat)
- **Scale**: a cat at 50 % the size of training cats is unrecognised unless you augment / multi-scale train.
- **Rotation**: rotated cats are unrecognised unless you augment.
- **Lighting / colour shifts**: addressed by normalisation + augmentation.

These three failure modes motivate **data augmentation** and **multi-scale architectures**. They are also why CLIP-style models (Module 11) needed billions of training pairs.

---

## 4. §9.3 — Pooling

Pooling replaces a small spatial region with a single summary statistic — typically *max* (for activations) or *mean* (for some classification heads).

**Why pool:**
- Adds *approximate* translation invariance over a small window.
- Reduces spatial resolution → cheaper compute downstream.
- Provides robustness to small distortions.

**Two technical points the textbook makes:**
- Pooling is *not* equivalent to a strided convolution, but they often serve the same purpose. Modern CNNs (ResNet, EfficientNet) sometimes replace pooling with stride-2 convolutions for a learnable down-sampler.
- **Global average pooling (GAP)** in the final layer (used by ResNet, MobileNet) replaces the dense classifier head's parameters with a fixed average over the spatial dimensions. This drops parameters and almost always helps regularisation.

---

## 5. §9.4 — Convolution + Pooling as an "Infinitely Strong Prior"

The deepest pedagogical move in the chapter. Goodfellow re-frames the architectural choices as a **strong Bayesian prior**:
- "The function we want to learn must be translation-equivariant" → infinite prior probability of conv-shaped weights, zero probability elsewhere.

This framing is useful because it predicts when CNNs **fail**:
- If the data is *not* translation-equivariant (e.g., a fixed-position face attribute), the prior is wrong and the CNN cannot help.
- If the data has *fewer* spatial symmetries than the prior assumes (text, graphs), a CNN is the wrong tool — and you reach for the Transformer (Module 11) or graph networks.

Carry this with you: **an architecture is a prior.** Pick the prior that matches your data.

---

## 6. §9.5 — Variants of the Basic Convolution

A whirlwind tour of *all* the conv variants. Most appear in code as a single argument to `nn.Conv2d`.

- **Strided convolution.** Skip every `s` positions → down-sample without pooling.
- **Padded convolution** — keep the output size equal to the input by padding zeros (or reflections) around the edges. (Reading 2 derives the exact arithmetic.)
- **Locally-connected layer** — same connectivity as a conv, but **no weight sharing** between positions. Used in old face-recognition systems where features are *not* translation-equivariant.
- **Tiled convolution** — share weights only between *some* positions, rotating through a set of kernels. Rarely used today.
- **Depthwise separable convolution** — each input channel has its own kernel (depthwise), then a 1×1 conv mixes channels (pointwise). The trick behind MobileNet's parameter efficiency.
- **Grouped convolution** — split channels into `g` groups, each with its own kernel. Used in ResNeXt and ShuffleNet.
- **1×1 convolution** — a per-pixel linear mixing of channels; the cheapest way to change `C` without touching the spatial dimensions.

Modern CNNs (EfficientNet, ConvNeXt) are mostly *combinations* of these variants. The 2016 chapter pre-dates them, but the building blocks are all here.

---

## 7. §9.6–9.7 — Structured Outputs and Data Types

### 7.1 Structured outputs (briefly)
Most CNNs we use produce a *vector* (classification) or *one number per pixel* (segmentation, depth). Goodfellow emphasises that the same convolutional backbone can produce structured outputs by tail-changing — e.g., a U-Net for segmentation.

### 7.2 Data types beyond 2-D images
Convolutions are dimension-agnostic. Common cases:
- 1-D conv: audio waveforms, sensor time series, text (sometimes).
- 2-D conv: images.
- 3-D conv: videos (treating time as a third spatial axis) or volumetric medical scans (CT/MRI).

The principles (sparse interactions, parameter sharing, equivariance) carry across dimensions.

---

## 8. §9.8 — Efficient Convolution Algorithms

A short section that is *the* reason your CNNs train in minutes, not days.

- **im2col + GEMM** — reshape the image so each spatial position becomes a row, then a giant matrix multiply does the convolution. This is what PyTorch's CPU backend used historically; it's what cuDNN does under the hood for many GPU paths.
- **FFT-based convolution** — for very large kernels, `conv ⇔ pointwise-multiply in frequency domain`. Asymptotically faster, but constant factors make it useful only when `k_h, k_w` exceed ~9.
- **Winograd's minimal filtering** — algebraically clever way to use ~50 % fewer multiplications for small kernels (typically 3×3, 5×5). cuDNN auto-selects between Winograd / GEMM / FFT based on dimensions.

For students: you don't pick the algorithm; cuDNN does. But knowing they exist explains why benchmarks vary across PyTorch versions and GPU generations.

---

## 9. §9.9–9.11 — Random Features, Neuroscience, History

The chapter ends with three short sections worth skimming:

- **§9.9** Random / unsupervised features. Some early CNN papers showed that *random* convolutions plus a learnt classifier on top are surprisingly competitive on small datasets. Modern self-supervised methods (Module 11 reading: SimCLR, MAE) are the spiritual descendants.
- **§9.10** The neuroscientific basis. Hubel and Wiesel's cat visual-cortex experiments (1962) — simple cells = oriented-edge detectors (the receptive-field idea); complex cells = pooled-over-translation (the invariance idea). CNNs are a *very* loose abstraction of this — don't over-extend the analogy.
- **§9.11** CNNs and the history of deep learning. Fukushima's Neocognitron (1980), LeCun's LeNet-5 (1989, 1998), AlexNet (2012). AlexNet is the inflection point: it beat ImageNet by 10 % and made the rest of deep learning's third wave possible.

---

## 10. Connection table — where each Chapter 9 idea returns in DL2026

| §9 idea | Returns in |
|---------|-----------|
| Receptive field, parameter sharing | Lab 5 (CIFAR-10 CNN), Module 11 (vision Transformers replace this with patches). |
| Pooling | Lab 5, Module 11 (max-pooling is rare in Transformers; GAP survives). |
| 1×1 conv | Lab 5 (channel-mixer), Module 11 (FFN inside Transformer block). |
| Depthwise separable | Module 13 (efficient deployment / MobileNet). |
| Equivariance vs. invariance | Module 9 (RL agents need translation invariance in vision pre-processing). |
| Architecture as prior | Module 11 (Transformer = weaker prior + more data + more compute). |

---

## 11. Common confusions Chapter 9 resolves

- **"Convolution and cross-correlation are the same."** Mathematically no; PyTorch's "conv" is cross-correlation. For learnable kernels it doesn't matter.
- **"Padding=same always exists."** Only for *odd* kernels with stride 1. For even kernels or strided convs, asymmetric padding may be needed.
- **"Pooling makes the network translation-invariant."** Only *approximately*, and only for translations smaller than the pool window. A pooled CNN is still *not* invariant to large translations.
- **"More channels is always better."** No — parameter count is `C_in · C_out · k_h · k_w`. Doubling channels at every layer is a fast way to explode the parameter budget.

---

## 12. Vocabulary

You should be able to define each in one sentence by Session 5:

- **Kernel / filter** — the learnable weight tensor of a conv layer.
- **Receptive field** — the patch of input pixels that influence one output unit.
- **Stride** — step size of the kernel as it slides.
- **Padding** — zeros (or reflections) added around the input.
- **Channel** — one of `C_in` feature maps per spatial position.
- **Feature map** — one of `C_out` 2-D activation grids produced by a conv layer.
- **Equivariance** — output translates with input.
- **Invariance** — output doesn't change under input transformation.
- **Pooling** — local summary statistic (max or average) over a small spatial window.
- **Global average pooling (GAP)** — average over the *whole* spatial dimensions; collapses HW → 1×1.
- **Depthwise separable conv** — depthwise conv (per-channel) followed by a 1×1 pointwise conv.

---

## 13. Study questions — answer in your own words

1. Compute the parameter count of (a) a dense layer mapping a 224×224 RGB image to a 1000-dim vector, and (b) a single 3×3 conv with 32 output channels on the same image. What is the ratio?
2. Show that a 1×1 conv with `C_in = C_out = N` is mathematically identical to a per-pixel dense layer of dimension `N → N`. Why is it still useful?
3. A stack of two 3×3 convolutions has the *same receptive field* as one 5×5 conv. Why is the two-3×3 stack usually preferred?
4. State the difference between equivariance and invariance using the formal definitions in §9.2.3. Give a one-sentence example of each.
5. A practitioner reports a CNN works well on training data but fails on a test image that is a horizontally-flipped version of a training image. Diagnose: which structural prior is being violated? What is the cheapest fix?
6. Why does global average pooling regularise the final classifier head?
7. The textbook says "convolution + pooling is an infinitely strong prior." Give one type of data where this prior is *wrong*.

Bring written answers (≤ 1 page) to Session 5 lab.

---

## 14. Key takeaways

1. CNNs encode three structural priors: **sparse interactions, parameter sharing, equivariance**. Every CNN trick is downstream of those three.
2. **Receptive fields grow with depth**; that's how a small kernel sees a large patch.
3. **Pooling adds approximate invariance**; conv layers alone are only equivariant.
4. **Architecture is a prior.** When the prior matches the data, CNNs are unbeatable per parameter; when it doesn't (text, graphs), use a different architecture.
5. The chapter is the canonical 2-hour read for understanding *why* CNNs work. Module 11 will eventually show you a model that *doesn't* impose this prior — and what that costs.

---

## 15. Recommended next steps

- Read **Module 5 Reading 2** — *A Guide to Convolution Arithmetic* — to nail down the output-shape formulas you will paste into every CNN you ever build.
- Watch **Stanford CS231n Lecture 5 (CNNs)** by Justin Johnson (or Karpathy's original 2016 version) for the canonical 1-hour video version of this chapter.
- Skim **He et al., "Deep Residual Learning for Image Recognition" (2015)** — the cure for the depth problem you'll meet in Lab 5.
- Bookmark **Christopher Olah, "Conv Nets: A Modular Perspective"** — the best visual essay on the chapter's ideas.

> When you can write the `Y[c_out, i, j] = Σ ... K[...]` summation from memory and explain it to someone who has never seen it, you have absorbed §9.1–9.4.