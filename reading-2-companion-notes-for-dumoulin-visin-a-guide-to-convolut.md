# Reading 2 — Companion Notes for Dumoulin & Visin, 'A Guide to Convolution Arithmetic'

Practical reference for conv output-shape arithmetic: forward, transposed, dilated, and pooled. Includes worked tables, a quick-reference card, and the formulas you will paste into every CNN you build in DL2026.

---

# Companion Notes — Dumoulin & Visin, *A Guide to Convolution Arithmetic for Deep Learning*

> **Module 5 Reading 2** — A guided study companion to **"A guide to convolution arithmetic for deep learning"** by Vincent Dumoulin and Francesco Visin (arXiv 1603.07285). The paper is freely available at **https://arxiv.org/abs/1603.07285** and the *animations* are at **https://github.com/vdumoulin/conv_arithmetic**. The original PDF is ~30 pages of figures and tables — read it once with the animations open in a second window, then come back to these notes for the practical summary you will need on every CNN you ever build.

If Goodfellow Chapter 9 (Reading 1) is *why* CNNs work, Dumoulin & Visin is *how the shapes work out*. Together they cover both halves of "I understand convolutions".

---

## 0. How to use this reading

1. **Skim the animations first.** The GitHub repo has GIFs for each case. Five minutes there is worth twenty minutes of formulas.
2. **Read these notes** to pin down the formulas you will *paste* every time you size a CNN layer.
3. **Memorise §3.1 and §6.1** — the output-size formulas for forward and transposed convolutions. You will reach for them in every later module that touches images (Module 8: autoencoders; Module 10: GAN generators; Module 11: vision Transformers).

You should leave this reading able to predict the output shape of *any* `nn.Conv2d` / `nn.ConvTranspose2d` / `nn.MaxPool2d` call **without running the code**.

---

## 1. Why the shapes matter

Half the bugs in early CNN projects come from "I expected `(B, 64, 28, 28)` and PyTorch gave me `(B, 64, 14, 14)`". The fix is always the same: open the arithmetic table, compute the expected shape, and compare to what PyTorch actually produced. Reading 2 is the table.

The other half come from **strided / transposed / dilated** variants that re-introduce subtle off-by-one errors. We address each in turn.

---

## 2. The four shape parameters

For every conv (or pool), four numbers determine the output shape per spatial axis:

| Symbol | Meaning |
|--------|---------|
| `i` | input spatial size (along this axis) |
| `k` | kernel size |
| `s` | stride |
| `p` | zero-padding (per side; symmetric in PyTorch) |

For 2-D convolutions, you have *two* of each (one per axis). PyTorch usually accepts an `int` (same on both axes) or a `tuple` (different). The formula is applied **independently per axis** — there is no interaction between H and W.

---

## 3. Forward convolution (the formula you'll use most)

### 3.1 Output size — the master formula

```
o = floor((i + 2p − k) / s) + 1
```

**Memorise this.** It is the answer to "what is the output size of a `Conv2d(in, out, k, stride=s, padding=p)` along one spatial axis with input size `i`?"

### 3.2 Worked examples

| `i` | `k` | `s` | `p` | `o` (output) |
|-----|-----|-----|-----|--------------|
| 28  | 3   | 1   | 0   | 26 |
| 28  | 3   | 1   | 1   | 28  ← "same" |
| 28  | 5   | 1   | 2   | 28  ← "same" |
| 28  | 3   | 2   | 1   | 14  ← halve spatial |
| 32  | 3   | 1   | 0   | 30 |
| 32  | 5   | 2   | 0   | 14 |
| 224 | 7   | 2   | 3   | 112 ← ResNet stem |
| 224 | 3   | 1   | 1   | 224 ← VGG block |
| 7   | 1   | 1   | 0   | 7   ← 1×1 conv |

If you can produce the last column from the first four columns in your head, you have absorbed §3.

### 3.3 The "same padding" formula
When `s = 1` and you want `o = i`, the required padding is:
```
p = (k − 1) / 2          # works exactly when k is odd
```
For *even* kernels, PyTorch does not support strictly "same" padding; you must add asymmetric padding manually with `F.pad` first.

### 3.4 The "valid padding" rule
`p = 0`. The output strictly shrinks by `k − 1` per axis (when `s = 1`). Use when you want the conv to only "look at" valid in-image pixels, e.g. at the boundary of an image.

### 3.5 Sanity-check rules
- `o` must be a positive integer. If the floor produces 0 or negative, your conv parameters are wrong.
- `s ≥ k` is unusual but legal — it means you are sub-sampling the input (every `s`-th window).
- The `floor` matters: when `(i + 2p − k)` is not divisible by `s`, some right/bottom edge pixels are *dropped*. PyTorch silently does this; the animations make it obvious.

---

## 4. Pooling

Pooling uses the **same** formula as forward convolution (with no learnable weights). `MaxPool2d(k, s, p)` produces an output of size `floor((i + 2p − k) / s) + 1`.

In practice, the canonical pool is `k = s = 2, p = 0` → output is `floor(i / 2)`. Half the spatial size, no learnable parameters.

> **Trap.** `nn.AvgPool2d(kernel_size=2, stride=2)` gives the same shape but a different (linear) reduction. `nn.AdaptiveAvgPool2d(output_size)` will compute the right kernel/stride to produce *exactly* `output_size` — used at the end of ResNet / MobileNet to make the network input-size-agnostic.

---

## 5. Padding variants

Three named padding strategies are common:

- **No-padding ("valid")** — `p = 0`. The output shrinks by `k − 1` per axis (s=1).
- **Half / same** — pad with `(k − 1)/2`. Output size equals input size (when `s = 1`).
- **Full** — `p = k − 1`. Output size is `i + k − 1` (when `s = 1`). Rare in modern CNNs but appears in some image-deconvolution contexts.

In PyTorch:
- `padding=0` ⇒ "valid".
- `padding=(k-1)//2` ⇒ "same" for odd `k`, stride 1.
- `padding="same"` (PyTorch ≥ 1.9) handles odd kernels; for even kernels or `stride > 1` PyTorch raises an error and tells you to do it manually.

---

## 6. Transposed (also called "fractionally-strided" or "deconvolution") convolutions

### 6.1 The output-size formula

```
o_T = (i − 1) · s − 2p + (k − 1) + output_padding + 1
    = s · (i − 1) + k − 2p + output_padding
```

**`output_padding`** is the new parameter you'll see in `nn.ConvTranspose2d` — it lets you bump the output by 0 or 1 pixel to disambiguate cases where the forward formula would produce ambiguity. (See §6.4.)

### 6.2 What transposed conv is and is not

It is *not* a true inverse of convolution. It is a learnable up-sampler — i.e., a conv that *increases* spatial size. Treat it as "a conv operating on a spatially-inflated input where every input pixel is followed by `s − 1` zeros".

Goodfellow refers to it (correctly) as the **gradient of a strided convolution** with respect to its input — which is why frameworks call it `conv_transpose` and not `deconv`.

### 6.3 Worked examples

| `i` | `k` | `s` | `p` | `output_padding` | `o_T` |
|-----|-----|-----|-----|------------------|------|
|  7  | 3   | 2   | 1   | 1                | 14   |
| 14  | 4   | 2   | 1   | 0                | 28   |
| 28  | 4   | 2   | 1   | 0                | 56   |
|  4  | 4   | 4   | 0   | 0                | 16   |

You will produce all of these in the Module 8 (autoencoder) and Module 10 (GAN generator) labs.

### 6.4 Why `output_padding` exists
Two different *input* sizes can produce the *same* output size under a strided forward conv. The transposed conv is therefore one-to-many; `output_padding` is the parameter that picks which of the candidate output sizes you want.

Rule of thumb: pair every `stride=2, padding=1, kernel=3` forward conv with a `stride=2, padding=1, kernel=3, output_padding=1` transposed conv to recover the original spatial size.

### 6.5 The "checkerboard artefact" warning
Transposed convs *with kernel size not divisible by stride* produce visible **checkerboard artefacts** in generated images. The standard fix is either:
- pick `kernel = 4, stride = 2` (divisible), or
- replace transposed conv with `nn.Upsample(scale_factor=2, mode='nearest')` followed by a `Conv2d` (the so-called "resize-conv" pattern from Module 10's reference snippet).

---

## 7. Dilated (atrous) convolutions

Dilated convolutions introduce a dilation parameter `d` — the kernel is "stretched" by inserting `d − 1` zeros between each kernel element. The effective kernel size becomes `k' = k + (k − 1) · (d − 1) = (k − 1) · d + 1`.

### 7.1 The output-size formula

```
o = floor((i + 2p − k') / s) + 1
  = floor((i + 2p − ((k − 1) · d + 1)) / s) + 1
```

### 7.2 Why use dilation
- Grow the receptive field *without* increasing parameter count.
- Avoid down-sampling: useful in semantic segmentation, where you want the feature map at the original resolution but with a large receptive field.
- Building block of **DeepLab** (semantic segmentation) and **WaveNet** (audio, where dilation grows exponentially with depth).

### 7.3 Worked examples

| `i` | `k` | `s` | `p` | `d` | effective `k'` | `o` |
|-----|-----|-----|-----|-----|---------------|-----|
| 28  | 3   | 1   | 1   | 1   | 3             | 28  |
| 28  | 3   | 1   | 2   | 2   | 5             | 28  ← keeps size, doubled receptive field |
| 28  | 3   | 1   | 4   | 4   | 9             | 28  ← keeps size, 9-pixel receptive field |

---

## 8. The full conv vocabulary in one table

| Operation | Output size formula |
|-----------|--------------------|
| Forward conv / pool | `floor((i + 2p − k) / s) + 1` |
| Forward conv with dilation `d` | `floor((i + 2p − ((k − 1)d + 1)) / s) + 1` |
| Transposed conv | `s(i − 1) + k − 2p + output_padding` |
| Global average pool to `(1, 1)` | `1` |
| Adaptive avg pool to `(H_o, W_o)` | exactly `(H_o, W_o)` |

If you printed these four lines on a sticky note next to your monitor, you would skip 80 % of the CNN shape-mismatch errors people hit in industry.

---

## 9. Putting it together — a worked architecture

Trace the spatial shape through a tiny LeNet-style CNN on MNIST (`B × 1 × 28 × 28`):

```
input:                        (B, 1, 28, 28)
Conv2d( 1, 32, k=3, p=1)  →   (B, 32, 28, 28)        floor((28 + 2 − 3)/1)+1 = 28
MaxPool2d(k=2, s=2)       →   (B, 32, 14, 14)        floor((28 − 2)/2)+1   = 14
Conv2d(32, 64, k=3, p=1)  →   (B, 64, 14, 14)        same
MaxPool2d(k=2, s=2)       →   (B, 64,  7,  7)        floor((14 − 2)/2)+1   =  7
Flatten()                 →   (B, 64*7*7) = (B, 3136)
Linear(3136, 10)          →   (B, 10)
```

Compare this trace to one of your *own* notebooks. The discipline is: every time you write a conv/pool layer, write the expected shape next to it in a comment, and let the comment be the bug-detector.

---

## 10. Common conv-arithmetic mistakes

1. **Off-by-one from `floor`.** When `(i + 2p − k)` isn't divisible by `s`, the boundary pixels drop silently. If you need a specific output size, compute `p` so the division is exact.
2. **Asymmetric padding for even kernels.** PyTorch's `Conv2d(padding=int)` only does symmetric padding. For an even kernel + stride 1 + "same" output, use `F.pad` to add `(k//2 − 1, k//2)` manually.
3. **Forgetting `output_padding` on transposed convs.** Common cause of "my decoder outputs 27×27 instead of 28×28."
4. **Mixing up `padding` and `dilation`.** Padding adds zeros *around the input*; dilation adds zeros *inside the kernel*. Both can be 1 with very different effects.
5. **Computing per-axis arithmetic only once.** When `kernel_size = (3, 5)` you must compute `o_H` and `o_W` separately.
6. **Assuming `(B, C, H, W)` for everything.** Some libraries (TensorFlow) default to `(B, H, W, C)` (NHWC). PyTorch is NCHW. When converting between, the conv arithmetic is identical *per spatial axis*; only the channel-axis position differs.
7. **`stride > k`.** Legal but unusual — the conv literally skips pixels. Often a sign of a bug if accidental.

---

## 11. Quick reference card (cut out, paste on your monitor)

```
Forward conv / pool
  out = floor((i + 2p − k) / s) + 1
  "same" pad (k odd, s=1):  p = (k − 1) / 2

Forward conv with dilation
  out = floor((i + 2p − ((k − 1)d + 1)) / s) + 1

Transposed conv
  out = s(i − 1) + k − 2p + output_padding
```

Print it. Tape it to the wall. Refer to it every time PyTorch surprises you.

---

## 12. Connections to the rest of DL2026

| Reading 2 idea | Returns in |
|----------------|------------|
| Forward conv arithmetic | Lab 5 (CIFAR-10 CNN), Lab 8 (VAE encoder), Lab 11 (vision Transformer patches). |
| Same / valid padding | Module 5 lab (sweep `padding=0` vs. `1` on accuracy). |
| Transposed conv | Module 8 (autoencoder decoder), Module 10 (GAN generator). |
| Dilation | Module 7 (dilated 1-D convs for sequences as an RNN alternative). |
| Adaptive average pool | Lab 5 (CIFAR-10 final classifier). |

---

## 13. Study questions — answer in your own words

1. What is the output spatial size of `Conv2d(3, 64, kernel_size=7, stride=2, padding=3)` applied to a `224 × 224` image? (This is the ResNet stem; verify.)
2. Show that two stacked `3 × 3` convolutions have the same receptive field as one `5 × 5` conv. How many parameters does each cost (assume `C_in = C_out = 64`)?
3. You want a `Conv2d(64, 128, kernel=3, stride=2)` to halve spatial size from `32 × 32` to `16 × 16`. What `padding` value makes the arithmetic exact?
4. A practitioner stacks `ConvTranspose2d(64, 32, kernel=3, stride=2, padding=1)` and observes the output is `(B, 32, 13, 13)` from a `(B, 64, 7, 7)` input. They wanted `(B, 32, 14, 14)`. What single argument fixes the problem?
5. Compute the receptive field of the third 3×3 conv in a stack of three (`s=1, p=1`) starting from a single input pixel. Generalise to `L` stacked 3×3 convs.
6. Why is `nn.AdaptiveAvgPool2d((1, 1))` followed by a `nn.Linear` head a popular pattern? (Hint: input-size independence.)
7. Pick **one** GAN/VAE architecture from any module and trace the spatial-shape transformations top-to-bottom using only the formulas in §3 and §6. Verify against the paper's stated dimensions.

Bring written answers (≤ 1 page) to Session 5.

---

## 14. Key takeaways

1. **One formula** for forward conv: `o = floor((i + 2p − k)/s) + 1`. Memorise it.
2. **One formula** for transposed conv: `o = s(i − 1) + k − 2p + output_padding`. Memorise it.
3. **Pooling uses the same forward formula** as conv, with no learnable weights.
4. **Dilation expands the receptive field for free** — at the cost of skipping intermediate pixels.
5. The four numbers `(i, k, s, p)` plus dilation determine every conv layer's spatial behaviour. Print the formulas, tape them to your monitor, never guess.

---

## 15. Recommended resources

- **The animations** at `https://github.com/vdumoulin/conv_arithmetic` — open in a second window while reading.
- **PyTorch `nn.Conv2d` docs** — the canonical reference for the arithmetic in code.
- Sutskever et al. "*Striving for Simplicity: The All Convolutional Net*" (ICLR 2015) — argues for replacing pooling with stride-2 convolutions; useful background for the §4 discussion.
- Odena et al. "*Deconvolution and Checkerboard Artifacts*" (Distill, 2016) — the visual explanation of why "resize-conv" beats `ConvTranspose` in many image-generation contexts.
- Lab 5 walkthrough notebook — applies every formula in this reading to a CIFAR-10 model end-to-end.

> When you can predict the output shape of any conv/pool/transposed-conv before running the code, you have absorbed Module 5 Reading 2.