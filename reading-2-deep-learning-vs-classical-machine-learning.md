# Reading 2 — Deep Learning vs. Classical Machine Learning

When to reach for a neural network and when classical methods win. Includes a decision flow and a worked side-by-side example.

---

# Deep Learning vs. Classical Machine Learning

> **Module 2 Reading 2** — When to reach for a deep model, and when *not* to. A practitioner's comparison, not an ideological one.

---

## 1. Why this question matters

Deep learning has become so prominent that students often assume **every** ML problem deserves a neural network. It does not. A linear model, a gradient-boosted tree, or even a regular-expression rule may be:
- faster to train,
- easier to debug,
- more accurate on small data,
- regulatory-compliant where deep models are not.

The goal of this reading is to give you a clear, dispassionate decision framework: *given a problem, which family of methods should you start with, and why?*

---

## 2. What we mean by "classical ML"

"Classical machine learning" is an umbrella for methods that **operate on hand-engineered features** and typically have a **small, well-understood hypothesis space**. The list you will actually meet in industry:

| Family | Representative algorithms | Typical use |
|--------|----------------------------|-------------|
| Linear / generalised linear | Linear regression, logistic regression, ridge, lasso, GLMs | Tabular, baselines, interpretability |
| Tree-based | Decision trees, random forests, **XGBoost, LightGBM, CatBoost** | Tabular, structured competitions, mixed-type features |
| Kernel methods | SVM (RBF, polynomial), kernel ridge | Small-data classification, sequences |
| Instance-based | k-NN | Tiny data, embeddings retrieval |
| Probabilistic | Naïve Bayes, Gaussian Mixture Models, HMM | Text baselines, generative modelling |
| Unsupervised | k-means, PCA, t-SNE, UMAP | EDA, dimensionality reduction |

The hallmark is **the human picks the features**. The model only fits coefficients.

---

## 3. What we mean by "deep learning"

Deep learning is the family of methods that **learn a hierarchical representation from raw inputs** by composing many parametric, differentiable transformations and training the whole stack end-to-end with gradient descent.

The key shift is *the features are part of the model*. You do not engineer "edge detectors" — the first conv layer learns them.

Sub-families you will meet this term:
- **MLPs / Feedforward networks** (Module 4)
- **CNNs** for grid-structured data (Module 5)
- **RNNs / LSTMs / GRUs** for sequences (Module 7)
- **Auto-encoders & VAEs** (Module 8)
- **GANs & diffusion models** for generation (Module 10)
- **Transformers** for sequences and beyond (Module 11)
- **Deep RL** (Module 9)

---

## 4. Side-by-side comparison

| Axis | Classical ML | Deep Learning |
|------|--------------|---------------|
| **Feature engineering** | Done by the human; expensive, domain-specific. | Learned end-to-end; the model is the feature extractor. |
| **Data hunger** | Often works with hundreds to thousands of rows. | Usually needs tens of thousands to millions of samples (less if pretrained). |
| **Compute** | CPU is fine; minutes to hours. | GPU/TPU usually required; hours to weeks. |
| **Interpretability** | High for linear/trees (coefficients, feature importance, SHAP). | Low by default; needs special tooling (saliency, attention maps, probing). |
| **Sample efficiency** | Strong on tabular and small-N regimes. | Poor on small data unless transfer learning is used. |
| **Inductive bias** | Encoded by feature engineering + model choice. | Encoded by architecture (CNN = spatial, RNN = temporal, attention = relational). |
| **Hardware footprint at inference** | Tiny (kilobytes). | Often megabytes to gigabytes. |
| **Calibration** | Well-studied; reliable probabilities. | Often over-confident; needs temperature scaling. |
| **Hyperparameter sensitivity** | Moderate; defaults usually work. | High; learning rate, batch size, init, schedule, regularisation all matter. |
| **Reproducibility** | Mostly deterministic given a seed. | Stochastic at every level (init, ordering, hardware, numerics). |

---

## 5. Where each approach actually wins

### 5.1 Cases where classical ML beats deep learning
- **Small tabular data** (< 50k rows). Gradient-boosted trees dominate. The "Tabular Playground" Kaggle history is a near-clean sweep for XGBoost/LightGBM.
- **Strongly heterogeneous features** (mixed numeric, categorical, missing). Trees handle these natively.
- **Strict interpretability requirements** (credit scoring, insurance pricing, healthcare). A logistic regression with monotonic constraints clears regulator review; a 12-layer Transformer does not.
- **Tight inference budgets** (edge IoT, microcontrollers). A 5-coefficient logistic model fits in a few kilobytes.
- **Hard real-time guarantees** (sub-millisecond latency on CPU). A linear model is single-digit microseconds.
- **Cold-start problems with little data** but strong domain knowledge.

### 5.2 Cases where deep learning beats classical ML
- **Perceptual data**: images, audio, video. CNNs and Transformers crush hand-crafted features by huge margins.
- **Language**: translation, summarisation, dialogue. Transformers + pretraining have no classical competitor at scale.
- **Sequential decision-making with large state spaces**: Atari, Go, robotics. Deep RL dominates.
- **Multi-modal problems**: image-text, video-audio. Shared embeddings are a deep-only construct.
- **Generative modelling of complex distributions**: photo-realistic image generation, voice cloning. Diffusion and GANs are deep-only.
- **Problems with abundant unlabeled data**: self-supervised pretraining (BERT, MAE, CLIP) gives deep models a huge sample-efficiency advantage *after* pretraining.

### 5.3 Heuristic decision flow (use this in your project)

```
Is the input perceptual (image / audio / raw text / video) ─┐
   YES → start with a pretrained deep model.                │
   NO  → continue.                                          │
                                                            │
Is the data tabular and well-structured?                    │
   YES → start with gradient boosting (XGBoost/LightGBM).   │
         Use a small MLP / TabTransformer only after        │
         gradient boosting plateaus.                        │
   NO  → continue.                                          │
                                                            │
Is the data sequential with rich structure?                 │
   YES → start with a small Transformer or LSTM,            │
         compare against classical (ARIMA, HMM) baselines.  │
   NO  → fall back to classical ML.                         │
```

**Always have a classical baseline.** If your fancy deep model is not at least *as good* as logistic regression, the deep model is broken, not better.

---

## 6. The "bitter lesson" and the "data lesson"

### 6.1 The Bitter Lesson (Rich Sutton, 2019)
> "General methods that leverage computation are ultimately the most effective." Approaches that bake in human prior knowledge keep getting overtaken by approaches that throw more compute and data at the problem.

This is the **best argument for deep learning at scale**. The wins of AlphaGo, GPT-4, AlphaFold are all stories of compute + data + general learning beating clever prior knowledge.

### 6.2 The Data Lesson (less famous, equally important)
> "On most enterprise problems, you don't have GPT-scale data, so the bitter lesson hasn't kicked in yet." If your dataset is 10k rows of tabular spreadsheets, XGBoost will win. The bitter lesson is asymptotic; many real problems are not.

Holding both lessons in mind simultaneously is the mark of a senior practitioner.

---

## 7. Hybrid approaches (the future, increasingly the present)
- **Deep features into a classical classifier**: extract embeddings from a pretrained CNN/BERT, then train a logistic regression or XGBoost on those embeddings. Common in industry; gives you DL accuracy + classical interpretability.
- **Tree-based ensembles + neural late stages**: Wide & Deep (Google), DeepFM (recommenders).
- **Symbolic + neural**: neuro-symbolic models for reasoning tasks.
- **Distillation**: train a deep model, then compress its knowledge into a small classical/shallow model for deployment.

---

## 8. A side-by-side worked example

**Problem.** Predict whether a customer will churn in the next 30 days. You have 80k rows, 60 columns (numeric + categorical + 3 free-text fields).

**Classical-first approach.**
1. One-hot encode categoricals, TF-IDF the free-text fields.
2. Train LightGBM with 5-fold CV. ~2 minutes on a laptop.
3. AUC: 0.87. SHAP values explain the top 10 drivers.

**Deep approach.**
1. Same encoding for tabular, fine-tune a small BERT for the free-text features.
2. Concatenate embeddings, feed into a 3-layer MLP.
3. AUC: 0.89, after 6 hours on a GPU and a week of hyperparameter tuning.

**Verdict.** The deep model is *slightly* better but costs ~30× more time and money, and is harder to ship. If churn losses are huge (a SaaS company with $10M ARR), the +0.02 AUC may be worth it. If you are a startup, ship LightGBM and revisit later.

This kind of trade-off analysis is **what we expect to see in your project proposal**.

---

## 9. Common mistakes
- **Skipping the baseline.** Always train logistic regression or gradient boosting first.
- **Comparing apples to oranges.** Don't compare a hand-tuned XGBoost to a random-init Transformer with default hyperparameters.
- **Reporting accuracy on imbalanced data.** Use AUC-ROC, F1, or PR-AUC depending on the cost asymmetry.
- **Confusing model complexity with model quality.** A 12-layer Transformer that overfits is worse than a 2-feature logistic model that doesn't.
- **Believing every paper.** Recent literature has documented that with proper tuning, gradient boosting matches or beats deep models on most tabular benchmarks (Grinsztajn et al. 2022 *"Why do tree-based models still outperform deep learning on tabular data?"*).

---

## 10. Key takeaways
1. Deep learning is dominant on perceptual and language data, where features are hard to engineer.
2. Classical ML is dominant on small tabular data, regulated domains, and tight inference budgets.
3. Always train a classical baseline. Use it to (a) verify the deep model adds value, and (b) catch bugs.
4. Hybrid pipelines (deep features → classical head) are often the best of both worlds.
5. The Bitter Lesson explains why DL keeps winning at the frontier; the Data Lesson explains why it doesn't always win in your job.

---

## 11. Recommended resources
- Sutton. *The Bitter Lesson.* (Essay)
- Grinsztajn, Oyallon, Varoquaux. *Why do tree-based models still outperform deep learning on tabular data?* NeurIPS 2022.
- Goodfellow et al. *Deep Learning*. Ch. 1 (what is DL) and Ch. 5 (ML basics).
- Géron. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. Ch. 1–2 for the classical view, Ch. 10–11 for the bridge.
- Domingos. *A Few Useful Things to Know About Machine Learning.* Communications of the ACM 2012 — still the best 9-page overview of how *all* ML approaches generalise.

> The professional question is never "Deep or classical?" — it is "What is the cheapest method that meets the requirement?"