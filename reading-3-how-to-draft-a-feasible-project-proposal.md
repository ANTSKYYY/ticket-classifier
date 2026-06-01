# Reading 3 — How to Draft a Feasible Project Proposal

Blueprint for the Module 2 project proposal: required sections, quality checklist, common mistakes, and a worked sample.

---

# How to Draft a Feasible Project Proposal

> **Module 2 Reading 3** — A blueprint for the project proposal due in Session 4. Read this before you write a word.

---

## 1. Why this matters more than you think

The proposal is **not paperwork**. It is the moment you de-risk the project. A clean proposal can save you from spending two weeks fighting an unworkable dataset, an unrealistic metric, or an unsolvable problem. A messy proposal almost always becomes a messy final report.

In our experience grading roughly 200 deep-learning project proposals, **the projects that earned an A on the final almost all had clean proposals**, and **most of the projects that failed had warning signs in the proposal that no one acted on**. Take it seriously.

---

## 2. The five questions a good proposal answers

A reader (your TA, your instructor, a hiring manager) should be able to extract these five answers in two minutes:

1. **What problem are you solving, in one sentence?**
2. **Who or what benefits if you solve it?**
3. **What data will you train on, and where does it live right now?**
4. **What model and baseline will you compare, and on what metric?**
5. **By when, and what does "done" look like?**

If your draft cannot answer any one of these in plain English, **rewrite before you submit**.

---

## 3. The required sections

We use this structure for every proposal. It maps 1-to-1 onto the grading rubric.

### 3.1 Problem statement (≈ 150 words)
- One paragraph: the real-world problem and why it is interesting.
- Who currently solves it? How well?
- A sentence on what the gap is.
- **Avoid** vague openers like "AI is changing the world…". Start with the user's pain.

**Example (good):** *"Skin-lesion photos taken at primary-care clinics are referred to dermatology specialists with a median wait of 9 weeks; ~78% turn out benign. We aim to train a CNN that flags clearly benign lesions on-device, removing 50% of referrals while keeping false negatives below 1%."*

**Example (bad):** *"We will use deep learning to detect skin cancer."*

### 3.2 Dataset (≈ 200 words)
- **Source(s)** with link or citation.
- **Size**: number of samples, train/val/test split, class balance.
- **Licence**: is it usable for academic work? Will you redistribute? (See §6.)
- **Schema**: input modality, label modality, feature dimensions.
- **Known issues**: label noise, demographic bias, missing-ness, leakage risks.
- **Why this dataset is sufficient** to train a deep model (size, diversity, signal).

If your only candidate dataset has 500 samples and no pretrained backbone exists, **the project is not feasible** — go back to step 1.

### 3.3 Baseline (≈ 100 words)
- The classical or simple model you will compare against.
- The metric on which the comparison is fair (see §3.5).
- A *number* you will treat as the bar: "logistic regression on hand-crafted features hits 0.71 F1; we aim to exceed 0.80."

Without a baseline, your deep model has nothing to prove.

### 3.4 Deep model (≈ 200 words)
- Family (CNN / RNN / Transformer / VAE / GAN / RL agent).
- Specific architecture you start from (e.g., ResNet-18, `bert-base-uncased`, a 2-layer GRU).
- Pretrained weights, if any.
- Loss function and optimisation strategy.
- One *named risk* and how you will mitigate it.

Use **the smallest model that plausibly works**. A team that "scales up if needed" almost always beats a team that "starts at GPT-4 size".

### 3.5 Evaluation (≈ 100 words)
- **Primary metric** — exactly one number that "win" is measured by.
- **Secondary metrics** — calibration, fairness, latency, model size.
- **Failure cases** you will examine qualitatively.
- A *paragraph* of how you will report results in the final paper (table, plot, error analysis).

If you cannot define "success" precisely, you cannot succeed.

### 3.6 Milestones & timeline (≈ 100 words)
Lay out four checkpoints across the 16-session schedule (Modules 2 → 15):

| Week | Milestone |
|------|-----------|
| Session 4 (proposal due) | Dataset acquired, EDA notebook done. |
| Session 7 | Baseline running end-to-end, metric pinned. |
| Session 11 | First deep model trained, beats baseline. |
| Session 14 | Final ablations, error analysis, slides drafted. |
| Session 15 | Final presentation. |

Match each milestone to a *team member* who owns it.

### 3.7 Team & roles (≈ 60 words)
- Names of 3–4 members.
- One sentence per person: what they own. ("Alex — data pipeline & EDA. Priya — model & training loop. Wen — evaluation & writeup. Jordan — slides & demo.")
- This is also where you will be honest with each other if a role looks impossible to fill.

### 3.8 Risks & mitigations (≈ 100 words)
Two or three concrete risks with a planned response:

| Risk | Mitigation |
|------|-----------|
| Dataset turns out to be too small. | Fallback to a public benchmark we have already identified (named here). |
| GPU quota runs out mid-training. | Shrink model to fit on Colab Free; precompute embeddings. |
| Class imbalance hurts metric. | Use focal loss + class-weighted sampling. |

The point is not to predict the future — it is to demonstrate that you have *thought about* the future.

---

## 4. Quality checklist (verify before submission)

- [ ] One-sentence problem statement that mentions the *user* and the *pain*.
- [ ] Dataset is real (linked, downloadable now), large enough, and licensed.
- [ ] Baseline is named, with a target metric and a target number.
- [ ] Deep model is the *smallest* defensible choice for the problem.
- [ ] Primary metric is *one* number.
- [ ] Four dated milestones, each with an owner.
- [ ] Two or more named risks with mitigations.
- [ ] No promises you cannot keep ("state-of-the-art", "publish at NeurIPS").
- [ ] Length: 2 pages PDF. Not 6. Not 1.

---

## 5. Common mistakes (every cohort, every year)

1. **The "I'll figure out the dataset later" proposal.** Treat data as the project's foundation; if you can't link to a downloadable dataset *today*, your proposal is incomplete.
2. **The "we'll train a Transformer from scratch on 10k samples" proposal.** This will fail. Use a pretrained backbone.
3. **The "we'll beat state-of-the-art" proposal.** Don't. Set realistic targets *above the baseline*.
4. **The "everyone does everything" team plan.** Specify owners, or nothing will get done.
5. **The "we'll evaluate qualitatively" proposal.** Define one number, even if it is imperfect.
6. **The 1-page bullet-list proposal.** It is impossible to think clearly in 12 bullets. Write prose.

---

## 6. Ethics, licensing, and reproducibility checklist
- **Personal data**: anonymised? Consent obtained or covered by the dataset licence?
- **Copyright**: are you scraping protected content? If yes, scope down or substitute.
- **Bias audit**: is there any chance your model could disparately affect a protected group? Discuss in §3.5.
- **Reproducibility**: will you publish the seed, code, environment file, and (where licence permits) the data hash?

A two-sentence ethics note that names a specific risk earns more credit than a generic platitude.

---

## 7. A worked sample

**Title:** *"On-device dermatology triage with MobileNetV3"*

**Problem:** Reduce dermatology referrals from primary care by flagging clearly benign skin lesions on-device, cutting wait time by ≥4 weeks for the average patient.

**Dataset:** ISIC 2019 challenge (CC-BY-NC, 25,331 dermoscopy images, 8 classes). 70/15/15 split. Known issues: under-representation of darker skin tones — we will report metrics per Fitzpatrick group as a fairness check.

**Baseline:** Logistic regression on hand-crafted asymmetry-border-colour-diameter ("ABCD") features. Reported AUC ≈ 0.74 on the same split.

**Deep model:** MobileNetV3-Small (≈ 2.5 M parameters, 64-MB on-device footprint) pre-trained on ImageNet, fine-tuned with focal-loss + class-balanced sampling.

**Risk:** Class imbalance and fairness gap. **Mitigation:** weighted sampler + per-group AUC reporting + temperature scaling for calibration.

**Evaluation:** Primary metric: macro-AUC. Target: ≥ 0.85. Secondary: per-Fitzpatrick AUC, on-device inference latency (target ≤ 150 ms on iPhone-class hardware), model size.

**Milestones:** Data and EDA by S4 → baseline by S7 → first deep model by S11 → ablations and slides by S14 → presentation S15.

**Team:** Alex (data + EDA), Priya (model + training), Wen (evaluation + paper), Jordan (slides + demo).

Two pages, every required answer, every named risk. **That's the bar.**

---

## 8. Key takeaways
1. A proposal is risk-reduction, not paperwork.
2. The five mandatory answers: problem, user, data, baseline & metric, deadline.
3. The smallest defensible model is almost always the right starting point.
4. One primary metric. One target number. One baseline to beat.
5. Four milestones, each with one owner.
6. Two named risks, each with a mitigation.
7. Two pages. Prose, not bullets.

---

## 9. Recommended resources
- Goodfellow et al. *Deep Learning*. Ch. 11 (practical methodology).
- Andrew Ng. *Machine Learning Yearning.* Free e-book — Chapters 1–10 on choosing a metric and an error analysis strategy.
- Sculley et al. *Hidden Technical Debt in Machine Learning Systems.* NeurIPS 2015.
- Hugging Face *Datasets* hub — searchable, licence-tagged datasets to ground your proposal.
- Papers With Code → *Datasets* and *Benchmarks* — use these to find a baseline number quickly.

> If your proposal could describe two completely different projects, it is too vague. Tighten it until it could only describe yours.