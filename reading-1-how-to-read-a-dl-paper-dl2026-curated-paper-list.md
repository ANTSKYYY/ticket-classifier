# Reading 1 — How to Read a DL Paper · DL2026 Curated Paper List

Five-Pass Reading Method (Keshav), the eight-question Critique Checklist, plus the eight curated papers each team will choose from for Assignment 6 and Lab 6.

---

# How to Read a Deep-Learning Paper — and the DL2026 Curated Paper List

> **Module 6 Reading 1** — A practical framework for reading research papers in deep learning, plus the curated list of eight candidate papers each team will pick from for Assignment 6 (the Paper Critique) and Lab 6 (Lightning Talks).

This reading has two jobs: (a) teach you a *process* for reading papers, and (b) hand you a menu of options for the next two deliverables.

---

## 0. How to use this reading

1. **Read §1–§4 first.** Five-Pass Reading, the critique checklist, common reading mistakes — these apply to every paper you'll ever read.
2. **Then read §5** to pick your team's paper from the curated list. Each option includes a one-paragraph context, a difficulty rating, and the specific contribution to focus on.
3. **Bring your selection to Session 5** — we will allocate papers there with a quick first-come-first-served sign-up; no two teams may present the same paper.

---

## 1. Why reading papers is a deep-learning skill

By 2026, deep learning publishes more in a single month than a student can absorb in a lifetime. The skill we need is *not* "read every paper" — it's **decide quickly which papers are worth your time, extract the core idea, and judge the evidence**. That skill is what Module 6 teaches.

Three reasons it deserves its own week:

1. **The reading rate matters.** Senior practitioners skim 10–20 papers per week and read 2–3 carefully. Students often try to read every line of every paper and burn out. The Five-Pass method (§2) is the established cure.
2. **The literature is your safety net.** Many bugs you'll hit in Modules 7–13 (vanishing gradients, mode collapse, label noise, distribution shift) have *known fixes* documented in a paper somewhere. Knowing the literature is a debug tool, not a hobby.
3. **You will present and defend a paper.** Lab 6 is a 5-minute talk + 2-minute Q&A. Almost every job interview for an ML role includes a "tell us about a recent paper" question. This is practice for that.

---

## 2. The Five-Pass Reading Method (Keshav, 2007 — adapted)

Srinivasan Keshav's "*How to Read a Paper*" (ACM SIGCOMM CCR, 2007) is the canonical short guide. We adapt it for deep-learning papers:

### Pass 1 — Skim (5 minutes)
**Read:** title, abstract, section headings, all figures, all tables, conclusion.
**Skip:** all body text.
**Output:** can you write down (a) the problem in one sentence, (b) the contribution in one sentence, (c) the headline number? If yes, the paper is worth Pass 2. If no, drop it.

### Pass 2 — Setup (15 minutes)
**Read:** introduction, related-work intro paragraphs, method section's first paragraph, all figures' captions, experimental-setup table.
**Skip:** detailed math, proofs, secondary experiments.
**Output:** a half-page of notes — problem, prior approaches, claimed contribution, datasets, baseline, primary metric, target number.

### Pass 3 — Method (45–60 minutes)
**Read:** the full method section *with a pen and paper*. Work the math forward; re-draw the figures in your own notation.
**Output:** can you state the method in one paragraph in your own words *without* the paper open? If yes, you understand it.

### Pass 4 — Evidence (30 minutes)
**Read:** the experiments section line by line. For *each* number:
- What is the baseline? Is it competitive or a strawman?
- What is the variance (error bars / multiple seeds)?
- Are the ablations honest — do they isolate the variables they claim?
- Could a different dataset show a different story?

**Output:** a list of three things you trust about the result, and three you don't.

### Pass 5 — Critique (15 minutes)
**Re-read:** the abstract with everything you now know.
**Output:** a 100-word verdict — is the claim supported? What would extend this work? What would refute it?

Pass 5's output is roughly the shape of your Assignment 6 critique.

---

## 3. The DL2026 Critique Checklist

When you're reading any paper this term, hold these eight questions ready:

1. **Problem.** What real-world problem (or piece of one) does this paper attack?
2. **Claim.** What does the paper claim to contribute that wasn't true before?
3. **Method.** State the method in one paragraph — algorithm, architecture, data flow.
4. **Evidence.** What is the *primary metric*, the *primary number*, and the *primary baseline*? Are there error bars?
5. **Ablations.** Which design choices did the authors isolate? Which did they not?
6. **Generality.** What's the smallest perturbation (different dataset, different model size, different domain) that might break the claim?
7. **Reproducibility.** Did the authors release code/data? Is the compute budget reasonable for a small lab?
8. **Influence.** What did this paper *enable* in the subsequent literature?

If you can answer all eight, you understand the paper at the level a hiring manager hopes for.

---

## 4. Common reading mistakes (every cohort, every year)

- **Reading linearly from the first word.** Most papers should be skimmed first. Figures and tables carry more signal per minute than the introduction.
- **Trusting the abstract.** Abstracts are marketing. The numbers in the tables are the truth.
- **Confusing "popular" with "important".** Citation count measures attention, not contribution. Many heavily-cited papers turn out to be re-frames of older work.
- **Ignoring the related-work section.** It tells you the contribution's *neighbourhood*. Read it.
- **No notes.** A paper you "read" but did not write about is a paper you will not remember by Friday. Use a notebook.
- **Mistaking complexity for depth.** Some papers are dense because the *math is intricate*; others because the *writing is bad*. Re-read the dense parts twice before assuming the second case.

---

## 5. The curated paper list — pick one with your team

Each team picks **exactly one** of the following eight options. Sign-up is first-come-first-served at the start of Session 5; one paper per team. Each entry below names the official PDF, the difficulty (★ to ★★★★), the specific contribution to focus on, and a "watch out for" line.

> All eight are openly available — search "arXiv: <id>" or click the linked URL.

### A — He et al., 2015. *Deep Residual Learning for Image Recognition.* arXiv:1512.03385
- **What's the deal:** Introduced **residual connections** (`h = h + F(h)`) to train networks 100+ layers deep on ImageNet. Won ILSVRC 2015.
- **Difficulty:** ★★ — short, clean, very readable.
- **Focus on:** the *degradation* problem (deeper isn't always better) and how the residual block solves it. Why does the network choose to *learn nothing* (identity) when adding a layer would hurt?
- **Watch out for:** the paper offers two intuitions for *why* residuals help (gradient flow, identity preserving). Both are debated in later literature — say what *you* think.
- **Connects to:** Module 5 (depth-vs-vanishing problem), Lab 4 (your depth ablation).

### B — Vaswani et al., 2017. *Attention Is All You Need.* arXiv:1706.03762
- **What's the deal:** The **Transformer** architecture. Replaced recurrence with self-attention. Foundation of GPT, BERT, T5, ViT, AlphaFold.
- **Difficulty:** ★★★ — concise but dense. Read the *Annotated Transformer* (harvardnlp.github.io) alongside.
- **Focus on:** the multi-head self-attention block. Why is the `1/√d_k` scaling needed? What does each head learn?
- **Watch out for:** the original paper hides several engineering choices (warm-up schedule, label smoothing, BLEU score) that are *load-bearing*. Modern reproductions have re-explored each.
- **Connects to:** Module 11 (Transformers and NLP).

### C — Mnih et al., 2013/2015. *Playing Atari with Deep Reinforcement Learning / Human-Level Control Through Deep Reinforcement Learning.* arXiv:1312.5602 + Nature 2015
- **What's the deal:** **Deep Q-Networks (DQN)** — combined CNNs with Q-learning to play Atari from raw pixels.
- **Difficulty:** ★★★ — the 2013 arXiv is short; the 2015 Nature version is the canonical citation.
- **Focus on:** experience replay + target networks. Without them, DQN diverges. Why?
- **Watch out for:** "human-level" headline only holds on a subset of games; on hard-exploration games like Montezuma's Revenge, DQN is near zero. Module 9 revisits this.
- **Connects to:** Module 9 (Reinforcement Learning).

### D — Goodfellow et al., 2014. *Generative Adversarial Networks.* arXiv:1406.2661
- **What's the deal:** GANs. Two networks in a minimax game; sampler vs. discriminator.
- **Difficulty:** ★★ — short paper, easy reading; the *training tricks* came later (DCGAN, WGAN).
- **Focus on:** the value function `min_G max_D V(D, G)`, the equilibrium proof, why training is fragile.
- **Watch out for:** the paper claims "the discriminator should always be optimal first". Modern practice does *not* follow this — most GAN papers update D and G roughly equally. Discuss the gap.
- **Connects to:** Module 10 (Generative Adversarial Networks).

### E — Devlin et al., 2018. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* arXiv:1810.04805
- **What's the deal:** **Masked language modelling** pre-training + a single fine-tune-anything paradigm. Re-set the NLP leaderboard.
- **Difficulty:** ★★★ — long, lots of experimental detail.
- **Focus on:** the masked-LM pre-training objective vs. GPT-style left-to-right. Why is bidirectionality so much more sample-efficient?
- **Watch out for:** modern critiques argue the *next-sentence-prediction* objective in BERT was actively unhelpful (RoBERTa dropped it and beat BERT). Address this in your critique.
- **Connects to:** Module 11.

### F — Radford et al., 2021. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* arXiv:2103.00020
- **What's the deal:** Joint image–text contrastive pre-training. Zero-shot ImageNet without ImageNet labels.
- **Difficulty:** ★★★ — long but well-structured; tables are the heart of the contribution.
- **Focus on:** the contrastive loss + the 400 M image-text pair dataset. *Was* the algorithm new, or was it the *scale*?
- **Watch out for:** CLIP's zero-shot accuracy on ImageNet matches a *fully-supervised* ResNet-50 — but CLIP was trained on 400 M pairs. Module 11 reading will revisit the scale vs. method debate.
- **Connects to:** Module 11.

### G — Ho et al., 2020. *Denoising Diffusion Probabilistic Models (DDPM).* arXiv:2006.11239
- **What's the deal:** **Diffusion models** — learn to reverse a fixed noising process. Foundation of Stable Diffusion, Imagen, DALL-E 3.
- **Difficulty:** ★★★★ — dense math; budget extra time.
- **Focus on:** the forward noising schedule + the simplified MSE loss. What is the relationship to score matching?
- **Watch out for:** the original sampling procedure took 1000 steps per image. Subsequent work (DDIM, classifier-free guidance) brought this to 20–50; mention them.
- **Connects to:** Module 10 (generative models; we touch GANs but diffusion is the modern default).

### H — Jumper et al., 2021. *Highly Accurate Protein Structure Prediction with AlphaFold (AlphaFold 2).* Nature 596, 583–589.
- **What's the deal:** Solved the 50-year-old protein-folding problem to a useful approximation.
- **Difficulty:** ★★★★ — biology + DL together. The *supplementary* is denser than the main paper.
- **Focus on:** the Evoformer block + the structure-module's invariant point attention. How does the network bake in *biological* priors?
- **Watch out for:** the *evaluation* is on the CASP14 protein-folding contest; the dataset of multiple-sequence alignments is most of the contribution. Discuss whether the architectural novelty would matter without that data.
- **Connects to:** Module 11 (architecture-as-prior, attention mechanism).

---

## 6. Picking a paper — three rules

1. **Pick something your team *wants* to read.** A lukewarm pick produces a lukewarm critique.
2. **Match the team's strongest skill.** Vision-heavy team → A, F, G. RL-curious → C. NLP-curious → B, E. Math-strong → D, G. Cross-disciplinary → H.
3. **One paper per team across the cohort.** First sign-up wins. Have a backup ready.

---

## 7. Where each paper returns in DL2026

| Paper | Module it reappears in |
|-------|------------------------|
| ResNet (A) | Module 5 (depth), Module 12 (init) |
| Transformer (B) | Module 11 |
| DQN (C) | Module 9 |
| GAN (D) | Module 10 |
| BERT (E) | Module 11 |
| CLIP (F) | Module 11 |
| DDPM (G) | Module 10 |
| AlphaFold-2 (H) | Module 11 (as a case study) |

Picking a paper that *also* reappears later in the course means you'll absorb its content twice — once now, once when you need it.

---

## 8. What you produce next (Assignment 6 + Lab 6)

- **Assignment 6 — Paper Critique (Individual, ≤ 1 page, due before Session 8).** A written critique using the §3 checklist. Each team member writes their *own* — no shared text.
- **Lab 6 — Lightning Talks (Team, presented live in Session 6).** A 5-minute talk + 2-minute Q&A on the team's chosen paper. Slides submitted before Session 6 starts.

The reading expects you to pick a paper *before* Session 5 ends.

---

## 9. Key takeaways

1. The Five-Pass Reading Method is the canonical efficient-reading workflow — use it on every paper you read this term.
2. The DL2026 Critique Checklist (§3) is what you should be able to answer after Pass 4 of any paper.
3. The eight curated papers each illuminate one Module in DL2026 — picking strategically reduces total course load.
4. Reading papers is a *workplace* skill, not an academic ritual.
5. Writing a critique is the cheapest way to test whether you actually understood a paper. If you can't write 600 words, you didn't.

---

## 10. Recommended next steps

- **Read Keshav (2007).** *How to Read a Paper.* 2 pages. Linked from `ccr.sigcomm.org`. Foundational.
- **Skim *The Annotated Transformer*** (harvardnlp.github.io) if you picked paper B — it's the canonical line-by-line guide.
- **Open the *Papers With Code* page of your chosen paper.** Look at the latest leaderboards — does the paper still hold up?
- **Schedule 60 minutes per pass.** Three sessions of focused reading is the typical cost of doing Lab 6 well.

> When you can give the 5-minute version of any deep-learning paper after one focused reading day, you have absorbed Module 6.