# Reading 3 — Skim Guide: Three Illustrative Prior-Year Project Posters

Three example posters with a five-question skim framework, cross-poster pattern table, and a list of common failure modes to avoid.

---

# Skim Guide — Three Illustrative Prior-Year Project Posters

> **Module 1 Reading 3** — A structured walk through three example final-project posters in the style of DL2026 prior years. *These three posters are illustrative composites built by the instructor team to teach you what "good" looks like.* 

The goal of this reading is not to imitate any single poster, but to internalise a **pattern**: how a good poster compresses 8 weeks of work into one A1 sheet that a busy reader can decode in 90 seconds.

---

## 1. How to skim a deep-learning poster

A research poster is **not** a paper printed large. It is a billboard. A skimmer should be able to answer five questions in under two minutes:

1. **What problem?** (1 sentence)
2. **What data?** (1 line + 1 image)
3. **What method?** (1 architecture diagram)
4. **What result?** (1 headline number + 1 table or plot)
5. **What did the team learn?** (3 bullets)

Anything that does not advance one of these five answers is a distraction. As you skim each poster below, run the five-question check yourself before reading the commentary.

A useful trick: cover the body of the poster with your hand and read **only the title, the headline number, and the conclusion bullets**. If those three alone tell a coherent story, the poster is good. If they don't, the poster has buried its lede.

---

## 2. Poster A — "On-Device Skin-Lesion Triage with MobileNetV3"

**Team:** Alex Wei, Priya Kumar, Wen Zhang, Jordan Park · Spring 2025
**Headline number:** *Macro-AUC 0.871 · 60 ms latency on iPhone 13 · 2.5 M parameters*

### Layout
- **Left column (problem & data):** dermatology referral bottleneck graph; 8-way class histogram of ISIC-2019; sample dermoscopy images per class.
- **Centre column (method):** clean MobileNetV3-Small block diagram with the inverted-residual block exploded once; focal-loss equation; class-balanced sampler diagram.
- **Right column (results & analysis):** ROC curve vs. logistic-regression baseline (AUC 0.74); confusion matrix; per-Fitzpatrick-group AUC bar chart highlighting the fairness gap on Type V/VI skin; ablation table showing the contribution of focal loss + balanced sampling.

### What to learn from this poster
1. **It opens with a real-world bottleneck** ("9-week median referral wait"), not a generic "AI is important". The reader immediately knows *why* anyone should care.
2. **The baseline is named and beaten**, with the exact delta visible (Δ AUC = +0.13). This is the right rhetorical move.
3. **Fairness is treated as a first-class result, not an afterthought.** The Fitzpatrick-stratified AUC bar chart is the most memorable visual on the poster, *because it shows a remaining problem rather than hiding it.*
4. **Failure modes are made concrete** with three example mis-classifications and a one-line explanation each. Honesty earns credibility.

### What you could do differently
- The architecture diagram lifts the inverted-residual block directly from the MobileNetV3 paper. Re-drawing it in *your* notation would have been stronger.
- The team did not report on-device energy use; this would have been a natural secondary metric for an "on-device" pitch.

### Five-question skim answers
1. **Problem:** Triage benign skin lesions on-device to reduce dermatology referrals.
2. **Data:** ISIC-2019 (25,331 dermoscopy images, 8 classes).
3. **Method:** Fine-tune MobileNetV3-Small with focal loss and a class-balanced sampler.
4. **Result:** Macro-AUC 0.871 vs. 0.74 logistic baseline; 60 ms inference on an iPhone 13.
5. **Learnings:** Fairness gap on darker skin tones remains; balanced sampling matters more than focal loss in their ablation.

---

## 3. Poster B — "Code-Switching Sentiment Analysis with a Small Transformer"

**Team:** Liu Yifan, Maya Hernandez, Tom O'Brien · Autumn 2024
**Headline number:** *Macro-F1 0.78 on a hand-labelled 4-language Sina Weibo corpus, beating multilingual BERT zero-shot by +0.11.*

### Layout
- **Top banner:** four colour-coded Weibo example posts each mixing Mandarin, Cantonese, English, and pinyin.
- **Left:** data-collection pipeline diagram (scraping → de-duplication → human labelling with Cohen's κ = 0.81 inter-annotator agreement).
- **Centre:** architecture — a 6-layer Transformer encoder with shared byte-level BPE tokeniser, plus three illustrative attention-head visualisations.
- **Right:** results table (their model vs. mBERT zero-shot vs. mBERT fine-tuned vs. logistic on TF-IDF); cross-lingual confusion analysis; "what the attention heads attend to" qualitative panel.

### What to learn from this poster
1. **The problem is anchored in a dataset gap.** "No code-switching benchmark exists in this dialect mix" is a much sharper motivation than "we did NLP".
2. **Data work is given equal weight to modelling.** Half of the poster is about the corpus, including labeller agreement statistics. Reviewers and employers love this — most students under-emphasise data quality.
3. **They report against three baselines**, not one — TF-IDF, mBERT zero-shot, mBERT fine-tuned. This is the gold-standard comparison style.
4. **The attention visualisations are honest.** One panel shows the model attending to emojis in a useful way; another shows it attending to seemingly nothing in a confusing case. Showing both signals integrity.

### What you could do differently
- They trained from scratch on 50k tweets — feasible but tight. A larger run with a pretrained multilingual checkpoint and *then* a custom head would likely have improved Macro-F1 by 3–5 more points.
- The poster is dense — they ran out of room and used 9 pt body text in three places. Plan layout earlier.

### Five-question skim answers
1. **Problem:** Sentiment classification of code-switching social-media posts.
2. **Data:** 50k hand-labelled Weibo posts mixing Mandarin/Cantonese/English/pinyin.
3. **Method:** 6-layer Transformer encoder with byte-level BPE tokenisation.
4. **Result:** Macro-F1 0.78 — beats multilingual BERT zero-shot by +0.11.
5. **Learnings:** Data quality (κ = 0.81) was the largest single contributor; attention heads pick up emoji semantics; tokeniser choice (byte-level) matters more than expected.

---

## 4. Poster C — "Distilling AlphaZero-Style Self-Play into a 30 ms Mobile Chess Engine"

**Team:** Karim Saleh, Lin Mei, Ravi Singh, Sara Ahmed · Spring 2025
**Headline number:** *2,000-Elo on the FICS server with a 4.7 MB model and 30 ms move time on a Pixel 7.*

### Layout
- **Left:** problem — "modern chess engines are powerful but big". A bar chart of "engine size vs. Elo" for Stockfish, LeelaChessZero, and their distilled student.
- **Centre top:** teacher–student diagram: teacher = LeelaChessZero (320 MB), student = a 4.7 MB residual conv net + tiny policy/value heads.
- **Centre bottom:** training-curve plot showing distillation loss vs. self-play games.
- **Right:** Elo progression across distillation rounds; ablation showing the contribution of (a) policy distillation, (b) value distillation, (c) Monte-Carlo-tree-search rollouts at training time.

### What to learn from this poster
1. **The constraint defines the project.** "30 ms / 4.7 MB" is non-negotiable; everything else flows from it. Constraints make projects shippable.
2. **They picked a known-hard area (RL + games) but de-risked it** by using an open-source teacher (LeelaChessZero). The deep technical novelty is the distillation pipeline, not the agent.
3. **The ablation panel is the strongest part.** It is the only place a reader can see *which engineering decision earned each Elo point*. Always have an ablation panel.
4. **They include a screenshot of the deployed app**, with a real game annotated. The reader walks away with a memory of the artefact, not just a number.

### What you could do differently
- They report only Elo on the FICS server; an evaluation against a fixed set of test puzzles would have given a more reproducible baseline.
- A short discussion of opening-book vs. neural play would have rounded out the analysis.

### Five-question skim answers
1. **Problem:** Run a chess engine of useful strength on a phone within 30 ms / 5 MB budget.
2. **Data:** Self-play games generated by LeelaChessZero (teacher).
3. **Method:** Policy + value distillation into a small residual CNN with MCTS at training time.
4. **Result:** 2,000-Elo on FICS, 30 ms / 4.7 MB on a Pixel 7.
5. **Learnings:** Policy distillation alone gets you 80 % of the way; MCTS at training time adds the last 200 Elo; the constraint *makes* the project shippable.

---

## 5. Patterns across all three posters

| Pattern | Poster A | Poster B | Poster C |
|---------|----------|----------|----------|
| Named real-world bottleneck | ✅ | ✅ | ✅ |
| Baseline named & beaten | ✅ | ✅ (three baselines) | ✅ |
| Ablation panel | ✅ | ✅ | ✅ (the strongest panel) |
| Failure-mode panel | ✅ | ✅ | partial |
| One memorable headline number | ✅ | ✅ | ✅ |
| Fairness / ethics / honesty about gaps | ✅ | ✅ | partial |
| Reproducible evaluation | ✅ | ✅ | ⚠️ (FICS Elo is noisy) |

Five out of seven boxes is roughly the bar for an A-grade poster. All three of these would clear it.

---

## 6. What to avoid (from posters we did *not* show)

Common patterns that sank prior-year posters:
1. **No baseline.** A 92 % accuracy means nothing without "92 % vs. what?".
2. **Architecture diagrams copied from the paper.** Re-draw them.
3. **Body text smaller than 18 pt.** Nobody reads it.
4. **A "future work" section longer than the "results" section.** Indicates the project didn't actually finish.
5. **A model the team did not understand.** Reviewers ask one specific question and the team collapses.
6. **A title with no specifics.** "Deep Learning for Healthcare" — could be anything. "Macro-AUC 0.87 on ISIC-2019 with a 2.5 M-parameter MobileNet" is a title.

---

## 7. What you should do *now*

1. Open the three poster summaries above and circle the **headline number** in each.
2. For each poster, write the *opposite* headline — a version that hides the contribution. Compare. This builds your sensitivity to clarity.
3. Before Session 2, sketch a one-paragraph "headline of *your* project". You will not lock it in yet — but starting early is the difference between a great proposal and a rushed one.

---

## 8. Key takeaways
1. A good poster compresses 8 weeks into 5 readable answers.
2. The strongest panels are almost always the ablation and the failure modes.
3. Constraints (latency, size, fairness) make projects shippable and posters memorable.
4. Always have a baseline. Always have an ablation. Always show one failure case.
5. Your project should chase a *named* number against a *named* baseline by a *named* date.

---

## 9. Recommended resources
- *How to give a good research talk* — Patrick Winston, MIT, YouTube.
- *Designing conference posters* — Colin Purrington's online guide.
- ICLR & NeurIPS poster galleries (publicly hosted) — skim 20 of them for visual patterns.
- The CourseWise repository's `prior-year-posters/` folder (to be populated by Session 2) for real DL2026 archive examples.

> Your final-week-of-term self will thank your Session-1 self for studying these three patterns now, not later.