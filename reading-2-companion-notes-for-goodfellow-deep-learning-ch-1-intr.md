# Reading 2 — Companion Notes for Goodfellow, *Deep Learning*, Ch. 1 (Introduction)

Section-by-section guide to read alongside Ch. 1 of the Goodfellow book, including study questions and connections to later DL2026 modules.

---

# Companion Notes — Goodfellow et al., *Deep Learning*, Chapter 1 (Introduction)

> **Module 1 Reading 2** — A guided study companion to read *alongside* Chapter 1 of Goodfellow, Bengio, and Courville's *Deep Learning* (MIT Press, 2016). The book is freely available at **https://www.deeplearningbook.org**. These notes summarise, expand, and connect the chapter to the rest of DL2026 — they are **not** a substitute for reading the original.

---

## 0. How to use this reading

1. **Open the original chapter** at deeplearningbook.org → Chapter 1 ("Introduction"). It is about 25 pages.
2. **Read the original first**, with a pen and a notepad. Mark anything you do not yet understand.
3. **Come back to these notes** and use them as scaffolding: a section-by-section paraphrase, plus pointers to the rest of the course where each idea is picked back up.
4. **Answer the study questions in §8** before Session 2.

Do not paste the original text into your submitted work; cite the page number instead. Goodfellow's book is copyrighted; only short, attributed quotations belong in your project reports.

---

## 1. Why this chapter is the most important you will read all term

Chapter 1 sets up three ideas that you will see again and again in DL2026:

1. **Deep learning is one *answer*, not the *question*.** The question is "how does a machine acquire knowledge?" and the answer "compose simple learned representations into complex ones" is the deep-learning move.
2. **Representations are the central object.** Every architecture in this course is a story about *what kind of representation* a network is good at building.
3. **The current success of deep learning is a function of three trends — data, compute, and algorithms — all crossing thresholds at roughly the same time.** None of the three alone would be enough.

If you internalise only those three ideas, you will read the rest of the book in stereo.

---

## 2. The "knowledge problem" — §1 pp. 1–4

The opening of the chapter argues that early AI attempted to solve hard intuitive problems (vision, language) by *hard-coding* human knowledge, and failed. Knowledge-based systems (Cyc, expert systems of the 1980s) struggled because the world is too rich to enumerate by hand.

The deep-learning move is to **let the machine discover the rules** by extracting features from large amounts of data. Crucially, the *features themselves* are learned, not designed.

Connection to DL2026:
- Reading 2 of Module 2 makes this concrete: classical ML uses hand-engineered features; deep learning learns them.
- This argument is the philosophical justification for Modules 4, 5, 7, 11 — every architecture is *one way* of structuring learned features.

---

## 3. The depth question — §1 pp. 4–8

Why "deep"? The chapter distinguishes:

- **Shallow representations** — a single layer of features (raw pixels, hand-crafted edge filters, bag-of-words). The model sees a flat picture of the world.
- **Deep representations** — features of features of features. Each layer reshapes the input into a representation in which the next layer's job is easier. Edges → textures → motifs → object parts → objects → scenes.

Goodfellow uses two definitions of "depth":
1. **Computational graph depth** — the longest path of operations from input to output.
2. **Probabilistic model depth** — the depth of the dependency graph among latent variables.

These two are not always the same. CNNs are deep in (1) but only moderately so in (2). Modern Transformers stretch both.

Connection to DL2026:
- Module 4: depth-vs-width experiments, vanishing gradients.
- Module 11: depth via stacked self-attention layers.

---

## 4. The "three waves" of neural networks — §1 pp. 12–24

The chapter narrates the history of neural networks as three waves, each riding a different banner:

### 4.1 Wave 1 — Cybernetics (1940s–1960s)
- McCulloch & Pitts (1943) — the artificial neuron as a logical gate.
- Rosenblatt's perceptron (1958) — learnable thresholds, a learning rule.
- The winter: Minsky & Papert (1969) showed perceptrons cannot solve XOR. Funding dried up.

### 4.2 Wave 2 — Connectionism (1980s–mid 1990s)
- Rumelhart, Hinton, Williams (1986) — **backpropagation** for multi-layer networks.
- The winter: deep networks of the era couldn't be trained well (vanishing gradients, slow hardware, small datasets). Support vector machines stole the spotlight in the late 1990s.

### 4.3 Wave 3 — Deep learning (2006–present)
- Hinton's deep belief networks (2006) and the "unsupervised pre-training" idea triggered renewed interest.
- The real turning point: **AlexNet (2012)** beat the ImageNet competition by a wide margin using a deep CNN trained on GPUs.
- Since then: word embeddings (2013), seq2seq (2014), GANs (2014), residual networks (2015), Transformers (2017), GPT (2018-present), AlphaFold (2020-2021).

Lesson to absorb: **the field has died before**. Each winter killed funding and reputations for years. The current rise is real, but humility about its limits is a professional skill.

Connection to DL2026:
- Every architecture we study has a wave-3 birth date. The course is, in chronological order, a tour of the recent history of the field.

---

## 5. Why deep learning works *now* — §1 pp. 16–24

The chapter lists three crossings of threshold:

1. **Data.** ImageNet (14M labelled images), Wikipedia text dumps, CommonCrawl, audio corpora — these did not exist before the mid-2000s. Without them, deep models overfit to tiny benchmarks.
2. **Compute.** GPUs (originally for graphics) turned out to be ideal for the dense matrix multiplications that dominate deep models. By 2012 a single GPU could train a network that would have required a cluster in 2002.
3. **Algorithms.** ReLU activations (2010-2012), dropout (2014), batch normalisation (2015), Adam (2015), residual connections (2015), attention (2017) — each removed a specific barrier to training deeper networks.

Goodfellow's view, paraphrased: *none of the three alone would have been enough*. This is the most important paragraph in the chapter — re-read it.

Connection to DL2026:
- Module 12 covers the algorithmic enablers (regularisation, optimisation).
- Module 13 covers the platform enablers (GPUs, distributed training, mixed precision).

---

## 6. What deep learning is *not* — your reading-between-the-lines task

Chapter 1 is careful to be honest about limitations. As you read, mark the places where the authors hedge. A non-exhaustive list:

- Deep learning **does not reason** the way humans do. It is statistical pattern matching that can be coaxed into reasoning by clever prompting (Modules 11, future courses) but does not have built-in logic.
- Deep learning is **data-hungry**. With a few hundred examples it almost always loses to a tree-based model.
- Deep learning is **brittle**. Small adversarial perturbations can flip predictions. The chapter only hints at this; it becomes a research field of its own.
- Deep learning is **opaque**. We do not, even today, have a satisfying theory of why these models generalise as well as they do. Empiricism rules.

Carrying this scepticism with you is what separates a senior engineer from a hype-driven one.

---

## 7. Key vocabulary introduced in this chapter

You should be able to define each in one sentence by Session 2:

- **Representation learning** — learning, jointly with the task, the features used to perform it.
- **Feature** — any function of the raw input used by the model.
- **Depth** — the longest path of operations from input to output in a computational graph.
- **Multilayer perceptron (MLP)** — the canonical fully-connected deep network.
- **Knowledge base** — a hand-curated set of facts and rules; the *anti-pattern* the chapter argues against.
- **Connectionism** — the second-wave name for neural networks, emphasising distributed representations.
- **Distributed representation** — encoding a concept across many neurons rather than one neuron per concept.
- **Unsupervised pre-training** — the 2006-era trick that helped initialise deep networks; mostly superseded by ReLU + better init.
- **End-to-end learning** — training the entire pipeline (features + classifier) jointly with one loss.

---

## 8. Study questions — answer in your own words

1. Why did knowledge-based AI of the 1980s fail at perceptual tasks?
2. Give two operational definitions of "depth" and a network that scores high on one but not the other.
3. Name two reasons the second wave of neural-network research died, and the specific technical fix from the third wave that addressed each.
4. The chapter argues that data, compute, and algorithms all had to cross thresholds together. Pick one and explain what would happen if only that one had improved.
5. If a manager asks "should we use deep learning for this?" what three properties of the problem would push your answer toward "yes" and which three toward "no"? (Hint: see Module 2's reading on classical vs. deep ML.)
6. Why does the chapter call deep learning a "modern instance of representation learning" rather than the inventor of representation learning?

Bring written answers (1 page max) to Session 2. We will spot-check them during discussion.

---

## 9. Discussion prompts for Session 1
- *"Could the next winter come?"* — what would have to happen to halt the current wave of deep learning?
- *"Where does deep learning still lose to humans?"* — sample-efficient learning, common-sense reasoning, planning, etc.
- *"What problem in your life would you most like to apply deep learning to?"* — get used to thinking about applications; this is how project proposals start.

---

## 10. Key takeaways
1. Deep learning is the latest answer to the much older question of how a machine learns *representations*.
2. The novelty of the third wave is **end-to-end learning of hierarchical representations**, not a new mathematical idea.
3. "Why now?" = data × compute × algorithms.
4. The field has survived two winters. Healthy scepticism is professional, not pessimistic.
5. Chapter 1's vocabulary is foundational; every later chapter builds on it.

---

## 11. Recommended next steps
- Read Chapter 5 (ML basics) before Module 4.
- Watch Goodfellow's *Heroes of Deep Learning* interview (Andrew Ng's series) for a 30-minute oral version of the same history.
- Skim the introduction of Yann LeCun's *Deep Learning* (Nature, 2015) for a complementary 8-page version of this story.
- Bookmark Christopher Olah's blog (colah.github.io) for the best visual intuition writing in the field.

> If, after this reading, you can explain in three sentences why deep learning works now, you have already absorbed the most valuable idea in the chapter.