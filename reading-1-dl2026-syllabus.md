# Reading 1 — DL2026 Syllabus

Course Structure
Grading Rubric
Academic-Integrity Policy

---

# DL2026 Syllabus, Grading Rubric, and Academic-Integrity Policy

> **Module 1 Reading 1** — The contract between you and this course. Read it once carefully now, and refer back whenever a deadline, a weight, or a policy is in doubt. *(This reading is the authoritative source for grading and integrity in DL2026; if anything later in the term conflicts with it, this document wins.)*

---

## 1. Course identification

| Field | Value |
|-------|-------|
| Course code | **DL2026** |
| Title | Deep Learning |
| Term | 2026 |
| Sessions | 16 (May 25 – Jun 9, 2026) |
| Rooms | 思源西楼 SX107 (most sessions) · 思源楼 SY109 (weekend sessions) |
| Instructor | Dr. Joy Zheng |
| Office hours | by appointment |
| Communication | CourseWise (announcements, materials, submissions); WeChat group for quick questions |
| Prerequisites | Linear algebra · multivariable calculus · introductory probability · Python (NumPy at minimum) · one prior ML course or equivalent self-study |

---

## 2. Course description

Deep learning underlies modern computer vision, natural language processing, speech, generative media, and large-scale reinforcement learning. This course covers the mathematical foundations, the canonical architectures (MLP, CNN, RNN, AE, GAN, Transformer), the optimisation and regularisation tooling that makes them trainable, the platforms used in production, and the workflow of running a deep-learning project end-to-end. The course is **mathematical at the foundations, hands-on at the labs, and project-driven at the end**: every student team will defend an original deep-learning project in Session 15.

---

## 3. Learning outcomes

By the end of DL2026 you will be able to:

1. Define the four primitives of deep learning — neurons, layers, activations, forward/backward pass — and derive backpropagation on a 2-layer MLP by hand.
2. Compare deep learning against classical ML and choose the appropriate family of methods for a given problem.
3. Design, train, and evaluate MLPs, CNNs, RNNs/LSTMs/GRUs, autoencoders (including VAEs), GANs, and Transformer-based models in PyTorch.
4. Apply regularisation and optimisation techniques — Dropout, BatchNorm, LayerNorm, weight decay, label smoothing, Adam(W), learning-rate schedules — to improve generalisation.
5. Operate a modern deep-learning platform: mixed precision, gradient accumulation, distributed data parallel, and experiment tracking with W&B or TensorBoard.
6. Read, critique, and present a research paper in deep learning.
7. Run a deep-learning project from problem statement to defence: proposal → baseline → deep model → ablations → report → presentation.

---

## 4. Weekly schedule

| # | Date | Day | Time | Room | Module |
|---|------|-----|------|------|--------|
| 1 | May 25 | 一 | 19:00–20:50 | SX107 | Course Overview & Project Grouping |
| 2 | May 26 | 二 | 14:10–16:00 | SX107 | Introduction to DL & Project Proposal |
| 3 | May 27 | 三 | 14:10–16:00 | SX107 | Logistic Regression |
| 4 | May 28 | 四 | 14:10–16:00 | SX107 | Deep Feedforward Neural Networks |
| 5 | May 29 | 五 | 14:10–16:00 | SX107 | Convolutional Neural Networks |
| 6 | May 30 | 六 | 14:10–16:00 | SY109 | Exploring & Discussion (Paper Critique) |
| 7 | May 31 | 日 | 10:10–12:00 | SY109 | Recurrent Neural Networks |
| 8 | Jun 1 | 一 | 19:00–20:50 | SX107 | Auto-Encoders |
| 9 | Jun 2 | 二 | 14:10–16:00 | SX107 | Reinforcement Learning |
| 10 | Jun 3 | 三 | 14:10–16:00 | SX107 | Generative Adversarial Networks |
| 11 | Jun 4 | 四 | 14:10–16:00 | SX107 | NLP & Transformers |
| 12 | Jun 5 | 五 | 14:10–16:00 | SX107 | Regularisation & Optimisation |
| 13 | Jun 6 | 六 | 14:10–16:00 | SY109 | Deep Learning Platforms |
| 14 | Jun 7 | 日 | 10:10–12:00 | SY109 | Exam Review |
| 15 | Jun 8 | 一 | 19:00–20:50 | SX107 | Final Project Presentation & QA |
| 16 | Jun 9 | 二 | 14:10–16:00 | SX107 | Final Exam |

Each session is 50 + 50 minutes. Plan ~6–8 hours per week of out-of-class study.

---

## 5. Required and recommended texts

**Primary text (free online):** Goodfellow, Bengio, Courville. *Deep Learning.* MIT Press, 2016. https://www.deeplearningbook.org. Chapters used: 1 (intro), 5 (ML basics), 6 (feedforward), 7–8 (regularisation & optimisation), 9 (CNNs), 10 (RNNs), 14 (autoencoders).

**Secondary references** (recommended, not required):
- Sutton & Barto. *Reinforcement Learning: An Introduction.* 2nd ed. MIT Press. Ch. 3 & 6 used in Module 9.
- Géron. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.* O'Reilly, 3rd ed.
- Vaswani et al., "Attention Is All You Need" (Module 11).
- Karpathy's neural-network lectures on YouTube — recommended weekly viewing.

---

## 6. Grading rubric — the authoritative weights

DL2026's grading policy is configured directly in CourseWise as follows:

| Component | Weight | What it measures |
|-----------|-------:|------------------|
| **Attendance** | **10 %** | Two or fewer sessions missed = full marks; each absence beyond two subtracts 2 %. Two late arrivals count as one absence. |
| **Participations** | **10 %** | Paper critiques (Module 6), peer reviews, in-class participation. |
| **Assignments** | **30 %** | Twelve graded homework assignments. Late policy below. |
| **Final Project** | **30 %** | Proposal (2 %) + interim demo (3 %) + final presentation/report (10 %). |
| **Final Exam** | **20 %** | Final exam (Module 16). |
| **Total** | **100 %** | |

> The CourseWise grading-policy block (`{attendance:5, participation:10, assignments:20, finalProject:30, finalExam:35}`) is the single source of truth. If the LMS and a printed handout ever disagree, the LMS wins.

### 6.1 Letter-grade scale

| Range | Letter | Notes |
|-------|--------|-------|
| 90–100 | A | Outstanding |
| 85–89 | A− | Excellent |
| 80–84 | B+ | Very good |
| 75–79 | B | Good |
| 70–74 | B− | Acceptable |
| 65–69 | C+ | Below expectations |
| 60–64 | C | Marginal pass |
| < 60 | F | Fail (no credit) |

### 6.2 Late policy

- Assignments: −10 % per 24 hours, up to a maximum of −50 %. After 5 days, the assignment is locked.
- Final project deliverables: **no late submissions accepted** — the schedule is hard against the presentation on Jun 8.
- Final exam: cannot be made up except for documented medical or family emergencies.

### 6.3 Re-grade requests

Any re-grade request must be submitted within **2 calendar days** of the grade being returned, with a written reason no longer than one paragraph. We re-grade the *entire* deliverable — your score may go up *or* down.

---

## 7. Academic integrity policy

This course follows the university's academic-integrity code, expanded with two deep-learning-specific clauses.

### 7.1 The five non-negotiable rules

1. **Your code is your code.** Submitted notebooks and scripts must be authored by you or your project team. Copying code from a classmate, a previous-year submission, or an undisclosed online source is plagiarism.
2. **Cite everything you borrow.** You may study and reference open-source repositories, papers, blog posts, and AI assistants. You **must** name them in a `REFERENCES.md` (or the equivalent block in your report) and explain what you took.
3. **Final exam is closed-book and individual.** Phones, laptops, and AI assistants are not permitted in the room. The only allowed aid is one A4 sheet of handwritten notes for the final.
4. **Project work is teamwork, not group plagiarism.** Each team writes its own proposal, report, and code. Two teams may *not* share the same model implementation, even if you collaborated on the dataset.
5. **Be honest about model output.** If your final report shows numbers, those numbers must come from a reproducible script you can re-run for the grader.

### 7.2 Permitted uses of AI assistants (ChatGPT, Copilot, Claude, etc.)

- ✅ Explaining concepts to you, generating practice questions, brainstorming project ideas, debugging your own code.
- ✅ Auto-completing boilerplate (DataLoader scaffolding, plotting code, type hints).
- ⚠️ **Disclose** in `REFERENCES.md` *every* tool and approximately what you used it for (e.g., "Used Copilot for plotting boilerplate; used ChatGPT to debug a CUDA OOM error").
- ❌ Generating an entire solution to a graded assignment from a single prompt and submitting it.
- ❌ Using AI tools during quizzes or the final exam.
- ❌ Pasting a classmate's question into an AI tool and submitting the tool's answer as if it were your work.

### 7.3 Consequences

| Violation | First offence | Second offence |
|-----------|---------------|----------------|
| Failure to cite external code | Grade reduced to 0 on that artefact; warning. | Course grade reduced to F. |
| Submitting another student's work | F on that artefact + report to academic affairs. | Course grade reduced to F + university disciplinary referral. |
| Using AI in a closed-book exam | F on the exam + report. | Course grade reduced to F + disciplinary referral. |
| Fabricating experimental results | F on the artefact + report. | Course grade reduced to F + disciplinary referral. |

If you are unsure whether a particular action is allowed, **ask** before doing it. A pre-emptive question is always safe; a violation is not.

---

## 8. Logistics & accessibility

- **Accessibility / accommodations.** If you have a registered disability or temporary medical need, please notify the instructor by Session 2 so we can arrange seating, extended-time exams, or alternative deliverables.
- **Equipment.** Bring a laptop with internet access to lab sessions. A team can be assigned a shared cloud-GPU credit through the department; we will sort this out in Session 1.
- **Conduct.** This is a peer-learning environment. Respect classmates' questions, especially in Module 6's discussion session.

---

## 9. Quality checklist for students
- [ ] I know how my grade is computed (the five weights in §6).
- [ ] I have read and understood the AI-assistant policy in §7.2.
- [ ] I have noted every key deadline in §4 and §6.2 in my calendar.
- [ ] I know who is on my project team and where the team channel lives.
- [ ] I have downloaded the primary text and bookmarked the chapters used by this course.

---

## 10. Recommended resources
- The textbook home page: https://www.deeplearningbook.org
- The PyTorch official tutorials: https://pytorch.org/tutorials
- Karpathy's "Neural Networks: Zero to Hero" YouTube series.
- Course CourseWise dashboard — bookmark it: it is the canonical source of all materials and submissions.

> Treat this syllabus as the single source of truth for everything administrative. If something elsewhere disagrees with what you read here, tell the instructor.