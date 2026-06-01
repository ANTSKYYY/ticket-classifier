# Reading 2 — Companion Notes for Mnih et al., DQN (Human-Level Control through Deep RL)

Companion to the DQN paper: experience replay, target network, ε-greedy, what DQN does NOT solve (hard exploration, sample efficiency, continuous actions), the Rainbow follow-up line.

---

# Companion Notes — Mnih et al., *Human-Level Control through Deep Reinforcement Learning* (DQN)

> **Module 9 Reading 2** — A guided study companion to **Mnih et al. (2015), *Human-level control through deep reinforcement learning*** (Nature 518, 529–533). The earlier arXiv version (1312.5602, "*Playing Atari with Deep Reinforcement Learning*", 2013) is openly accessible at **https://arxiv.org/abs/1312.5602**. The Nature 2015 version is the canonical citation; the arXiv version is shorter and reads faster.

DQN is the paper that **made deep reinforcement learning a field**. Read it. The two engineering tricks (experience replay + target network) are the punchline of Reading 2 and of Assignment 9.

---

## 0. How to use this reading

1. **Open the original** (arXiv 1312.5602 is fine; Nature 2015 is denser but has the iconic results figure).
2. **Read it once** — 9 pages on arXiv; budget 45 min.
3. **Come back here** for the DL2026 framing and the connection to Assignment 9.
4. **Answer the study questions in §10** before the Module 9 lab.

---

## 1. Why this paper matters

Before 2013, RL worked on **small** state spaces — Backgammon, gridworlds, low-dim continuous-control tasks. The DQN paper showed for the first time that a **single agent**, with no domain knowledge, could learn to play **49 Atari games** at human level *from raw pixels*. The headline figure (Fig. 3 in the Nature paper) is one of the most cited in deep learning.

The contribution is *not* the architecture (it's a small CNN). The contribution is **two engineering tricks that made the Bellman bootstrap stable under function approximation**:

1. **Experience replay** — store past `(s, a, r, s')` transitions, sample mini-batches from them.
2. **Target network** — use a *delayed copy* of the Q-network for the bootstrap target.

Without these, training diverges. With them, you get a stable agent that learns from raw pixels. This is the canonical 2013–2015 "deep + classical RL = stable" recipe.

---

## 2. The setup

The agent observes **84×84 grayscale pixels** (4 frames stacked — to capture motion). The action set is the joystick's discrete buttons (4–18 per game). The reward is the game score delta, clipped to `{−1, 0, +1}` to make a single agent work across 49 games.

The Q-network is a small CNN:
```
input  (4, 84, 84)
Conv(4→32, 8×8, stride 4)  + ReLU
Conv(32→64, 4×4, stride 2) + ReLU
Conv(64→64, 3×3, stride 1) + ReLU
Linear(7×7×64 → 512) + ReLU
Linear(512 → |A|)            ← Q-values, one per action
```

The output layer produces `Q(s, a)` for **all actions in one forward pass** — a small but important architectural choice. (Naive alternative: feed `s` and `a` together, one forward pass per action. DQN's choice is `|A|×` cheaper.)

---

## 3. The two tricks — the heart of the paper

### 3.1 Experience replay (Lin, 1992 → DQN, 2013)

Store transitions `(s_t, a_t, r_t, s_{t+1})` in a circular buffer of size 1 million. Each training step, sample a random **mini-batch** of 32 transitions and compute the loss on them.

Why this matters:

- **Decorrelates samples.** Consecutive transitions in an Atari episode are highly correlated. SGD assumes i.i.d. samples — without replay, gradients are biased.
- **Reuses data.** Every transition contributes to many gradient updates instead of being thrown away.
- **Stabilises training.** The distribution of training targets changes slowly because the buffer mixes old and new transitions.

Modern variant: **Prioritised Experience Replay** (Schaul et al., 2016) — sample transitions with high TD error more often.

### 3.2 Target network (DQN, 2015 — only in the Nature paper, not the 2013 arXiv)

The naive Q-learning loss is:
```
L = E[(r + γ max_{a'} Q_θ(s', a') − Q_θ(s, a))²]
```
Notice `Q_θ` appears on *both* sides — once as the bootstrap target and once as the model. As `θ` updates, the target moves; gradients chase a moving fence. Training oscillates and often diverges.

The fix: introduce a **separate target network** `Q_{θ⁻}` with weights `θ⁻` that are *copied from `θ` every C steps* (typically `C = 10 000` in Atari). The loss becomes:
```
L = E[(r + γ max_{a'} Q_{θ⁻}(s', a') − Q_θ(s, a))²]
```
Now the target is *stationary* between copies, and standard SGD works.

> If you take away only one thing from Reading 2: the target network is the load-bearing engineering trick that made deep Q-learning stable.

Modern variants:
- **Polyak averaging (soft updates):** `θ⁻ ← τ θ + (1 − τ) θ⁻` with `τ ≈ 0.005` per step. Used in DDPG and SAC.
- **Double DQN** (van Hasselt et al., 2016) — uses the online network to pick the action and the target network to evaluate, fixing a *maximisation bias* in DQN.

---

## 4. The full DQN algorithm

```
Initialise replay buffer D of capacity N
Initialise Q-network θ randomly
Initialise target network θ⁻ ← θ

for episode in 1..M:
    Initialise s_1
    for t in 1..T:
        With prob. ε pick random a_t; else a_t = argmax_a Q_θ(s_t, a)
        Execute a_t, observe r_t and s_{t+1}
        Store (s_t, a_t, r_t, s_{t+1}, done) in D

        Sample mini-batch from D
        target = r + γ max_{a'} Q_{θ⁻}(s', a')    if not done
        target = r                                 if done
        Compute loss = (target − Q_θ(s, a))²
        Backprop and SGD step on θ

        Every C steps: copy θ⁻ ← θ
        Decay ε from 1.0 to 0.05 linearly
```

You will implement this in Assignment 9. The whole thing is ~80 lines of PyTorch.

---

## 5. What the paper does *not* solve

The DQN paper is honest about its limits:

- **Hard exploration games.** *Montezuma's Revenge*, *Pitfall* — DQN scores near zero. Cure: intrinsic motivation (RND, ICM), bootstrap from demonstrations.
- **Sample efficiency.** DQN needs ~50 M frames per game (~ 38 days of game time). Modern algorithms (Rainbow, MuZero) are far better.
- **Stochasticity in the environment.** Atari is deterministic; DQN can over-fit to specific action sequences. The *sticky actions* benchmark (Machado et al., 2018) addresses this.
- **Continuous actions.** DQN's `argmax_a` only works for discrete action spaces. Continuous control needs DDPG / SAC.
- **Catastrophic forgetting.** Sequential learning of many games breaks DQN (the network forgets earlier games).

Each of these motivates a follow-up paper line. You'll spot them in modern RL.

---

## 6. The "Rainbow" line (post-DQN)

DQN is the *baseline* in modern RL; the Rainbow paper (Hessel et al., 2018) combines six post-DQN improvements:

1. **Double DQN** — kills maximisation bias.
2. **Prioritised replay** — sample by TD error.
3. **Dueling network** — separate value and advantage streams.
4. **Multi-step returns** — `n`-step TD target.
5. **Distributional Q-learning (C51)** — predict a distribution over returns, not a scalar.
6. **Noisy networks** — parameter-space exploration.

You won't implement Rainbow in Assignment 9 — but knowing what each component does is the difference between "I trained a DQN" and "I understand the modern variant."

---

## 7. Connections to DL2026

| DQN idea | Returns in / connects to |
|---|---|
| Experience replay (decorrelation, data reuse) | Module 13 (data-pipeline patterns for training stability). |
| Target network as a stationary regression target | Same idea reappears in **EMA teachers** for self-supervised methods (MoCo, BYOL) — Module 11 / SSL literature. |
| ε-greedy exploration | Module 11 (decoding temperatures in LLMs — analogous knob). |
| Frame-stacking for temporal info | Module 7 (RNNs as the alternative — train a *recurrent* DQN with no frame-stack). |
| Discrete-action Q-learning | Diverges from continuous-control descendants (DDPG, SAC); see *Spinning Up*. |

---

## 8. Common confusions the DQN paper resolves

- **"Q-learning is supervised learning."** Yes, *with bootstrapped labels*. The labels move during training — that's why we need a target network.
- **"The replay buffer is just a dataset."** No — it's a *moving* dataset. Old transitions get evicted as the buffer fills. Distribution drift is real.
- **"Larger γ is always better."** Larger γ extends the effective horizon, but increases variance. CartPole works fine at `γ = 0.99`; some long-horizon tasks need `γ = 0.999`.
- **"Reward clipping helps generalisation."** It hides *signal* (large rewards) but standardises across games. For a single-task agent, *do not* clip rewards.

---

## 9. Engineering tips Mnih et al. leave implicit

1. **Use Huber loss**, not MSE. The DQN paper uses MSE; modern implementations use Huber (smooth-L1) because TD errors can be large early in training.
2. **Don't bootstrap past terminal states.** Mask the bootstrap target with `(1 − done)`. Forgetting this is the #1 DQN bug.
3. **Choose `lr ≈ 1e-4`, not `1e-3`.** DQN is fragile to high learning rates; this is part of what makes it harder than supervised training.
4. **Warm-up the replay buffer.** Don't start gradient updates until the buffer has, say, 10 k transitions.
5. **Anneal ε from 1.0 to 0.05 over the first ~10 % of training.** Long exploration tail.
6. **Track average return over a 100-episode rolling window** — single-episode return is noisy.

You will need every one of these in Assignment 9.

---

## 10. Study questions

1. Why does Q-learning's `argmax` action-selection make it **off-policy**? In one sentence, what does the agent's actual behaviour during training have to do with what it learns about?
2. The DQN loss is `(r + γ max_{a'} Q_{θ⁻}(s', a') − Q_θ(s, a))²`. **Why** can't we just use `Q_θ` on both sides? Sketch the divergent dynamic that occurs.
3. State two roles the **replay buffer** plays in DQN. Which one would *not* matter on a fully i.i.d. data source?
4. The paper uses **reward clipping** `r ∈ {−1, 0, +1}`. Name one benefit and one cost.
5. DQN fails on *Montezuma's Revenge*. Name two reasons. What's the simplest modern fix?
6. Why is the target network usually copied every C ≈ 10 000 steps and not every step? What would happen with `C = 1`?
7. A practitioner says "I'll use DQN for continuous joint angles in a robot arm." What's wrong with this?

Bring written answers (≤ 1 page) to Lab 9.

---

## 11. Key takeaways

1. DQN = **Q-learning + CNN + experience replay + target network**.
2. The two engineering tricks (replay + target net) are *the* contribution; the architecture is incidental.
3. DQN is **sample-inefficient**, **fails on hard exploration**, and **only handles discrete actions** — but it works on 49 Atari games from pixels.
4. The Rainbow improvements (Double DQN, prioritised replay, dueling, n-step, distributional, noisy nets) are the modern recipe.
5. The target-network trick reappears in many places — EMA teachers, momentum encoders, slow-moving baselines — wherever you need a *stationary* regression target in a system with moving labels.

---

## 12. Recommended next steps

- Read **van Hasselt et al. (2016), *Deep Reinforcement Learning with Double Q-learning*** — the cleanest fix for DQN's overestimation bias.
- Skim **Hessel et al. (2018), *Rainbow: Combining Improvements in Deep Reinforcement Learning*** — the ablation study you'll cite in your project.
- The **OpenAI Spinning Up curriculum** has a step-by-step DQN walkthrough in PyTorch that complements Assignment 9.
- **Lab 9 first**, then Assignment 9 — the FrozenLake tabular Q-learning builds the intuition for what the neural DQN should be approximating.

> When you can sketch the DQN training loop from memory and explain why the target network exists, you have absorbed Module 9 Reading 2.