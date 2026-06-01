# Reading 1 — Companion Notes for Sutton & Barto Ch. 3 (MDPs) + Ch. 6 (TD)

Companion to Sutton & Barto §3+§6: MDP tuple, value functions, Bellman equations, TD(0), SARSA, Q-learning, ε-greedy, convergence conditions, and the bridge to function approximation.

---

# Companion Notes — Sutton & Barto, *Reinforcement Learning: An Introduction*, Ch. 3 (MDPs) & Ch. 6 (TD)

> **Module 9 Reading 1** — A guided study companion for **Chapter 3 (Finite Markov Decision Processes)** and **Chapter 6 (Temporal-Difference Learning)** of Sutton and Barto's *Reinforcement Learning: An Introduction* (2nd ed., 2018). The book is **freely available** at **http://incompleteideas.net/book/the-book-2nd.html**. Read the original first; this reading is the structured second pass for DL2026.

The Sutton-Barto book is the canonical RL textbook. Read it. Even the parts of Chapter 3 that feel like notation overhead pay off when you train your first DQN in Assignment 9.

---

## 0. How to use this reading

1. **Open the original** at incompleteideas.net → 2nd edition PDF → Chapter 3 and Chapter 6. ~70 pages combined.
2. **Read each chapter once**, with the example problems worked.
3. **Come back to these notes** for the DL2026 framing, the deep-RL bridge, and the connections to the DQN paper (Reading 2).
4. **Answer the study questions in §10** before the Module 9 lab.

---

## 1. Why these two chapters

Chapter 3 sets up the **mathematical object** all of RL works on — the Markov Decision Process (MDP). Without it, deep RL is just "supervised learning with weird losses". With it, you can reason about what an agent should *want*.

Chapter 6 introduces **Temporal-Difference (TD) learning** — the family of algorithms that includes Q-learning, the backbone of DQN. Everything you do in Lab 9 (tabular Q-learning) and Assignment 9 (Deep Q-Network) is TD learning at heart.

Chapters 4 and 5 (DP and Monte Carlo) are interesting but optional for DL2026. Skim if curious.

---

## 2. §3.1 — The agent-environment loop

The picture you must draw on the whiteboard once a year:

```
                ┌───────────────────────┐
       state    │                       │
   ┌──────────► │   ENVIRONMENT         │
   │            │                       │
   │   reward,  └─────────┬─────────────┘
   │   next       action  │
   │   state            ◄─┘
   │                       ▲
   │                       │
   │                       │
   │   ┌─────────────────────────────┐
   └───┤   AGENT (policy π)          │
       └─────────────────────────────┘
```

The loop:
1. Agent observes state `s_t`.
2. Agent picks action `a_t = π(s_t)`.
3. Environment returns reward `r_{t+1}` and next state `s_{t+1}`.
4. Repeat.

The agent's job: pick a policy `π` that maximises *expected cumulative reward*.

---

## 3. §3.2–3.3 — Goals, returns, episodes

### 3.1 Goals: the reward hypothesis
Sutton & Barto state the **reward hypothesis** plainly: any goal can be encoded as the maximisation of expected cumulative reward. (This is a bold claim; in practice, reward design is itself an art — see the AlphaGo paper's careful shaping of "win" vs. "intermediate position quality".)

### 3.2 Returns
The **return** at time `t`:
```
G_t = R_{t+1} + γ R_{t+2} + γ² R_{t+3} + ... = Σ_{k=0..∞} γ^k R_{t+k+1}
```
where `γ ∈ [0, 1)` is the **discount factor**.

Why discount?
- *Mathematical convergence* of infinite sums.
- *Mortality* — agents don't live forever; future rewards are uncertain.
- *Modelling preference* — humans typically prefer near rewards to identical far rewards.

In practice, `γ = 0.99` is the workhorse. CartPole's effective horizon at `γ = 0.99` is `1 / (1 − γ) = 100` steps.

### 3.3 Episodes
- **Episodic tasks** — terminate at a *terminal state* (e.g., CartPole falls; the game ends).
- **Continuing tasks** — go on forever (e.g., a server scheduler).

For episodic tasks, `G_t` is a finite sum (until termination). For continuing tasks, `γ < 1` is needed for convergence.

---

## 4. §3.4–3.6 — The MDP, policies, value functions

### 4.1 The MDP tuple
A finite Markov Decision Process is `(S, A, P, R, γ)`:
- `S` — state space.
- `A` — action space.
- `P(s' | s, a)` — transition probabilities.
- `R(s, a, s')` — reward function (sometimes a distribution).
- `γ` — discount.

The **Markov property**: the next state depends *only* on the current state and action, not on history. This is critical — it lets us replace "history" with "state".

### 4.2 Policy
A policy `π(a | s)` is a probability distribution over actions given the state. Two flavours:
- **Deterministic**: `π(s) = a` (one action per state).
- **Stochastic**: `π(a | s)` (a distribution).

### 4.3 Value functions
Two important value functions:
- **State value** `v_π(s) = E_π[G_t | S_t = s]` — expected return starting in state `s` and following `π`.
- **Action value (Q-function)** `q_π(s, a) = E_π[G_t | S_t = s, A_t = a]` — expected return from state `s`, taking action `a`, then following `π`.

The Q-function is what DQN learns. **Internalise it.**

---

## 5. §3.5 — Bellman equations

The recursive identity that powers all of RL:

```
v_π(s) = Σ_a π(a | s) Σ_{s', r} P(s', r | s, a) [r + γ v_π(s')]
q_π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a' | s') q_π(s', a')]
```

The Bellman equation says: the value of `s` equals the expected immediate reward + the discounted value of the *next* state. It's the recursion that lets us bootstrap.

### 5.1 The Bellman *optimality* equations
For the optimal policy `π*` and optimal Q-function `q* = q_{π*}`:
```
q*(s, a) = E[R_{t+1} + γ max_{a'} q*(S_{t+1}, a')]
```

The optimal Q-function satisfies a Bellman equation **where the next-state expectation uses `max_a`** instead of the policy's expectation. This is *the* equation DQN learns.

> **The TL;DR:** DQN's loss is just "make `Q_θ(s, a)` look like `r + γ max_{a'} Q_θ(s', a')`". Everything else is plumbing.

---

## 6. §6.1–6.2 — Temporal-Difference Learning

### 6.1 The TD(0) idea
Suppose we want to learn `v_π(s)`. Two ways to update our estimate after seeing one transition `(s_t, a_t, r_{t+1}, s_{t+1})`:

- **Monte Carlo**: wait until the episode ends, observe the actual return `G_t`, update `V(s_t) ← V(s_t) + α (G_t − V(s_t))`. High variance, low bias.
- **TD(0)**: bootstrap immediately using the current estimate of the next state's value:
```
V(s_t) ← V(s_t) + α (r_{t+1} + γ V(s_{t+1}) − V(s_t))
```
The term `δ_t = r_{t+1} + γ V(s_{t+1}) − V(s_t)` is the **TD error**. Low variance (one step), higher bias (we trust our own bootstrap).

In practice, TD methods dominate RL because the variance reduction is huge.

### 6.2 SARSA — on-policy TD control
SARSA updates `Q(s_t, a_t)` toward `r + γ Q(s_{t+1}, a_{t+1})`, where `a_{t+1}` is the action the agent *actually took* (sampled from `π`). It learns the value of *the policy being followed*.

### 6.3 Q-learning — off-policy TD control
Q-learning updates `Q(s_t, a_t)` toward `r + γ max_{a'} Q(s_{t+1}, a')`. The `max` makes Q-learning **off-policy** — it learns the value of the *greedy* policy, regardless of what the agent actually did.

**Q-learning update (the equation you'll paste in Lab 9):**
```
Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') − Q(s, a)]
```

---

## 7. §6.4 — Q-learning convergence

Q-learning converges to `q*` *with probability 1* under three conditions:
1. Every state-action pair is visited infinitely often.
2. The learning rate `α` decays appropriately (`Σ α = ∞, Σ α² < ∞`).
3. The Q-function is represented *tabularly* (one entry per `(s, a)` pair).

Condition 3 fails for any interesting problem — CartPole has continuous states, so you can't store a Q-value per state. **Replacing the table with a neural network is the DQN move** (Reading 2).

This loss of tabular guarantees is *the* reason deep RL is hard. The Q-learning convergence theorem dies; we have to fix the resulting instabilities with engineering tricks (experience replay, target networks — Reading 2 §3).

---

## 8. §6.5 — Exploration: the ε-greedy policy

Pure greedy action selection (always pick `argmax_a Q(s, a)`) starves the model of exploration — it never learns about actions it hasn't tried.

The simplest fix: **ε-greedy**.
- With probability `1 − ε`, pick `argmax_a Q(s, a)`.
- With probability `ε`, pick a random action.

`ε` typically decays from 1.0 (full exploration at start) to ~0.05 (mostly exploitative late in training). This is what you'll use in Lab 9 + Assignment 9.

Better explorers (UCB, Thompson sampling, intrinsic curiosity) exist; ε-greedy is the workhorse default.

---

## 9. When the chapter is now slightly dated

Sutton & Barto 2nd ed. (2018) is more current than most textbooks — but two things have shifted:

1. **Sample-efficiency wars.** Model-based RL (Dreamer, MuZero) and offline RL (CQL) are not in this chapter. They matter for many 2026 industrial problems.
2. **LLMs as world models.** The 2018 book pre-dates the rise of LLMs as zero-shot policies and as world models for downstream RL. Reading 2 is similarly pre-LLM.

The MDP formalism, the value-function definitions, and the TD-error update — *those* have not changed. Master them.

---

## 10. Study questions

1. Define the five elements of an MDP and give a concrete example for each in **CartPole**.
2. Compute `v_π(s)` for a deterministic policy in a 2-state MDP given numerical transition probabilities and rewards (worked example in Sutton & Barto §3.5 — re-derive).
3. State the **Bellman optimality equation for `q*`** and explain in two sentences why it has a `max` instead of an `E_π`.
4. The TD(0) update is `V(s) ← V(s) + α δ`. Write out `δ` and explain why it's a *biased* estimate of the return.
5. Why is Q-learning called **off-policy** while SARSA is on-policy? What practical difference does this make in CartPole?
6. State the three conditions under which tabular Q-learning is guaranteed to converge to `q*`. Which one *breaks* when we replace the table with a neural network?
7. ε-greedy is sometimes criticised. Sketch a 5-state corridor MDP where ε-greedy takes exponentially longer than a smarter explorer to find the rewarding state.

Bring written answers (≤ 1 page) to Lab 9.

---

## 11. Key takeaways

1. **MDPs** are the substrate: `(S, A, P, R, γ)`.
2. **Value functions** `v_π` and `q_π` are *expected returns*; the Bellman equations express their recursive structure.
3. **TD learning** bootstraps: update toward `r + γ V(s')` instead of waiting for `G_t`. Lower variance, slight bias.
4. **Q-learning** is off-policy TD control; it converges in the tabular limit but loses that guarantee with function approximation.
5. **ε-greedy** is the standard exploration heuristic; better ones exist but this is the workhorse.

---

## 12. Recommended next steps

- **Module 9 Reading 2** — the DQN paper. The function-approximation extension of Q-learning.
- David Silver's **Reinforcement Learning Lecture Series** (DeepMind, YouTube). Lecture 1 covers MDPs; Lecture 5 covers Q-learning.
- The OpenAI Spinning Up curriculum (`spinningup.openai.com`) — modern deep-RL practical guide.
- *Try Lab 9 first* before Assignment 9 — tabular FrozenLake builds the intuition for what neural Q-learning *should* be approximating.

> When you can write the Q-learning update from memory and explain what makes it off-policy, you have absorbed Module 9 Reading 1.