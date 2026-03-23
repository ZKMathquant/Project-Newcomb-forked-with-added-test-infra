# Newcomb's Problem (Standard Formulation): Bayesian, Game-Theoretic, and Infra-Bayesian Perspectives

This document analyzes the **standard (Wikipedia/philosophical)** formulation
of Newcomb's problem, where the predictor forecasts the agent's actual choice
with high accuracy. This differs from the repo's formulation (where the
predictor samples from the agent's policy distribution) in ways that matter
for mixed strategies and learning.

---

## 1. Setup

### The scenario

A being called **Omega** (the predictor) presents you with two boxes:

- **Box A** (transparent): always contains **$1,000**
- **Box B** (opaque): contains **$1,000,000** or **$0**

You choose either:
- **One-box**: take only Box B
- **Two-box**: take both Box A and Box B

Before you choose, Omega has already predicted your choice:
- If Omega predicted you would **one-box**, Box B contains $1,000,000.
- If Omega predicted you would **two-box**, Box B contains $0.

Omega has been correct in a very large fraction of past cases. Let
**q** denote Omega's accuracy: P(Omega correct).

### Payoff matrix

Let M = $1,000,000 and K = $1,000.

|                            | One-box      | Two-box      |
|----------------------------|--------------|--------------|
| **Omega predicted 1-box**  | M = 1,000,000| M+K = 1,001,000 |
| **Omega predicted 2-box**  | 0            | K = 1,000    |

### Key structural feature

Omega predicts the **actual choice**, not the policy/disposition. If you
flip a coin to decide, Omega predicts the coin's outcome (with accuracy q).
This means:

    P(Omega predicts one-box | you actually one-box) = q
    P(Omega predicts two-box | you actually two-box) = q

The prediction correlates with the **realized action**, not merely the
probability distribution over actions.

### How this differs from the repo

| Feature                | This formulation              | Repo formulation             |
|------------------------|-------------------------------|------------------------------|
| Predictor input        | Actual choice (with noise)    | Policy distribution          |
| P(pred=act)            | q (fixed accuracy)            | p² + (1-p)² (varies with p) |
| Mixed strategy effect  | Doesn't change predictor      | Changes predictor distribution|
| Payoffs                | $1M / $1K                     | 10 / 5                      |

This difference is crucial: in the repo, shifting your policy changes the
predictor's behavior (because it samples your policy). In the standard
version, the predictor tracks your actual choice regardless of how you
generate it.

---

## 2. Expected Utility as a Function of Policy

Let p = probability of one-boxing. The agent's action a is sampled from p.
Omega predicts a correctly with probability q.

Joint distribution over (prediction, action):

    P(pred=1box, act=1box) = p · q
    P(pred=1box, act=2box) = (1-p) · (1-q)
    P(pred=2box, act=1box) = p · (1-q)
    P(pred=2box, act=2box) = (1-p) · q

Expected utility:

    EU(p, q) = pq · M + (1-p)(1-q) · (M+K) + p(1-q) · 0 + (1-p)q · K

Expanding:

    EU(p, q) = Mpq + (M+K)(1 - p - q + pq) + K(q - pq)
             = Mpq + (M+K) - (M+K)p - (M+K)q + (M+K)pq + Kq - Kpq
             = pq[M + M + K - K] + p[-(M+K)] + q[-(M+K) + K] + (M+K)
             = 2Mpq - (M+K)p - Mq + (M+K)

**Derivative with respect to p:**

    ∂EU/∂p = 2Mq - (M+K)

This is **constant in p** — EU is linear in p for any fixed q. Therefore:

- If q > (M+K)/(2M) = 0.5005: EU increases in p → **one-box** (p = 1)
- If q < 0.5005: EU decreases in p → **two-box** (p = 0)
- If q = 0.5005: EU constant in p → **indifferent**

**Critical insight: mixing never helps.** Unlike Copy/Reverse Heads (where
mixing at p_H = 0.25 achieved a higher worst-case than any pure strategy),
Newcomb's has linear EU in p for any fixed q. There is no hedging benefit
from randomization when you face a single predictor with fixed accuracy.

---

## 3. The Classical Bayesian Agent

### 3.1 EDT (Evidential Decision Theory) Bayesian

The EDT Bayesian holds a prior P(q) over Omega's accuracy and computes
expected utility by conditioning on the action:

    EU(one-box) = E_q[q] · M = q̄ · M

    EU(two-box) = E_q[1-q] · (M+K) + E_q[q] · K = (1-q̄)(M+K) + q̄K
                = M + K - q̄M

One-boxing is better when:

    q̄ · M > M + K - q̄ · M
    2q̄ · M > M + K
    q̄ > (M + K) / (2M) = 0.5005

So the EDT Bayesian one-boxes whenever **E[q] > 50.05%**. Given that Omega
is described as highly reliable, any reasonable prior concentrates well above
this threshold.

**Update rule**: After observing past predictions and outcomes, the Bayesian
updates P(q) via Bayes' rule. If Omega was correct in n out of n cases, the
posterior strongly favors q ≈ 1, and one-boxing is decisive.

### 3.2 CDT (Causal Decision Theory) Bayesian

CDT holds that your choice now cannot causally affect what Omega already
put in Box B. Let π = P(Box B has $1M) (determined by Omega's past
prediction). CDT treats π as fixed:

    EU(one-box) = π · M
    EU(two-box) = π · (M + K) + (1-π) · K = πM + K

Two-boxing is always $K better: **EU(two-box) - EU(one-box) = K = $1,000**.

CDT two-boxes regardless of Omega's accuracy, the prior, or any evidence.
The dominance argument: "Whatever is in Box B, I get $1,000 more by also
taking Box A."

### 3.3 Why the Bayesian picture is clear but contested

The Bayesian analysis reduces Newcomb's to a question about causal structure:

- EDT: your action is evidence about Box B's contents → one-box
- CDT: your action doesn't cause Box B's contents to change → two-box

No amount of Bayesian updating resolves this — it's a disagreement about
which conditional probabilities to use, not about the values of those
probabilities. Both CDT and EDT Bayesians agree on P(q); they disagree on
whether to condition on actions or intervene on them.

---

## 4. The Game-Theoretic Agent

### 4.1 Zero-sum game: Agent vs. adversarial Nature

Treat the prediction as Nature's move in a zero-sum game:

|                        | One-box      | Two-box      |
|------------------------|--------------|--------------|
| **Pred = one-box**     | 1,000,000    | 1,001,000    |
| **Pred = two-box**     | 0            | 1,000        |

Two-boxing **strictly dominates** one-boxing (1,001,000 > 1,000,000 and
1,000 > 0). The game-theoretic solution:

    Agent: two-box.  Nature: predict two-box.  Payoff: $1,000.

This is the CDT answer. It treats Nature as an unconstrained adversary who
can decorrelate the prediction from the action.

### 4.2 Zero-sum game with Nature choosing accuracy q

A more nuanced framing: Nature doesn't choose the prediction directly but
chooses the **accuracy** q ∈ [0, 1] (or some interval). The agent chooses
policy p ∈ [0, 1]. Payoff:

    EU(p, q) = 2Mpq - (M+K)p - Mq + (M+K)

This is **bilinear** in (p, q) — a standard bilinear saddle-point game.

**Saddle point**:

    ∂EU/∂p = 2Mq - (M+K) = 0  →  q* = (M+K)/(2M) = 0.5005
    ∂EU/∂q = 2Mp - M = 0       →  p* = 0.5

If Nature can choose any q ∈ [0, 1]:

    Minimax value = EU(0.5, 0.5005) = (M+K)/2 = $500,500

The agent plays p = 0.5 (coin flip) and Nature sets q = 0.5005 (barely
better than chance). Neither player can improve.

### 4.3 Constrained Nature: q ∈ [q_lo, q_hi]

If Nature's choice of q is restricted (reflecting what we know about Omega):

**Case 1: q_lo > 0.5005** (Omega is known to be good)

Both slopes of V(p) are positive → agent one-boxes. Worst case:

    V(1) = EU(1, q_lo) = q_lo · M

For q_lo = 0.9: V(1) = $900,000.

**Case 2: q_hi < 0.5005** (Omega is worse than a coin flip)

Both slopes negative → agent two-boxes.

    V(0) = EU(0, q_hi) = (1 - q_hi) · (M+K) + q_hi · K - ...

Hmm, let's just compute: V(0) = (M+K) - M·q_hi.
For q_hi = 0.4: V(0) = 1,001,000 - 400,000 = $601,000.

**Case 3: q_lo < 0.5005 < q_hi** (Omega's accuracy is uncertain)

V(p) is piecewise linear with a peak at **p = 0.5**:

- For p < 0.5: worst case q = q_hi. Slope = 2M·q_hi - (M+K).
  With q_hi > 0.5005, slope > 0 → V increases.
- For p > 0.5: worst case q = q_lo. Slope = 2M·q_lo - (M+K).
  With q_lo < 0.5005, slope < 0 → V decreases.

The maximum is at p = 0.5 with value:

    V(0.5) = (M+K)/2 = $500,500

(This holds for any [q_lo, q_hi] straddling 0.5005.)

### 4.4 The game theory verdict

| Omega's known accuracy      | Game-theoretic policy | Payoff         |
|------------------------------|-----------------------|----------------|
| q definitely > 0.5005       | One-box (p = 1)       | q_lo · M       |
| q definitely < 0.5005       | Two-box (p = 0)       | (M+K) − M·q_hi|
| q uncertain, spans 0.5005   | Coin flip (p = 0.5)   | $500,500       |
| q completely unknown [0,1]  | Coin flip (p = 0.5)   | $500,500       |

When Omega's accuracy is **known to be high** (the standard setup), game
theory correctly says one-box. When accuracy is **unknown**, game theory
hedges with a coin flip — which is suboptimal if Omega is actually good
(you'd get $500,500 instead of ≈$1,000,000).

---

## 5. The Infra-Bayesian Agent

### 5.1 Credal set formulation

The IB agent maintains Knightian uncertainty over q (Omega's accuracy). The
credal set is an interval [q_lo, q_hi].

The IB agent solves:

    p* = argmax_p  min_{q ∈ [q_lo, q_hi]}  EU(p, q)

Since EU is bilinear in (p, q), this is **mathematically identical to the
game-theoretic maximin** from Section 4.3. The IB agent and the
game-theoretic agent produce the same policy for the same uncertainty set.

### 5.2 So where does IB differ from game theory?

In the **static** (single-shot) problem: it doesn't. Both are solving the
same maximin. The difference is conceptual — game theory frames q as
Nature's strategic choice; IB frames it as epistemic uncertainty — but the
math is the same.

The difference emerges in **learning**. IB provides an update rule for
tightening [q_lo, q_hi] from observations.

### 5.3 The IB update rule

Suppose the agent plays Newcomb's problem repeatedly. Each round:

1. Agent commits to policy p_t.
2. Omega predicts.
3. Agent acts (sampled from p_t), receives reward r_t.

The agent observes (p_t, a_t, r_t) but not Omega's prediction. From the
reward, the agent can sometimes infer the prediction:

- If a_t = one-box and r_t = M: Omega predicted one-box.
- If a_t = one-box and r_t = 0: Omega predicted two-box.
- If a_t = two-box and r_t = M+K: Omega predicted one-box.
- If a_t = two-box and r_t = K: Omega predicted two-box.

**In all cases, the prediction is fully revealed by (action, reward)!**

This is a crucial difference from the repo formulation. In the repo, the
reward table entries can coincide (e.g., boxA and boxB might be close),
making the prediction ambiguous. In the standard formulation with M >> K,
each (action, reward) pair uniquely identifies the prediction.

**Update**: After n rounds, let c = number of correct predictions. The
IB agent updates:

    q_lo = max(q_lo,  c/n - δ(n))
    q_hi = min(q_hi,  c/n + δ(n))

where δ(n) = C/√n is a confidence bound.

As n grows, [q_lo, q_hi] concentrates around the true q. If the true
q > 0.5005 (as assumed), both bounds eventually exceed 0.5005, and the
agent shifts to one-boxing.

### 5.4 Convergence trajectory

**Phase 1: Wide credal set** (early rounds, [q_lo, q_hi] straddles 0.5005)

IB agent plays p ≈ 0.5 (coin flip). This is the maximin policy under
wide uncertainty. The agent gets ≈ $500,500 per round and learns about q
from the revealed predictions.

**Phase 2: Credal set tightens above 0.5005** (after enough observations)

Once q_lo > 0.5005, the IB agent shifts toward p = 1 (one-boxing).
The transition is smooth as the credal set narrows.

**Phase 3: Converged** (q_lo ≈ q_hi ≈ true q)

Agent one-boxes with certainty. Payoff ≈ q · M ≈ $1,000,000.

### 5.5 What the Bayesian does differently

A Bayesian with a uniform prior on q would compute E[q] = 0.5 initially.
Since 0.5 < 0.5005, the EDT Bayesian initially **two-boxes**. After
observing Omega's accuracy, the Bayesian updates E[q] upward and
eventually one-boxes.

But the Bayesian's trajectory depends critically on the prior:
- Prior concentrated near q = 1: one-boxes from the start.
- Uniform prior: two-boxes initially, then switches.
- Prior concentrated near q = 0: two-boxes for a very long time.

The IB agent's behavior depends on [q_lo, q_hi], which is arguably more
transparent: you specify what you're uncertain about, not a full
probability distribution over your uncertainty.

### 5.6 A subtlety: exploration incentives

When the IB agent plays p = 0.5, it's not "exploring" in the traditional
sense — it's playing the maximin policy under uncertainty. But as a side
effect, it generates informative observations (roughly equal numbers of
one-box and two-box actions), which efficiently tighten the credal set.

Compare with a CDT agent that two-boxes every round: it still observes
Omega's predictions (via rewards), so it still learns q. But it never
benefits from this knowledge, because CDT two-boxes regardless of q.

The IB agent's maximin policy under uncertainty **accidentally** generates
good exploration, and its policy **responds** to the tightened credal set.
This is a structural advantage of worst-case optimization in learning
settings.

---

## 6. The Deeper Question: Does Mixing Ever Help?

### 6.1 In the standard formulation: No

EU(p, q) is linear in p for any fixed q. Therefore:

- If you know q, the optimal policy is pure (p = 0 or p = 1).
- Under Knightian uncertainty over q, the optimal policy can be mixed
  (p = 0.5) only because the worst-case q varies with p. The mixing is
  a hedge against **your own uncertainty**, not a strategic advantage.

### 6.2 Contrast with Copy/Reverse Heads

In Copy/Reverse Heads, mixing was genuinely strategic: the agent mixed to
prevent the adversary (who chooses the environment) from exploiting a
deterministic policy. The game had a saddle point in the interior of the
strategy space.

In Newcomb's, the "adversary" is the unknown accuracy q. The agent mixes
only because it doesn't know where the threshold falls. Once q is known,
mixing is strictly suboptimal.

### 6.3 Contrast with the repo formulation

In the repo, EU(p) = (boxB - boxA) · p + boxA — also linear in p, also
no mixing benefit. But for a different reason: the repo's predictor samples
the policy, so the "accuracy" is endogenous (P(pred=act) = p² + (1-p)²),
and there is only one environment. There's nothing to hedge against.

In the standard formulation, the linearity comes from the fixed accuracy q
being independent of p. The mixing in the IB case hedges against unknown q,
not against the predictor itself.

---

## 7. The Three Approaches Compared

### Payoff table for a single shot with q = 0.99

| Agent                     | Policy | Expected payoff |
|---------------------------|--------|-----------------|
| CDT / Game theory (naive) | p = 0  | $11,990         |
| EDT Bayesian (E[q] = 0.99)| p = 1  | $990,000        |
| IB (q ∈ [0.95, 1.0])     | p = 1  | $950,000        |
| IB (q ∈ [0.0, 1.0])      | p = 0.5| $500,500        |
| Game theory (q ∈ [0.0,1]) | p = 0.5| $500,500        |

### Update rules

**CDT Bayesian:**

    # No update affects the decision
    a* = two-box,  always.

**EDT Bayesian:**

    # Prior: P(q) (e.g., Beta distribution)
    # After observing prediction outcome (correct/incorrect):
    P(q | data) ∝ q^c · (1-q)^(n-c) · P(q)    (c correct in n rounds)

    # Decision:
    q̄ = E[q | data]
    a* = one-box  if  q̄ > (M+K)/(2M) = 0.5005
         two-box  otherwise

**IB agent:**

    # Credal set: q ∈ [q_lo, q_hi]
    # After observing prediction outcome:
    q_lo ← max(q_lo, c/n − C/√n)
    q_hi ← min(q_hi, c/n + C/√n)

    # Policy optimization:
    if q_lo > 0.5005:       p* = 1  (one-box)
    elif q_hi < 0.5005:     p* = 0  (two-box)
    else:                   p* = 0.5 (coin flip)

**Game-theoretic agent (adversarial Nature chooses q):**

    # Static — no learning
    if q constrained above 0.5005:  p* = 1
    if q constrained below 0.5005:  p* = 0
    if q unconstrained:             p* = 0.5

---

## 8. Where IB Genuinely Adds Value in Standard Newcomb's

### 8.1 Not in the static analysis

For a single shot, IB's maximin over [q_lo, q_hi] is identical to the
game-theoretic maximin with Nature choosing q ∈ [q_lo, q_hi]. The bilinear
structure makes them the same optimization problem.

### 8.2 In the learning dynamics

IB provides a **principled update rule** for tightening the credal set
from observations, coupled with a **policy that responds to the tightened
set**. This combination produces good behavior:

1. Start uncertain → play p = 0.5 → generates informative observations.
2. Learn q → tighten [q_lo, q_hi] → shift toward one-boxing.
3. Converge → one-box with certainty → near-optimal payoff.

Game theory (static maximin) doesn't have a learning component. A Bayesian
learns but requires a prior and doesn't handle model misspecification.
IB sits between: it learns (like Bayes) with worst-case guarantees (like
game theory).

### 8.3 In robustness

If Omega's accuracy **changes over time** (or varies across encounters),
the IB agent's wide credal set provides automatic robustness. A Bayesian
with a tight posterior on q ≈ 1 would be slow to react if q suddenly
dropped. The IB agent, if its update rule allows re-widening (e.g.,
sliding-window bounds), adapts faster.

### 8.4 In handling richer uncertainty

The analysis above uses a single parameter q. In reality, Omega might have:
- Different accuracy for one-boxers vs two-boxers
- Accuracy that depends on the agent's deliberation process
- Contextual accuracy (better on Tuesdays, worse when tired)

IB can represent this as a credal set over a richer space of predictor
models, without committing to a specific parametric family. Game theory
can do this too (Nature chooses from the same set), but IB's update
machinery is designed for it.

---

## 9. The Honest Summary

| Question                                    | Answer                          |
|---------------------------------------------|---------------------------------|
| Does IB give a different static answer?     | No (same as game theory maximin)|
| Does mixing ever help in standard Newcomb's?| Only under uncertainty about q  |
| Does IB learn better than Bayes?            | More robust, less efficient     |
| Does IB beat game theory?                   | Only in learning (GT is static) |
| Is the dominance argument (CDT) refuted?    | Not by IB — same debate as EDT  |
| Is IB necessary for standard Newcomb's?     | Helpful for learning; not for the static paradox |

The standard formulation of Newcomb's problem is fundamentally a debate
about **causal structure** (does your choice affect the box contents?), not
about **uncertainty representation**. IB's main contribution — replacing
a single prior with a credal set — is orthogonal to the CDT/EDT divide.
An IB-CDT agent would still two-box (dominance holds under worst-case too).
An IB-EDT agent one-boxes once the credal set supports q > 0.5005.

Where IB shines is in the **repeated/learning** version of the problem:
given uncertainty about Omega's accuracy, the IB framework provides a
clean separation between "what I'm uncertain about" (the credal set) and
"how I act given my uncertainty" (maximin), with principled updates that
converge to the right answer. This is more than game theory offers (no
learning) and more robust than Bayes offers (no prior commitment).

The repo's formulation (policy-sampling predictor) is actually **more
favorable to IB** than the standard formulation, because it introduces
policy-dependence that game theory can't model. The standard formulation's
fixed-accuracy predictor is a simpler structure that both game theory and
IB handle equivalently in the static case.
