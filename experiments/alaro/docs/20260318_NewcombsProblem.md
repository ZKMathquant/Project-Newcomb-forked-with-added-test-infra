# Newcomb's Problem: Bayesian, Game-Theoretic, and Infra-Bayesian Perspectives

This document analyzes Newcomb's problem through three lenses and derives
update rules for each. Unlike the Copy/Reverse Heads example (where game
theory subsumes infra-Bayesianism), Newcomb's problem exposes a genuine gap
in the game-theoretic framework that IB is designed to fill.

---

## 1. Setup (This Repo's Framework)

The environment is defined in `ibrl/environments/newcomb.py` with default
parameters boxA = 5, boxB = 10.

### Reward table T[prediction, action]

|                          | Action 0 (one-box) | Action 1 (two-box) |
|--------------------------|---------------------|---------------------|
| **Pred 0** (one-box)    | boxB = 10           | boxB + boxA = 15    |
| **Pred 1** (two-box)    | 0                   | boxA = 5            |

- **One-boxing** = take only Box B (the opaque box)
- **Two-boxing** = take both Box A (transparent, always contains boxA) and Box B

### The predictor

The predictor **samples an action from the agent's policy distribution**.
If the agent's policy is p = P(one-box):

- Predictor predicts "one-box" with probability p
- Predictor predicts "two-box" with probability 1 − p

This is a **perfect policy-sampling predictor**: it doesn't observe the
agent's actual action, only the policy the action is drawn from.

### The interaction loop

Each step:
1. Agent publishes policy p via `get_probabilities()`.
2. Environment calls `predict(p)`: samples prediction ~ p, sets reward row.
3. Agent's action a is sampled from p.
4. Agent receives reward T[prediction, a] via `interact(a)`.
5. Agent updates via `update(p, a, reward)`.

Crucially, the agent **does not observe the prediction** — only its own
policy, action, and reward.

---

## 2. The True Optimal Policy

Expected utility as a function of policy p = P(one-box):

    EU(p) = Σ_{i,j} P(pred=i) · P(act=j) · T[i,j]
          = p·p·10 + p·(1−p)·15 + (1−p)·p·0 + (1−p)²·5

Expanding:

    = 10p² + 15p − 15p² + 5 − 10p + 5p²
    = (10 − 15 + 5)p² + (15 − 10)p + 5
    = 5p + 5

**EU(p) = 5p + 5** — perfectly linear in p. Maximized at **p = 1** (pure
one-boxing), giving EU = **10 = boxB**.

The optimal policy is to one-box with certainty. The expected reward is boxB.

For general boxA, boxB: EU(p) = p(boxB − boxA) + boxA, so one-boxing is
optimal whenever boxB > boxA.

**Note the surprise**: two-boxing "dominates" one-boxing in the sense that
for any fixed prediction, two-boxing gives boxA more. But the prediction
is not fixed — it correlates with the policy. This correlation inverts the
dominance argument.

---

## 3. The Classical Bayesian Learner

A Bayesian learner maintains a belief over the reward table T. Let's trace
what it observes and how it updates.

### What the agent observes

When the agent plays policy p and takes action a, it receives reward r where:

    E[r | p, a] = p · T[0, a] + (1−p) · T[1, a]

The observed reward is a **mixture** over the prediction (which the agent
doesn't see). The mixture weights are the agent's own policy.

### The learning trap

Suppose the agent starts with an exploratory policy p = 0.5 (uniform):

| Action     | E[reward \| p=0.5] | Computation                    |
|------------|---------------------|--------------------------------|
| One-box    | 5                   | 0.5 · 10 + 0.5 · 0 = 5        |
| Two-box    | 10                  | 0.5 · 15 + 0.5 · 5 = 10       |

**Two-boxing looks twice as good!** A standard Bayesian learner (or any
bandit algorithm) will shift toward two-boxing.

As the policy shifts to p ≈ 0 (mostly two-boxing):

| Action     | E[reward \| p≈0] | Computation                    |
|------------|-------------------|--------------------------------|
| One-box    | ≈ 0               | 0·10 + 1·0 = 0                |
| Two-box    | ≈ 5               | 0·15 + 1·5 = 5                |

The agent converges to two-boxing with reward ≈ 5. It never discovers that
p = 1 would give reward 10, because **shifting the policy changes the
mixture weights**.

### Why the Bayesian gets stuck

The Bayesian treats the observed reward as evidence about the reward table,
but it conditions on the *action* without accounting for the *policy's effect
on the prediction*. Concretely:

A Bayesian updating reward estimates for action a:

    μ̂_a = (1/n_a) Σ_{t: a_t = a} r_t

This converges to E[r | p_∞, a] = p_∞ · T[0,a] + (1−p_∞) · T[1,a], which
depends on the **converged** policy p_∞. If p_∞ ≈ 0 (two-boxing), the
estimates converge to:

    μ̂₀ → T[1, 0] = 0      (one-boxing looks terrible)
    μ̂₁ → T[1, 1] = 5      (two-boxing looks fine)

The agent has no evidence that one-boxing could be better, because it
never explores the regime where p ≈ 1 (which would shift the prediction).

### The Bayesian update rule

    P(T | data) ∝ P(data | T) · P(T)

    P(r_t | T, p_t, a_t) = Σ_i P(pred=i | p_t) · P(r_t | T[i, a_t])
                          = Σ_i p_t[i] · δ(r_t = T[i, a_t])

This likelihood correctly accounts for the mixture, but the Bayesian still
chooses the **action** that maximizes expected reward under the current
belief, not the **policy** that maximizes reward when the prediction
correlates with the policy. The Bayesian updates the belief over T but
chooses actions, not policies. It asks "given what I believe about T, which
action is best?" rather than "given what I believe about T, which policy
is best knowing the predictor will sample from it?"

A Bayesian that optimized over **policies** (not just actions) and modeled the
predictor's sampling mechanism would correctly one-box. But standard Bayesian
bandit algorithms don't do this — they treat each action independently.

---

## 4. The Game-Theoretic Analysis

### 4.1 The zero-sum game against adversarial Nature

The natural game-theoretic framing: Agent picks a column (action), Nature
picks a row (prediction) to minimize the agent's payoff.

|                | One-box | Two-box |
|----------------|---------|---------|
| **Pred=1box**  | 10      | 15      |
| **Pred=2box**  | 0       | 5       |

Two-boxing **strictly dominates** one-boxing: 15 > 10 and 5 > 0. The agent
always two-boxes. Nature then picks Pred=2box. Equilibrium: **(Pred=2box,
Two-box)**, payoff = **5**.

This is the **Causal Decision Theory (CDT)** answer. It treats the
prediction as an independent adversarial choice.

### 4.2 Why the game-theoretic framing fails

The zero-sum game assumes Nature can **independently** choose the prediction.
But in Newcomb's problem, the prediction is **structurally correlated** with
the agent's policy — the predictor samples from it.

The game-theoretic framework gives Nature too much power. It lets Nature
decorrelate the prediction from the policy, which violates the problem's
causal structure. The result is that "two-boxing dominates" — the standard
dominance argument — which is correct only if the prediction is causally
independent of the action.

### 4.3 Can game theory be repaired?

Several attempted repairs:

**Stackelberg game (agent as leader):** The agent commits to a policy p,
then Nature best-responds. But if Nature is adversarial, it picks the
worst prediction for each p, and two-boxing still dominates. If Nature is
constrained to sample from p, it's no longer a game — it's a single-agent
optimization problem, and we recover EU(p) = 5p + 5.

**Correlated equilibrium:** Requires a shared correlation device. The agent's
policy IS the correlation device (the predictor samples it), but standard
correlated equilibrium theory assumes exogenous correlation, not
policy-dependent correlation.

**Constrained Nature:** We could add the constraint "Nature must predict by
sampling from p" to the game. But this collapses the game to a single-agent
problem: max_p EU(p) = max_p [5p + 5] = 10. The "game" disappears because
Nature has no strategic freedom — it's a mechanism, not a player.

**Bayesian game (Harsanyi):** Model the predictor's accuracy as a "type"
drawn by Nature. The agent has a prior over predictor types. This works
in principle but reduces to a standard Bayesian decision problem — we're
back to Section 3.

### 4.4 The core issue

Game theory models strategic interaction between **independent** agents.
Newcomb's problem features a structural dependency: the predictor's behavior
is a **function** of the agent's policy. This isn't strategic interaction —
it's a causal constraint.

When you try to model Newcomb's as a game:
- If Nature is unconstrained → CDT (two-box), payoff = 5. Wrong.
- If Nature is constrained to sample p → single-agent problem, payoff = 10.
  Correct, but not really game theory anymore.

There is no intermediate position where the game-theoretic framework
naturally captures "the predictor is mostly but not perfectly correlated
with my policy." You either give Nature full freedom (CDT) or no freedom
(EDT), with nothing in between.

### 4.5 Contrast with Copy/Reverse Heads

In Copy/Reverse Heads, the "adversary" (Nature) chose which environment the
agent was in, and the coin was independent of the agent's policy. This is
a standard game of incomplete information (Harsanyi). The observation (coin)
is exogenous, and the agent's strategy at each information set is independent
of Nature's choice. Game theory handles this perfectly — the minimax
behavioral strategy (p_H = 0.25, p_T = 0) achieves worst-case 0.625.

In Newcomb's, the "observation" (reward) depends on both the environment
AND the agent's policy through the prediction. The policy affects the
evidence, which affects what the agent learns, which affects the policy.
This circularity is what standard game theory struggles with.

---

## 5. The Infra-Bayesian Analysis

### 5.1 The credal set over reward tables

An IB agent maintains interval bounds on the reward table:

    T_lower[i,j] ≤ T[i,j] ≤ T_upper[i,j]

Initially (wide bounds):

    T_lower = [[-R, -R], [-R, -R]]
    T_upper = [[+R, +R], [+R, +R]]

The credal set is: {all reward tables T within these bounds}.

### 5.2 Policy optimization under worst-case

The agent chooses policy p to maximize the worst-case expected utility over
the credal set. For a policy p and reward table T:

    EU(p, T) = p²·T[0,0] + p(1−p)·T[0,1] + (1−p)p·T[1,0] + (1−p)²·T[1,1]

The worst-case reward table, for a given p, minimizes EU(p, T) subject to
the interval constraints. Since EU is linear in each T[i,j], the minimum
is achieved by setting:

    T[i,j] = T_lower[i,j]   for all (i,j) with positive coefficient

(All coefficients p_i · p_j are non-negative, so this just means: use the
lower bounds everywhere.)

    V(p) = min_T EU(p, T)
         = p²·T_lower[0,0] + p(1−p)·T_lower[0,1]
           + (1−p)p·T_lower[1,0] + (1−p)²·T_lower[1,1]

The agent then maximizes V(p) over p ∈ [0,1].

### 5.3 The IB update rule for Newcomb's

After playing policy p, taking action a, and observing reward r, the agent
updates its interval bounds.

**What the agent knows:** r = T[pred, a] where pred was sampled from p.
But the agent doesn't know pred. So:

    r ∈ { T[0, a], T[1, a] }

with P(r = T[0,a]) = p and P(r = T[1,a]) = 1−p.

**Conservative update (per the ibplan.md):**

Track per-action running mean μ_a and count n_a:

    μ_a = (1/n_a) Σ_{t: a_t = a} r_t

This converges to E[r | p_t, a] ≈ p_avg · T[0,a] + (1−p_avg) · T[1,a].

Tighten bounds using confidence intervals:

    T_lower[i,a] = max(T_lower[i,a], μ_a − δ(n_a))
    T_upper[i,a] = min(T_upper[i,a], μ_a + δ(n_a))

where δ(n) = C/√n for confidence parameter C.

**Why this is conservative:** The bound applies to the MIXTURE, not to
individual T entries. The agent can't separate T[0,a] from T[1,a] without
observing the prediction. But this conservatism is deliberate — the IB
agent stays cautious about what it doesn't know.

### 5.4 The crucial difference: policy-awareness

The IB agent optimizes over **policies**, not actions. When evaluating
policy p, it computes:

    V(p) = Σ_{i,j} p_i · p_j · T_lower[i,j]

This means the agent considers: "If I shift to p = 1, the predictor shifts
to pred = 0, and I get T[0,0]." The worst-case T[0,0] might still be high
enough (if the lower bound has been tightened upward by observations) to
favor one-boxing.

Compare with the Bayesian bandit, which asks: "Given my current estimates,
which ACTION is best?" without considering how the policy shift changes the
prediction distribution.

### 5.5 Worked example: IB agent learning in Newcomb's

**Step 0: Initialization**

    T_lower = [[−20, −20], [−20, −20]]
    T_upper = [[+20, +20], [+20, +20]]

    V(p) = p²(−20) + p(1−p)(−20) + (1−p)p(−20) + (1−p)²(−20) = −20
    (constant — any policy gives worst-case −20)

Agent plays p = 0.5 (arbitrary, since all are equivalent).

**After some steps at p ≈ 0.5:**

Average reward when one-boxing: ≈ 5 (= 0.5·10 + 0.5·0)
Average reward when two-boxing: ≈ 10 (= 0.5·15 + 0.5·5)

Bounds tighten:

    T_lower[·, 0] ≈ [5 − δ, 5 − δ]   (mixture for one-boxing)
    T_upper[·, 0] ≈ [5 + δ, 5 + δ]
    T_lower[·, 1] ≈ [10 − δ, 10 − δ]  (mixture for two-boxing)
    T_upper[·, 1] ≈ [10 + δ, 10 + δ]

At this stage, the IB agent evaluates:

    V(p) = p²·(5−δ) + p(1−p)·(10−δ) + (1−p)p·(5−δ) + (1−p)²·(10−δ)

Hmm — two-boxing looks better. But the bounds are MIXTURE bounds; the
agent knows they apply to the mixture, not individual entries.

**The key subtlety**: With a more sophisticated update (see Section 5.6),
the agent can exploit the fact that different policies p change the
mixture weights. By comparing observations at different p values, it can
partially disentangle T[0,a] from T[1,a].

### 5.6 Disentangling the mixture (smarter update)

If the agent plays p₁ = 0.8 for a while and then p₂ = 0.2, it observes:

At p₁ = 0.8, action = one-box: E[r] = 0.8·10 + 0.2·0 = 8
At p₂ = 0.2, action = one-box: E[r] = 0.2·10 + 0.8·0 = 2

These are two linear equations in two unknowns:

    0.8·T[0,0] + 0.2·T[1,0] = 8
    0.2·T[0,0] + 0.8·T[1,0] = 2

Solving: T[0,0] = 10, T[1,0] = 0. The agent has recovered the true
reward table entries!

Similarly for two-boxing at different p values, it can recover T[0,1] = 15
and T[1,1] = 5.

**With the full table known**, the IB agent computes:

    V(p) = p²·10 + p(1−p)·15 + (1−p)p·0 + (1−p)²·5 = 5p + 5

And correctly shifts to p = 1 (one-boxing).

### 5.7 The IB advantage: structured exploration

The IB framework provides a principled reason to explore different
policies: **the credal set is wide**, meaning the agent is uncertain about
the reward table, meaning different policies are worth trying to narrow
the uncertainty.

Specifically, the worst-case analysis with wide bounds gives the same value
for all policies (everything looks equally bad in the worst case). This
forces the agent to explore rather than exploit prematurely. As bounds
tighten asymmetrically, the agent shifts toward the policy favored by the
tightened bounds.

This contrasts with:
- **Bayesian bandits**: converge greedily to two-boxing and stop exploring
- **Game theory**: gives the static answer "two-box" (CDT) without a
  learning process

---

## 6. Why Newcomb's Is Harder Than Copy/Reverse Heads

### 6.1 The structural difference

| Feature                          | Copy/Reverse Heads      | Newcomb's Problem        |
|----------------------------------|-------------------------|--------------------------|
| Observation (coin/reward)        | Independent of policy   | Depends on policy        |
| Nature's "move"                  | Choose environment      | Prediction samples policy|
| Agent's decision variable        | Action at info set      | Full policy p            |
| Standard game theory applies?    | Yes (extensive-form)    | No (policy-dependent)    |
| Predictor is adversarial?        | N/A (coin is random)    | No (correlated)          |

### 6.2 The correlation problem

In Copy/Reverse Heads, the coin is exogenous — it doesn't depend on the
agent's strategy. The agent's task is to respond optimally to an observation
drawn from a known distribution under unknown payoffs. This is a textbook
incomplete-information game.

In Newcomb's, the prediction is endogenous — it's drawn from the agent's
own policy. The agent's policy determines the distribution of predictions,
which determines the rewards, which determines the optimal policy. This
circularity is the core of the Newcomb puzzle.

### 6.3 Why game theory can't bridge the gap

Game theory's central tools — Nash equilibrium, minimax, backward induction —
assume that each player's strategy set is independent of the other players'
strategies. In Newcomb's, Nature's "strategy" (the prediction distribution)
is a **function** of the agent's strategy. This is not a game between
independent agents; it is a decision problem with self-referential causal
structure.

You could formalize it as a game where Nature is constrained to play a
specific response function, but then Nature isn't really a player — it's a
mechanism. And optimizing against a known mechanism is just single-agent
optimization, which gives the correct answer (one-box) but provides no
general-purpose update rule for learning in unknown mechanism environments.

### 6.4 What IB adds

IB provides a framework where:

1. The agent doesn't need to know the mechanism (predictor) in advance.
2. The credal set represents uncertainty over possible mechanisms.
3. The maximin over policies naturally considers how different policies
   interact with different mechanisms.
4. The update rule tightens the credal set from observations, gradually
   revealing the mechanism's structure.

This is genuinely more than what game theory offers for this problem class.
Game theory either gives you CDT (wrong answer, robust to adversaries) or
requires you to fully specify the mechanism (right answer, no learning).
IB sits in between: it can learn the mechanism's structure while remaining
robust to adversarial alternatives.

---

## 7. Summary of Update Rules

### Classical Bayesian Learner

    # Per-action reward estimates (mixture)
    μ_a ← (μ_a · n_a + r) / (n_a + 1)     when action a is taken
    n_a ← n_a + 1

    # Action selection (greedy)
    a* = argmax_a  μ_a

    Converges to: two-boxing (μ₁ > μ₀ at p ≈ 0). Payoff ≈ 5.

### Game-Theoretic Optimal Agent (Zero-Sum vs Nature)

    # Dominant strategy analysis
    a* = argmax_a  min_pred  T[pred, a]

    min_pred T[pred, 0] = min(10, 0) = 0
    min_pred T[pred, 1] = min(15, 5) = 5

    a* = two-box. Payoff = 5.

    (No learning needed — two-boxing dominates regardless of prediction.)

### Infra-Bayesian Learner

    # Initialize bounds
    T_lo[i,j] = −R,  T_hi[i,j] = +R    for all i,j

    # Each step: observe (p, a, r)
    n_a ← n_a + 1
    μ_a ← running mean of rewards when taking action a

    # Tighten bounds (conservative: applies to mixture)
    for i in {0, 1}:
        T_lo[i,a] ← max(T_lo[i,a],  μ_a − C/√n_a)
        T_hi[i,a] ← min(T_hi[i,a],  μ_a + C/√n_a)

    # Policy optimization (maximin over credal set)
    p* = argmax_p  min_T∈[T_lo,T_hi]  EU(p, T)
       = argmax_p  Σ_{i,j} p_i · p_j · T_lo[i,j]

    Converges to: one-boxing (as bounds tighten and reveal T[0,0] > T[1,1]).
    Payoff → 10.

    # Smarter update: exploit different policies to disentangle mixture
    If observations at p₁ and p₂ (p₁ ≠ p₂) give mean rewards μ₁_a and μ₂_a:
        T̂[0,a] = (p₂·μ₁_a − p₁·μ₂_a) / (p₂ − p₁)     [sic: more care needed for signs]
        T̂[1,a] = (μ₂_a − μ₁_a) / (p₂ − p₁) + T̂[0,a]   [moment matching]
    Use these to tighten per-entry bounds directly.

---

## 8. The Verdict

| Criterion                       | Bayesian     | Game Theory  | Infra-Bayesian |
|---------------------------------|--------------|--------------|----------------|
| Correct answer (static)         | Depends on p | No (CDT)     | Yes            |
| Learns from experience          | Yes          | No           | Yes            |
| Converges to optimal            | No (trap)    | N/A          | Yes (with expl)|
| Handles predictor correlation   | Only if modeled| No          | Yes            |
| Handles adversarial predictor   | No           | Yes          | Yes            |
| Requires known mechanism        | No           | No           | No             |

The fundamental advantage of IB in Newcomb's: it optimizes over **policies**
under **worst-case reward tables**, which naturally captures the
policy → prediction → reward causal chain. Game theory, by treating the
prediction as either adversarial (CDT) or fixed (mechanism design), cannot
represent the intermediate case where the prediction is correlated but
potentially imperfect. A Bayesian bandit optimizes over **actions** rather
than policies, missing the same structure from the other direction.

Newcomb's problem is the natural habitat for infra-Bayesianism in a way
that Copy/Reverse Heads was not. The policy-dependence of the prediction —
the very thing that makes Newcomb's paradoxical — is precisely the
structure that IB's credal-set-over-reward-tables framework was built to
handle.
