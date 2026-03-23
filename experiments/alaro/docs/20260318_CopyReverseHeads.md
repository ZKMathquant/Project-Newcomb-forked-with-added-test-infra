# The Copy / Reverse Heads Example: Update Rules for Bayesian and Infra-Bayesian Agents

This document walks through the "Copy / Reverse Heads" example from the
[Introduction to the Infra-Bayesianism Sequence](https://www.lesswrong.com/posts/zB4f7QqKhBHa5b37a/introduction-to-the-infra-bayesianism-sequence)
(Vanessa Kosoy & Diffractor). The example motivates *sa-measures* — the
machinery infra-Bayesianism uses to update beliefs while preserving dynamic
consistency under Knightian uncertainty.

---

## 1. Setup

A fair coin is flipped. The agent **observes** the result (heads or tails) and
then **states** either "heads" or "tails." The agent's reward depends on which
of two possible environments it is in:

### COPY environment

| Coin | Agent says "heads" | Agent says "tails" |
|------|--------------------|--------------------|
| H    | 1                  | 0                  |
| T    | 0                  | 1                  |

Rule: the agent gets 1 if its statement **matches** the coin, 0 otherwise.

### REVERSE HEADS environment

| Coin | Agent says "heads" | Agent says "tails" |
|------|--------------------|--------------------|
| H    | 0                  | 1                  |
| T    | 0.5                | 0.5                |

Rule: if the coin is **tails**, the agent gets 0.5 regardless. If the coin is
**heads**, the agent gets 1 for saying "tails" and 0 for saying "heads."

### The agent's uncertainty

The agent does **not** know which environment it is in. There are two different
epistemic stances it could take:

- **Bayesian**: place a single prior probability P(COPY) = p, P(REVERSE HEADS) = 1 - p.
- **Infra-Bayesian**: maintain *Knightian uncertainty* — the agent's credal set
  contains both environments, and it evaluates policies by their **worst-case**
  expected utility across the set.

---

## 2. All Policies Enumerated

A **policy** specifies what the agent says after each possible coin outcome.
There are exactly four deterministic policies:

| Policy               | On Heads, say | On Tails, say |
|----------------------|---------------|---------------|
| **Always Heads**     | heads         | heads         |
| **Always Tails**     | tails         | tails         |
| **Copy the Coin**    | heads         | tails         |
| **Reverse the Coin** | tails         | heads         |

---

## 3. Expected Utilities for Every Policy Under Every Environment

The coin is fair (P(H) = P(T) = 0.5), so expected utility = 0.5 * (reward on
heads) + 0.5 * (reward on tails).

### Always Heads

- **COPY**: 0.5 * 1 + 0.5 * 0 = **0.5**
- **REVERSE HEADS**: 0.5 * 0 + 0.5 * 0.5 = **0.25**
- Worst case: **0.25**

### Always Tails

- **COPY**: 0.5 * 0 + 0.5 * 1 = **0.5**
- **REVERSE HEADS**: 0.5 * 1 + 0.5 * 0.5 = **0.75**
- Worst case: **0.5**

### Copy the Coin

- **COPY**: 0.5 * 1 + 0.5 * 1 = **1.0**
- **REVERSE HEADS**: 0.5 * 0 + 0.5 * 0.5 = **0.25**
- Worst case: **0.25**

### Reverse the Coin

- **COPY**: 0.5 * 0 + 0.5 * 0 = **0.0**
- **REVERSE HEADS**: 0.5 * 1 + 0.5 * 0.5 = **0.75**
- Worst case: **0.0**

### Summary Table

| Policy           | EU (COPY) | EU (REVERSE HEADS) | Worst Case |
|------------------|-----------|---------------------|------------|
| Always Heads     | 0.50      | 0.25                | 0.25       |
| Always Tails     | 0.50      | 0.75                | **0.50**   |
| Copy the Coin    | 1.00      | 0.25                | 0.25       |
| Reverse the Coin | 0.00      | 0.75                | 0.00       |

The **maximin policy** (infra-Bayesian optimum) is **Always Tails**, with a
worst-case expected utility of **0.5**.

---

## 4. The Classical Bayesian Update Rule

A Bayesian agent holds a prior P(COPY) = p. Since the coin is fair in both
environments, observing "heads" or "tails" does not change the posterior:

    P(COPY | coin = H) = P(H | COPY) * P(COPY) / P(H)
                       = 0.5 * p / 0.5
                       = p

Both environments assign the same probability to each coin outcome, so the
**likelihood ratio is 1** and the **posterior equals the prior** for any
observation. The Bayesian agent never updates its environment belief from
coin observations alone.

### Bayesian decision after observing Heads

Given posterior p on COPY (= prior p), the agent chooses between:

- Say "heads": p * 1 + (1 - p) * 0 = p
- Say "tails": p * 0 + (1 - p) * 1 = 1 - p

So the Bayesian says "heads" iff p > 0.5, "tails" iff p < 0.5, and is
indifferent at p = 0.5.

### Bayesian decision after observing Tails

- Say "heads": p * 0 + (1 - p) * 0.5 = 0.5(1 - p)
- Say "tails": p * 1 + (1 - p) * 0.5 = 0.5 + 0.5p

The agent always prefers "tails" after observing tails (for any p ≥ 0).

### Bayesian dynamic consistency

The Bayesian is always dynamically consistent because Bayes' rule preserves
the connection between the prior plan and the conditional plan. The full
policy implied by "say heads iff p > 0.5 after heads; always say tails after
tails" is exactly the policy that maximizes expected utility before seeing the
coin, given the same prior p.

However, a Bayesian agent **must commit to a specific prior p**, which in this
setting amounts to guessing how likely COPY vs REVERSE HEADS is. If the guess
is wrong, the Bayesian can be exploited.

---

## 5. The Naive Infra-Bayesian Update (and Why It Fails)

A naive approach to infra-Bayesian updating would be:

> "I observe heads. Both environments assign equal probability to heads.
> So I still have Knightian uncertainty over {COPY, REVERSE HEADS}.
> I should maximize worst-case utility *conditional on heads*."

Under this naive update:

- Say "heads" | H: COPY gives 1, REVERSE HEADS gives 0 → worst case = **0**
- Say "tails" | H: COPY gives 0, REVERSE HEADS gives 1 → worst case = **0**

Both options have worst case 0. The agent is completely stuck — it has no
basis to choose. The naive update **throws away the information about what
the agent is already committed to doing in the other branch** (i.e., what it
will say when the coin is tails).

Worse, this creates **dynamic inconsistency**. Suppose the agent originally
committed to "Always Tails" (the maximin policy). After observing heads,
the naive update tells it "heads" and "tails" are equally good (both worst
case 0). But switching to "heads" after heads would change the overall policy
to "Copy the Coin" — which has worst case 0.25, **worse** than the 0.5 it
originally guaranteed. Past-you and future-you disagree.

---

## 6. The Infra-Bayesian Update Rule (sa-Measures)

The solution is to track **off-history expected utility** alongside the
conditional measure. This is formalized as an **sa-measure** (signed affine
measure): a pair (m, b) where:

- **m** is a (possibly unnormalized) measure over future outcomes given the
  observation
- **b ≥ 0** is the expected utility the agent has already "locked in" from
  the parts of the policy that don't depend on this observation (the
  off-history branches)

### The update procedure

1. **Pre-commit** to a full policy π (what to do on every possible observation).
2. **On observing h**, for each environment e in the credal set, compute:
   - **m_e**: the (unnormalized) conditional measure of e given h
   - **b_e**: the expected utility from the branches of π that don't pass
     through h, weighted by their probability under e
3. **Choose action** to maximize the worst case of m_e(action utility) + b_e
   across all environments e in the credal set.

### Worked example: committed to "tails on tails," observing Heads

Suppose the agent has committed to saying "tails" when tails is observed.
After observing heads, the sa-measures for each environment are:

**COPY (m_COPY, b_COPY):**
- b_COPY = (utility from tails branch) = P(T) * reward("tails" | T, COPY) = 0.5 * 1 = 0.5
- m_COPY assigns weight P(H) = 0.5 to the heads outcome

**REVERSE HEADS (m_RH, b_RH):**
- b_RH = P(T) * reward("tails" | T, REVERSE HEADS) = 0.5 * 0.5 = 0.25
- m_RH assigns weight P(H) = 0.5 to the heads outcome

Now evaluate saying "heads" vs "tails" after observing heads:

**Say "heads" after H:**
- COPY: 0.5 * 1 + 0.5 = **1.0**
- REVERSE HEADS: 0.5 * 0 + 0.25 = **0.25**
- Worst case: **0.25**

**Say "tails" after H:**
- COPY: 0.5 * 0 + 0.5 = **0.5**
- REVERSE HEADS: 0.5 * 1 + 0.25 = **0.75**
- Worst case: **0.5**

The maximin choice is **"tails"** (worst case 0.5 > 0.25), which is exactly
what the original "Always Tails" policy prescribed. **Dynamic consistency is
preserved.**

Note that the total worst-case values (0.5 for "tails," 0.25 for "heads")
match exactly the worst-case full-policy evaluations from Section 3. The
b-term carries forward the off-history commitment and prevents the agent from
fooling itself into switching plans.

### Worked example: committed to "heads on tails," observing Heads

Now suppose the agent committed to saying "heads" when tails is observed.

**COPY:** b_COPY = P(T) * reward("heads" | T, COPY) = 0.5 * 0 = 0.0
**REVERSE HEADS:** b_RH = P(T) * reward("heads" | T, REVERSE HEADS) = 0.5 * 0.5 = 0.25

**Say "heads" after H:**
- COPY: 0.5 * 1 + 0.0 = **0.5**
- REVERSE HEADS: 0.5 * 0 + 0.25 = **0.25**
- Worst case: **0.25**

**Say "tails" after H:**
- COPY: 0.5 * 0 + 0.0 = **0.0**
- REVERSE HEADS: 0.5 * 1 + 0.25 = **0.75**
- Worst case: **0.0**

Maximin says "heads" (worst case 0.25 vs 0.0). The resulting full policy
is "Always Heads" (say heads on tails, heads on heads), which indeed has
worst case 0.25 — consistent with the full-policy analysis.

---

## 7. Toward a General Rule

### The General Bayesian Update Rule

For a Bayesian agent with prior p over environments {e_1, ..., e_n}:

1. **Observe** history h.
2. **Update** the prior to posterior using Bayes' rule:

       P(e_i | h) = P(h | e_i) * P(e_i) / Σ_j P(h | e_j) * P(e_j)

3. **Choose action** a to maximize:

       Σ_i P(e_i | h) * reward(a | h, e_i)

This works because the Bayesian has a single prior, and Bayes' rule
guarantees dynamic consistency: the action that maximizes expected utility
given the posterior is always the continuation of the policy that maximizes
expected utility given the prior.

**Limitation**: Requires committing to a specific prior. Under model
misspecification or adversarial environments, the wrong prior leads to
arbitrarily bad outcomes.

### The General Infra-Bayesian Update Rule (sa-Measures)

For an infra-Bayesian agent with credal set C (a set of environments):

1. **Before any observations**, choose a full policy π* by solving the
   maximin problem:

       π* = argmax_π  min_{e ∈ C}  E_e[U(π)]

2. **On observing history h**, for each environment e ∈ C, construct the
   sa-measure (m_e, b_e):

       m_e = P_e(· | h)  (unnormalized conditional measure on future)

       b_e = Σ_{h' ∉ descendants(h)} P_e(h') * U(π*(h'), h', e)

   where b_e sums the expected utility from all branches of π* that do NOT
   pass through the current history h.

3. **Choose action** a to maximize:

       min_{e ∈ C}  [ m_e(reward(a | h, e)) + b_e ]

4. **Renormalize** if needed: after the raw update (restricting measures to
   histories consistent with h), scale back to probability-measure form.
   The key insight is that renormalization must preserve the b-term, so that
   the relative weighting of environments accounts for their off-history
   contributions.

### Why the b-term is necessary

Without b_e, the update reduces to the naive update of Section 5:

    min_{e ∈ C}  m_e(reward(a | h, e))

This throws away the information about what the agent has already committed
to doing on other branches. The adversary (Murphy) can then exploit this
amnesia by choosing different worst-case environments before vs. after the
observation, breaking dynamic consistency.

The b-term "anchors" the post-observation decision to the pre-observation
plan. It ensures that the environment that is worst-case after updating is
evaluated against the **full** policy, not just the conditional continuation.

### The pattern in general

The general principle is:

> **To update under Knightian uncertainty, you must carry forward the value
> of your commitments.** The conditional evaluation of an action is not just
> "how good is this action given what I've seen" but "how good is this action
> *plus everything I've already locked in* given what I've seen."

This is analogous to how, in game theory, the value of a subgame depends on
the strategy profile in the *entire* game, not just the subgame. The
sa-measure formalism makes this bookkeeping precise.

For finite environments with discrete observations, the rule simplifies to:

    V(a | h, π) = min_{e ∈ C}  [ P_e(h) * reward(a, h, e)  +  Σ_{h' ⊥ h} P_e(h') * reward(π(h'), h', e) ]

where h' ⊥ h means h' is a history not extending h (the "off-history"
branches), and π(h') is the action prescribed by the pre-committed policy.
The agent then picks a = argmax_a V(a | h, π).

This is the update rule that preserves the maximin guarantee across time
and makes infra-Bayesian agents dynamically consistent even under adversarial
uncertainty about which environment they inhabit.

---

## 8. The Game-Theoretic View: Was This Already Solved?

The entire setup is a **two-player zero-sum extensive-form game** between the
Agent and Nature ("Murphy"). Game theory has possessed the tools to solve this
since at least 1928 (von Neumann's minimax theorem). This section asks: does
the sa-measure machinery add anything beyond what standard game theory already
provides?

### 8.1 The game in extensive form

```
            Nature (hidden)
           /              \
        COPY          REVERSE HEADS
         |                  |
      Coin (fair)       Coin (fair)
       /    \            /     \
      H      T          H       T
      |      |          |       |
   Agent   Agent     Agent   Agent
   h/t     h/t       h/t     h/t
```

The agent cannot distinguish COPY-Heads from RH-Heads (or COPY-Tails from
RH-Tails), so it has two **information sets**: {observed H} and {observed T}.

### 8.2 Behavioral strategies: more than 4 policies

Sections 2-3 of this document restrict attention to the four **pure** (i.e.,
deterministic) policies. But game theory works with **behavioral strategies**:
the agent can randomize independently at each information set. A behavioral
strategy is a pair:

    (p_H, p_T) ∈ [0,1] × [0,1]

where p_H = probability of saying "heads" given heads, p_T = probability of
saying "heads" given tails. The four pure strategies are corners of this
square: (1,1), (0,0), (1,0), (0,1). But the full space is continuous.

### 8.3 Expected utility as a function of (p_H, p_T)

    EU_COPY(p_H, p_T) = 0.5 · [p_H · 1 + (1-p_H) · 0] + 0.5 · [p_T · 0 + (1-p_T) · 1]
                       = 0.5 · p_H + 0.5 · (1 - p_T)
                       = 0.5 · p_H + 0.5 - 0.5 · p_T

    EU_RH(p_H, p_T)   = 0.5 · [p_H · 0 + (1-p_H) · 1] + 0.5 · [p_T · 0.5 + (1-p_T) · 0.5]
                       = 0.5 · (1 - p_H) + 0.5 · 0.5
                       = 0.75 - 0.5 · p_H

**Key observation: EU_RH does not depend on p_T at all.** In REVERSE HEADS,
the tails branch pays 0.5 regardless of what you say. Only the heads branch
matters for distinguishing environments.

### 8.4 p_T = 0 is dominant: "say tails on tails" is free

Since EU_RH is independent of p_T, and EU_COPY *decreases* in p_T, we have:

- Lowering p_T (saying "tails" more often on tails) **strictly improves**
  COPY payoff while leaving RH payoff unchanged.

Therefore **p_T = 0 ("always say tails when you see tails") is weakly
dominant** — it is at least as good as any other choice in every environment,
and strictly better in COPY. No worst-case analysis or IB machinery is needed
to see this. It follows from plain dominance.

This is exactly the intuition: "any strategy where you say tails upon
observing tails is fine." The tails branch is solved. The only interesting
decision is on the heads branch.

### 8.5 The heads branch: where mixing helps

With p_T = 0 fixed, the game reduces to a single decision variable p_H:

    EU_COPY(p_H)  = 0.5 · p_H + 0.5
    EU_RH(p_H)    = 0.75 - 0.5 · p_H

The agent wants to maximize the worst case:

    max_{p_H}  min(0.5 · p_H + 0.5,  0.75 - 0.5 · p_H)

Setting the two expressions equal:

    0.5 · p_H + 0.5 = 0.75 - 0.5 · p_H
    p_H = 0.25

At this point both environments give utility:

    0.5 · 0.25 + 0.5 = **0.625**

### 8.6 The game-theoretic optimum beats the IB pure-strategy optimum

| Strategy                        | Worst-case EU |
|---------------------------------|---------------|
| Always Tails (IB pure optimum)  | 0.500         |
| **(p_H=0.25, p_T=0) (mixed)**  | **0.625**     |

The mixed behavioral strategy achieves worst-case utility **0.625**, which is
**25% better** than the "Always Tails" pure-strategy maximin of 0.5.

By restricting attention to pure strategies, the IB exposition leaves value
on the table. The minimax theorem guarantees that mixed strategies can always
do at least as well, and here they do strictly better.

### 8.7 Verification: the minimax theorem

Von Neumann's minimax theorem (1928) says that in finite two-player zero-sum
games, maximin = minimax when mixed strategies are allowed. Let's verify.

**Nature's side**: Nature mixes over {COPY, RH} with probability q on COPY.
The agent best-responds to each q. Nature minimizes the agent's
best-response value. With p_T = 0 fixed:

    Agent's EU given q = q · (0.5·p_H + 0.5) + (1-q) · (0.75 - 0.5·p_H)

Agent maximizes over p_H: the coefficient on p_H is q·0.5 - (1-q)·0.5 =
0.5·(2q - 1). So agent picks p_H = 1 if q > 0.5, p_H = 0 if q < 0.5.

- At q = 0.5: agent is indifferent, EU = 0.5·(0.5·p_H + 0.5) + 0.5·(0.75 - 0.5·p_H)
  = 0.25·p_H + 0.25 + 0.375 - 0.25·p_H = **0.625** (for any p_H)

Nature sets q = 0.5 and the agent cannot do better than 0.625. The minimax
value is **0.625**, matching the maximin. ✓

The saddle point is: Nature plays (0.5, 0.5), Agent plays (p_H=0.25, p_T=0).
At this equilibrium, neither player can improve by deviating.

### 8.8 What the min operator "erases" and why pure-strategy maximin is crude

The user's intuition deserves formal elaboration. In the pure-strategy
analysis, after observing heads, the naive worst-case for *each* action is 0:

    min(reward("heads"|H, COPY), reward("heads"|H, RH)) = min(1, 0) = 0
    min(reward("tails"|H, COPY), reward("tails"|H, RH)) = min(0, 1) = 0

The min operator is choosing a **different** worst-case environment for each
action. Against "heads" it picks RH; against "tails" it picks COPY. This is
unreasonably pessimistic — Nature can only be in ONE environment, not switch
adversarially per-action.

With **mixed strategies**, the agent forces Nature to commit. If the agent
plays p_H = 0.25, Nature cannot simultaneously be in the bad environment for
heads AND the bad environment for tails. The mixture hedges, and the
worst-case value rises from 0 to 0.625.

This is the fundamental insight of the minimax theorem applied to extensive
forms: **mixing at information sets creates strategic coupling that
constrains the adversary.**

### 8.9 Game theory already handles "dynamic consistency"

The IB sequence motivates sa-measures by the need for dynamic consistency
under Knightian uncertainty. But in game theory, this issue simply does not
arise when you solve the game correctly:

1. **Kuhn's theorem (1953)**: In extensive-form games with perfect recall,
   behavioral strategies and mixed strategies are equivalent. There is no
   gap between "pre-commit to a full policy" and "decide at each information
   set" — they yield the same set of outcome distributions.

2. **Sequential rationality**: The minimax behavioral strategy (p_H = 0.25,
   p_T = 0) is **sequentially rational** at every information set. After
   observing heads, the agent has no incentive to deviate from p_H = 0.25,
   because that mixing probability is part of the global equilibrium. After
   observing tails, saying "tails" is dominant. No recalculation or
   off-history bookkeeping is needed.

3. **Subgame perfection**: In the extensive form, the "after observing
   heads" subtree and the "after observing tails" subtree can be analyzed
   using backward induction. The globally optimal behavioral strategy is
   automatically locally optimal at each node.

The "dynamic inconsistency" that IB's Section on sa-measures addresses arises
specifically from applying **pointwise worst-case analysis** at each
information set — i.e., letting Nature pick a different environment at each
decision point. This is a self-inflicted problem. Game theory avoids it by
solving the game as a whole with behavioral strategies, where Nature must
commit to one environment (or one mixture).

### 8.10 Historical precedent

The key ideas were established decades before infra-Bayesianism:

- **von Neumann (1928)**: Minimax theorem for zero-sum games.
- **Wald (1950)**: Minimax decision theory — the direct predecessor of
  maximin over credal sets. Wald explicitly framed statistical decision
  problems as zero-sum games against Nature.
- **Kuhn (1953)**: Behavioral strategies in extensive-form games.
- **Harsanyi (1967-68)**: Games of incomplete information, where Nature
  moves first to select the "type" (= environment). This is exactly the
  structure of Copy/Reverse Heads.
- **Gilboa & Schmeidler (1989)**: Maximin expected utility with multiple
  priors — the decision-theoretic foundation for acting under credal sets.
  This is essentially the decision rule IB uses.

### 8.11 So is infra-Bayesianism necessary?

For **this example**: no. The game-theoretic solution is cleaner, more
powerful (it finds the mixed optimum at 0.625 rather than the pure-strategy
0.5), and doesn't require the sa-measure machinery.

For the **broader IB program**: the answer is more nuanced. IB aims to
extend these ideas to:

- **Infinite / continuous** environment spaces (where "enumerate all
  environments" isn't feasible)
- **Non-realizable** settings (where no environment in the credal set is
  the true one)
- **Learning** (tightening the credal set online from observations)
- **Measure-theoretic** foundations suitable for general RL

Standard finite game theory doesn't directly address these settings. The
sa-measure formalism is attempting to provide a **general-purpose update
rule** that works in the measure-theoretic RL setting, not just in finite
games.

However, the Copy/Reverse Heads example — intended as a motivating showcase —
actually undersells the approach by restricting to pure strategies and missing
the cleaner game-theoretic solution. The example demonstrates a problem
(dynamic inconsistency under naive worst-case updates) that game theory
already solves, and then offers a more complex fix (sa-measures) than the
existing one (behavioral strategies in extensive-form games).

### 8.12 Summary: two views of the same problem

| Aspect                   | Game Theory                          | Infra-Bayesianism           |
|--------------------------|--------------------------------------|-----------------------------|
| Formalism                | Extensive-form zero-sum game         | Credal set + maximin        |
| Strategy space           | Behavioral strategies (p_H, p_T)     | 4 pure policies             |
| Optimal worst-case value | 0.625                                | 0.5                         |
| Dynamic consistency      | Free (Kuhn's theorem)                | Requires sa-measures        |
| Sequential decisions     | Backward induction                   | Off-history bookkeeping (b) |
| On heads branch          | Mix with p_H = 0.25                  | "Doesn't matter" / say tails|
| Historical origin        | 1928-1989                            | 2020                        |
| Best suited for          | Finite games, known structure        | General RL under ambiguity  |
