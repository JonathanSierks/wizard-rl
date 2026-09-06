# wizard-rl

A self-play reinforcement learning agent for **Wizard**, built from scratch: game
engine, observation encoding, REINFORCE with a value baseline, and a diagnostic
toolkit. The agent learns to play and beats random opponents — but its **bidding
head collapses to a narrow range of bids and never recovers**.

Most of this repository is about finding out why.

```bash
python play.py          # play a game against the trained agent (CLI)
python train.py         # train from scratch
```

---

## The game, and what makes it hard for RL

Wizard is a trick-taking game. 60 cards: four suits of 13, plus 4 wizards (always
win) and 4 fools (always lose). Three players, 20 rounds — round *r* deals *r*
cards each. Every round has two phases:

1. **Bid** — announce exactly how many tricks you will take.
2. **Play** — play out the round, following suit when possible.

Scoring is what makes bidding hard: hitting your bid exactly gives
`20 + 10 × bid`; missing costs `10 × |bid − won|`. There is no partial credit.
Bidding 7 and taking 6 is worse than bidding 6 and taking 6, and the *last*
bidder may not make the bids sum to *r* ("screw the dealer"), so someone is
always forced to be wrong.

Four properties make this an awkward RL problem:

- **Imperfect information.** Opponents' hands are hidden; the only signal is what
  they bid and what they have played.
- **Two coupled decisions, one reward.** The round reward depends on the bid and
  on 1–20 subsequent card decisions. Credit assignment has to split a single
  number across both.
- **The task changes shape.** A 1-card round and a 20-card round are almost
  different games, and the agent sees each round size equally rarely.
- **Non-stationary self-play.** The opponents improve as the agent does, so the
  distribution the agent is fitting moves under it.

---

## Setup

**Encoding** — 317 dims: hand (60 multi-hot), cards already played (60), trump
(5 one-hot), current trick (3 × 60), all players' bids and tricks won (3 × 3),
plus round number, bidding position and a bid/play flag. Everything is encoded
*relative to the acting player*, so a strategy learned in one seat transfers to
the others.

**Network** — a shared trunk (317 → 256 → 256, ReLU) with four linear heads:
bid (21), play (60), value (1), and — added late in the project — a trick-count
predictor (21). Illegal actions are masked to `-inf` before the softmax, so the
policy cannot break the rules and never has to learn them.

**Training** — REINFORCE with a learned value baseline (i.e. advantage
actor-critic), Monte-Carlo returns, γ = 1. Self-play with three copies of the
current network. Each update collects 20 games ≈ 13,800 transitions, of which
~1,200 are bids. Per-head entropy bonuses (β_bid = 0.05, β_play = 0.01),
returns scaled by 50 so the value loss does not dominate the shared trunk.

**Evaluation** — every 50 updates, 200 games against two random agents and 200
against a frozen earlier checkpoint, on a fixed evaluation seed so the same card
deals are replayed each time.

---

## What happened

The agent reaches positive scores against random opponents after ~600 updates
and plateaus. The interesting part is what it never learns.

**Symptom.** In 20-card rounds the agent never bids more than 5, over thousands
of updates and a million games. The structural expectation is 6.67 tricks per
player, so this is not caution — it is a wall.

**First attempt, and a useful failure.** An entropy bonus on the bid head was the
obvious fix. It kept the *entropy* stable and made the ceiling *worse*: bids
maxed out at 4 instead of 5, and learning slowed. Symptom treated, cause
untouched.

**Diagnosis 1 — the head stops distinguishing situations.** Collecting bid logits
across many hands and running an SVD on the column-centred matrix gives
`s₀/Σs`: the fraction of variation living in a single direction. It rises from
0.26 to 0.77. Three players with completely different hands produce nearly
identical logits, scaled only by round size.

**Diagnosis 2 — the upper bid classes are frozen, not suppressed.** The standard
deviation of each class's logit across hands separates cleanly: classes 1–6 vary
with the hand (σ ≈ 2–10), classes 7–20 sit at σ ≈ 1.2, the level of unlearned
pass-through. The reason is in the REINFORCE gradient: for a class that was not
sampled, `∂L/∂z_k = A · p_k`. It is *proportional to the class's own
probability*. Once `p(bid 7) ≈ e⁻²⁰`, the gradient is numerically zero and the
class can never come back, no matter how good it would be.

![collapse](docs/figures/fig2_collapse.png)

**Diagnosis 3 — too few samples per round size.** ~1,200 bids per update spread
across 20 round sizes is 60 each, and the bid classes are *absolute*: what the
agent learns about "bid 1" in 3-card rounds does not transfer to "bid 7" in
20-card rounds. Training exclusively on rounds of 18+ cards (1,200 samples in one
round size) removed the ceiling immediately — confirming a data problem rather
than an optimisation one.

**Intervention — weighted round sampling.** Sampling round sizes ∝ r² instead of
the fixed 1→20 ramp raised the ceiling from 5 to 6 and roughly halved the rate of
logit growth. It did not solve the problem: r² gives round 20 about 2.8× more
samples, where the successful experiment had 20×.

![by round size](docs/figures/fig4_by_round_size.png)

**Reframe — the feedback is the problem, not the optimiser.** REINFORCE only ever
learns about the action it took. But Wizard's scoring function is *known
analytically*, so the only unknown is how many tricks a hand will take — and that
is **observed every round, regardless of what was bid**. That turns a bandit
problem into a supervised one.

A fourth head predicts `p(tricks won = k | s)`, trained with cross-entropy
against the observed count. Its gradient is `q_k − 1[k = won]` — no `p_k` factor,
so an unlikely-but-correct class gets the *strongest* push rather than none. The
bid then becomes arithmetic: pick the bid maximising expected points under the
predicted distribution and the known scoring rule.

![ceiling](docs/figures/fig1_bid_ceiling.png)

This works for exactly what it was built for. The ceiling disappears — the EV
rule bids up to 10 in 20-card rounds, where the policy head was pinned at 6 —
and prediction error at bid time drops to ~1.3 tricks. It does **not** fix
calibration: the agent still under-bids by ~2 tricks in large rounds, because the
labels it learns from are generated by its own ducking play style.

---

## The result that changed how I read every other number

Halfway through, evaluation was extended to a second opponent: a frozen earlier
checkpoint, alongside the random baseline. The two disagree completely.

![opponent dependence](docs/figures/fig3_opponent_dependence.png)

Same agent, same checkpoints. Against the frozen opponent, score climbs to +236
and bid calibration *improves* monotonically. Against random opponents, score
peaks around update 2,000 and returns to zero while calibration *degrades* to
−3 tricks.

Both measurements are correct. The agent has learned to bid low and duck — and
whether that is calibrated depends on whether the opponents take tricks
aggressively. The frozen opponent does; random agents do not. The 20 tricks in a
round have to go somewhere.

Read `score_vs_rl` alone and this project looks like a success story.

---

## Lessons learned

- **The choice of evaluation opponent can invert your conclusion.** Two honest
  metrics on the same checkpoints told opposite stories for 10,000 updates.
- **Aggregate metrics hide selective failure.** Overall bid accuracy sat at 0.28
  the whole time. Split by round size, small rounds were near-perfect and large
  rounds were getting steadily worse.
- **A collapsed action class is a gradient problem, not a hyperparameter one.**
  No entropy bonus or learning rate revives a class whose gradient is
  proportional to its own vanishing probability.
- **One change per run, or you learn nothing.** The one time three things changed
  at once, the result was uninterpretable and the run had to be repeated.
- **Check what your analysis code actually measures.** One finding here — that
  the agent ignored trump — was wrong: it came from a reconstructed copy of the
  encoder rather than the real one. Re-run against the actual `encode()`, the
  effect was clearly present. The conclusion had been confident and public for
  two days.
- **Behavioural tests beat loss curves for finding blind spots.** Probing the
  frozen network with synthetic hands, linear probes on the trunk, and gradient
  attribution over the 60 card inputs found things no training curve showed.

---

## To be continued

- **Multi-seed runs.** Everything here is a single seed. Two runs of an identical
  configuration once differed by 40 points, so small effects are not measurable
  yet.
- **An opponent pool** instead of one frozen checkpoint, to stop the evaluation
  from measuring one opponent's weaknesses.
- **`p(won | s, bid)` instead of `p(won | s)`.** The current predictor is
  confounded: the play head sees the bid and plays to it, so the labels depend on
  the decision being evaluated.
- **Permutation-invariant architecture.** Encoding each opponent as a small
  feature tuple through a shared MLP (DeepSets / attention) would give parameter
  sharing across opponents and support variable player counts.
- **Card embeddings.** "Red 9 in hand" and "red 9 in the current trick" are
  currently unrelated input neurons.

---

## Repository

| | |
|---|---|
| `game.py`, `cards.py`, `tricks.py` | rules engine, verified by invariant tests |
| `observations.py` | encoding, relative to the acting player |
| `model.py` | shared trunk, four heads |
| `player.py` | RL / random / human agents |
| `train.py` | self-play loop, loss, evaluation |
| `model_analysis.ipynb` | probing, gradient attribution, rank test |
| `docs/investigation.md` | the full derivations behind the summary above |
| `tests/` | pytest invariants for the game engine |
