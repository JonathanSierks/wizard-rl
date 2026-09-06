# wizard-rl

Self-play reinforcement learning for **Wizard**, implemented end to end: rules
engine, observation encoding, policy-gradient training with a learned baseline,
and an instrumentation layer for diagnosing training pathologies.

The agent reaches positive scores against random opponents. Its bidding head,
however, converges to a narrow band of bids and does not recover; the majority of
this work concerns identifying the mechanism and testing interventions against
it.

```bash
python play.py          # play against the trained agent (CLI)
python train.py         # train from scratch
tensorboard --logdir rl_runs
```

---

## 1. Task

Wizard is a trick-taking card game. The deck holds 60 cards: four suits of
thirteen, plus four wizards (win any trick) and four fools (lose any trick).
Three players play 20 rounds; round *r* deals *r* cards to each player. Each
round has two phases:

1. **Bidding** — each player announces the exact number of tricks they will win.
2. **Play** — the round is played out, following suit where possible.

Scoring admits no partial credit. An exact bid scores `20 + 10 × bid`; any
deviation costs `10 × |bid − won|`. The final bidder may not choose a value that
makes the bids sum to *r*, so at least one player is guaranteed to miss.

### Properties relevant to RL

| Property | Consequence |
|---|---|
| Imperfect information | Opponents' hands are unobserved; only bids and played cards are available. |
| Two coupled decisions, one reward | The round return depends jointly on one bid and on up to 20 card decisions. |
| Non-stationary round structure | A 1-card and a 20-card round are structurally different tasks; each round size is observed equally rarely. |
| Self-play non-stationarity | The opponent distribution shifts as the agent improves. |

The combination of a sparse terminal reward and two decision types sharing that
reward makes credit assignment the central difficulty.

---

## 2. Method

### Observation encoding

A 317-dimensional vector: hand (60, multi-hot), cards already played (60), trump
suit (5, one-hot), current trick (3 × 60), all players' bids and tricks won
(3 × 3), plus round number, bidding position, and a bid/play indicator.

All player-indexed fields are encoded **relative to the acting player**, so a
strategy learned in one seat transfers to the others rather than being relearned
per seat.

### Architecture

A shared trunk (317 → 256 → 256, ReLU) with four linear heads:

| Head | Output | Trained by |
|---|---|---|
| bid | 21 | policy gradient |
| play | 60 | policy gradient |
| value | 1 | MSE against the round return |
| tricks | 21 | cross-entropy against observed tricks won |

Illegal actions are masked to `-inf` prior to the softmax, so the policy cannot
violate the rules and does not expend capacity learning them.

### Training

REINFORCE with a learned value baseline (advantage actor-critic), Monte-Carlo
returns, γ = 1. Three copies of the current network play against each other.
Each update collects 20 games — approximately 13,800 transitions, of which
roughly 1,200 are bidding decisions.

Entropy bonuses are applied per head (β_bid = 0.05, β_play = 0.01) after the
bidding and card policies were found to require different amounts of
regularisation. Returns are divided by 50 so that the value loss does not
dominate the gradient reaching the shared trunk.

---

## 3. Instrumentation

The pathologies described in section 4 are not visible in loss curves or
aggregate scores. The following instrumentation was added incrementally; each
item exists because a specific ambiguity could not be resolved without it.

**Per-head loss decomposition.** Return, entropy, value and auxiliary terms are
logged separately. The summed loss is uninformative — its components differ by
orders of magnitude and carry opposite signs.

**Gradient norms per head and trunk.** The four heads share a trunk, so their
gradients are summed before the optimiser normalises them. The ratio of these
norms is the only direct measure of which objective is shaping the shared
representation.

**Forced-move filtering.** Transitions with exactly one legal action contribute
zero gradient but were diluting the logged entropy and distorting the maximum
logit. Metrics are computed over decisions with more than one legal option;
`n_decisions` is logged as a control.

**Normalised entropy.** Raw entropy is not comparable across round sizes — its
maximum is `log(n_legal)`. Metrics are reported as `H / log(n_legal)` ∈ [0, 1].

**Rank ratio.** Bid logits are collected across many hands and the
column-centred matrix is decomposed by SVD. The ratio `s₀ / Σs` measures the
fraction of output variation lying in a single direction; a value near 1
indicates the head has stopped distinguishing situations.

**Per-round-size error decomposition.** Bid error is reported as three separate
quantities per round size: signed mean (systematic bias), mean absolute error
(precision), and exact-hit rate. An agent can be unbiased and imprecise, and the
distinction determines which intervention applies.

**Behavioural probing** (`model_analysis.ipynb`). A frozen checkpoint is queried
with synthetic observations in which a single field varies. This yields
attribution over the 60 card inputs, trump sensitivity tests, and per-class logit
statistics. Linear probes fitted to the frozen trunk activations separate
"information absent from the representation" from "information present but unused
by the head".

**Evaluation protocol.** Every 50 updates, 200 games against two random agents
and 200 against a frozen earlier checkpoint. Evaluation runs on a fixed seed with
the RNG state saved and restored, so identical deals are replayed at every
measurement point and differences between checkpoints are attributable to the
network rather than to the cards.

---

## 4. Findings

### 4.1 The bidding head does not exceed a fixed ceiling

Across 12,650 updates the greedy bid in 20-card rounds never exceeds 6, against a
structural expectation of `r/3 = 6.67` tricks per player. The mean bid *declines*
over training, from 5.4 to 3.25.

An entropy bonus — the conventional response to premature determinism — was
applied first. It held entropy approximately constant and lowered the ceiling
from 5 to 4 while slowing convergence. The symptom was addressed; the mechanism
was not.

### 4.2 The head collapses to a one-dimensional output

![collapse](docs/figures/fig2_collapse.png)

Normalised entropy falls from 1.0 to 0.11, the maximum absolute logit grows
without saturating to 58, and the rank ratio rises from 0.26 to 0.77. Bid logit
vectors from three players holding entirely different hands are near-identical
within a round, differing approximately by a scalar that tracks round size.

### 4.3 The upper bid classes are frozen rather than suppressed

Per-class logit standard deviation across hands separates cleanly: classes 1–6
vary with the hand (σ ≈ 2–10), classes 7–20 sit at σ ≈ 1.2 — the level produced
by unlearned pass-through of the input. The head bias is approximately uniform
across all 21 classes, so the upper classes are not being actively penalised.

The mechanism is in the gradient. For a class that was not sampled,

```
∂L/∂z_k = A · p_k
```

which is proportional to the class's own probability. Once `p(bid 7) ≈ e⁻²⁰`,
the gradient is numerically zero in both directions and the class cannot be
recovered by adjusting learning rate or entropy weight.

### 4.4 The cause is sample scarcity per round size

Approximately 1,200 bidding decisions per update, distributed across 20 round
sizes, yields 60 per size. The bid classes are absolute: what is learned about
bidding 1 in 3-card rounds does not transfer to bidding 7 in 20-card rounds.

Training exclusively on rounds of 18 or more cards — 1,200 samples concentrated
in one round size — removed the ceiling immediately and produced bids in the 7–9
range. This isolates data availability rather than optimisation as the binding
constraint.

### 4.5 Weighted round sampling is a partial remedy

![by round size](docs/figures/fig4_by_round_size.png)

Sampling round sizes proportionally to `r²` rather than cycling 1→20 raised the
ceiling from 5 to 6 and approximately halved the rate of logit growth. It did not
resolve the problem: `r²` weighting yields roughly 2.8× more samples for round
20, where the isolating experiment used 20×.

Small rounds were not degraded by the reweighting, contrary to expectation —
signed bias at *r* = 3 remained within ±0.2 throughout.

### 4.6 Replacing bandit feedback with full-information feedback

Policy gradient learns only about the action taken. The Wizard scoring function,
however, is known analytically, so the only unknown quantity is how many tricks a
hand will take — and that is **observed every round, independent of what was
bid**.

A fourth head predicts `p(tricks won = k | s)`, trained by cross-entropy against
the observed count. Its gradient is

```
∂L/∂z_k = q_k − 1[k = won]
```

with no `p_k` factor: an improbable but correct class receives the largest
update rather than none. The label is available for every transition in the
round, not only the bidding one, giving roughly 13,800 training signals per
update in place of 1,200.

The bid is then computed rather than learned, by maximising expected points under
the predicted distribution and the known scoring rule:

```
EV(b) = q_b · (20 + 10b) − 10 · Σ_k q_k |b − k|
```

![ceiling](docs/figures/fig1_bid_ceiling.png)

The ceiling is removed: the rule bids up to 10 in 20-card rounds, where the
policy head remained pinned at 6, and prediction error at bid time falls to
approximately 1.3 tricks. Calibration is not resolved. The agent still under-bids
by roughly 2 tricks in large rounds, because the labels are generated by its own
conservative play — the play head observes the announced bid and plays toward it,
so the target depends on the decision being evaluated.

### 4.7 The evaluation opponent determines the conclusion

![opponent dependence](docs/figures/fig3_opponent_dependence.png)

The two evaluation protocols disagree over the same checkpoints. Against the
frozen opponent, score rises monotonically to +236 per game and bid calibration
in 20-card rounds improves from −1.36 to −0.69. Against random opponents, score
peaks near update 2,000 and returns to zero while calibration degrades from −1.21
to −3.17.

Both measurements are correct. The agent has converged to bidding low and
avoiding tricks, and whether that is well calibrated depends on how aggressively
the opponents compete for tricks — the *r* tricks in a round are distributed
regardless. The frozen opponent bids around 6.5 and competes; random agents do
not.

The frozen opponent is used only for evaluation and never appears in training, so
this is not opponent-specific learning. It is a measurement artefact: the metric
increasingly reflects the frozen opponent's weakness rather than the agent's
strength.

---

## 5. Limitations

- **Single-seed results.** Two runs of an identical configuration differed by 40
  points at the same update count. Effects smaller than that are not currently
  measurable, and all figures above should be read as one draw.
- **A single frozen evaluation opponent** rather than a pool, with the
  consequences described in 4.7.
- **The trick-count predictor is confounded** by the bid it is used to produce
  (4.6).
- **The value head contributes disproportionate gradient.** In the most recent
  configuration its gradient norm exceeds that of the policy heads by a factor of
  roughly 17 while explaining under 10% of return variance.

---

## 6. Methodological notes

- Aggregate metrics concealed selective failure throughout. Overall bid accuracy
  remained near 0.28 while small rounds approached good calibration and large
  rounds degraded steadily; the decomposition by round size was necessary to see
  either.
- A collapsed action class is a gradient-flow problem rather than a
  hyperparameter one. This determined which interventions were worth attempting:
  a hyperparameter search over learning rate and entropy weight cannot revive a
  class whose gradient is proportional to its own vanishing probability.
- Interventions were applied one per run, following an instance in which three
  simultaneous changes produced an uninterpretable result.
- One finding reported here was subsequently retracted. An analysis concluding
  that the agent ignored trump suit had been run against a reconstructed copy of
  the encoder rather than the project's own `encode()`. Re-run correctly, the
  trump effect was clearly present. The conclusion had stood for two days and had
  informed planned architecture changes.
- Behavioural probing of frozen checkpoints located blind spots that no training
  curve indicated.

---

## 7. Planned work

- **Ablation grid across seeds.** Five configurations (baseline, + value head,
  + `r²` sampling, + auxiliary trick head, + EV bidding) × 3 seeds, executed in
  parallel, to attach confidence intervals to the claims in section 4.
- **An opponent pool** sampled from checkpoints across training, replacing the
  single frozen reference.
- **`p(won | s, bid)`** in place of `p(won | s)`, removing the confound in 4.6 by
  conditioning the predictor on the bid and marginalising at decision time.
- **Permutation-invariant opponent encoding.** Representing each opponent as a
  feature tuple passed through a shared MLP (DeepSets or attention) would provide
  parameter sharing across opponents and support variable player counts.
- **Card embeddings.** "Red 9 in hand" and "red 9 in the current trick" are
  presently unrelated input dimensions.

---

## 8. Repository

| Path | Contents |
|---|---|
| `game.py`, `cards.py`, `tricks.py` | rules engine |
| `observations.py` | encoding, relative to the acting player |
| `model.py` | shared trunk, four heads |
| `player.py` | RL, random and human agents |
| `train.py` | self-play loop, loss, evaluation |
| `model_analysis.ipynb` | probing, gradient attribution, rank test |
| `tests/` | invariant tests for the rules engine |
| `docs/investigation.md` | derivations underlying section 4 |
