![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![Field](https://img.shields.io/badge/Field-Reinforcement%20Learning-blue)
![Task](https://img.shields.io/badge/Task-Self--Play%20%2F%20Imperfect%20Information-orange)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![TensorBoard](https://img.shields.io/badge/TensorBoard-FF6F00?logo=tensorflow&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white)

# Self-Play Reinforcement Learning for Wizard — Diagnosing a Policy Collapse

This project trains an agent to play **Wizard**, a trick-taking card game in which
players must first *announce* how many tricks they will win and then play to hit
that number exactly. Environment, observation encoding, training loop and
diagnostics are implemented from scratch in Python and PyTorch.

The agent learns to play competently and beats random opponents. The main problem was, however, that the **bidding policy collapses**: within a few hundred updates it stopped distinguishing
between hands and never announces more than 6 tricks in a 20-card round, where
the structural expectation is 6.67. Most of this work focused on identifying why and solving the problem,
through **five training configurations of increasing complexity**, each one a
response to a specific failure diagnosed in the previous. The final configuration
removes the ceiling by replacing the learned bidding policy with a supervised
trick-count predictor and an analytic decision rule. This solved the bidding problem, yet the overall performance against the heuristic agent baseline leaves room for further improvement.

**[▶ Play against the trained agent](#how-to-run)**

---

## Context

### The game

60 cards: four suits of thirteen, four **wizards** (win any trick) and four
**fools** (lose any trick). Three players play 20 rounds; round *r* deals *r*
cards each. Every round has two phases — **bid**, then **play**.

Scoring gives no partial credit:

$$
\text{points}(b, w) =
\begin{cases}
20 + 10b & b = w \\
-10\,|b - w| & b \neq w
\end{cases}
$$

where $b$ is the announced bid and $w$ the tricks actually won. Bidding 7 and
taking 6 scores worse than bidding 6 and taking 6. The last bidder may not choose
a value that makes all bids sum to $r$, so at least one player is always forced
to miss.

### As a reinforcement learning problem

| Property | Consequence |
|---|---|
| **Imperfect information** | Opponents' hands are hidden; only their bids and played cards are observable. |
| **Two coupled decisions, one reward** | The round return depends jointly on one bid and up to 20 card decisions. A single scalar must be split across both. |
| **Sparse, delayed reward** | Nothing is observed until the round ends, up to 20 tricks after the bid. |
| **The task changes shape** | A 1-card and a 20-card round are structurally different problems, and each round size is seen equally rarely. |
| **Self-play non-stationarity** | The opponents improve as the agent does, so the distribution being fitted moves underneath it. |

The bid is the hard part. It is a **single decision whose consequence is
determined by 20 later decisions**, in a game where the reward function punishes
near-misses as harshly as wild ones. Card play, by contrast, receives dense and
well-conditioned feedback.

---

## Method

### Observation encoding

A 317-dimensional vector per decision:

| Block | Dims | Content |
|---|---|---|
| hand | 60 | multi-hot over the deck |
| played | 60 | cards already played this round |
| trump | 5 | one-hot, including "no trump" |
| current trick | 180 | 3 × 60, by play order |
| bids and tricks won | 9 | 3 × (bid, won, has-bid flag) |
| meta | 3 | round size, bidding position, bid/play flag |

Player-indexed fields are encoded **relative to the acting player**, so a strategy
learned in one seat transfers to the others instead of being relearned per seat.

### Architecture

A shared trunk (317 → 256 → 256, ReLU) NN with four linear heads:

| Head | Output | Objective |
|---|---|---|
| bid | 21 | policy gradient |
| play | 60 | policy gradient |
| value | 1 | MSE against the round return |
| tricks | 21 | cross-entropy against observed tricks won |

Illegal actions are masked to $-\infty$ before the softmax, so the policy cannot
break the rules and spends no capacity learning them.

### Training and evaluation

REINFORCE with a learned value baseline (advantage actor-critic), Monte-Carlo
returns, $\gamma = 1$. Three copies of the current network play each other. One
update collects 20 games ≈ 13,800 transitions, of which ~1,200 are bids (later on this was increased to 30 games).

Evaluation runs every 50 updates on a **fixed seed with the RNG state saved and
restored**, so identical deals are replayed at every measurement point and
differences are attributable to the network rather than to the cards. Three
protocols: 200 games against random agents, a frozen earlier
checkpoint and a heuristic agent. Section [Results](#results) shows why running both mattered.

---

## Experiments

Five configurations, each additive, each motivated by a failure diagnosed in the
one before.

| | Configuration | Motivated by | Outcome |
|---|---|---|---|
| **A** | REINFORCE + per-head entropy bonus | — | Bids never exceed 5; logits grow without bound |
| **B** | + learned value baseline | High return variance suspected as the cause | Variance reduced, collapse unchanged |
| **C** | + `r²`-weighted round sampling | Diagnosis: too few bid samples per round size | Ceiling 5 → 6, collapse rate roughly halved |
| **D** | + trick-count head as auxiliary loss | Shared trunk under-represents hand strength | Prediction error at bid time → 1.3 tricks |
| **E** | + expected-value bidding rule | Reframe: the bid is a prediction problem, not a control problem | Ceiling removed (bids up to 10); calibration still off |

### A — REINFORCE baseline

The agent learns card play and reaches positive scores, but in 20-card rounds it
never bids above 5. An entropy bonus — the standard response to premature
determinism — was tried first. It held entropy roughly constant and made the
ceiling **worse** (4 instead of 5) while slowing convergence: the symptom was
treated, the mechanism untouched.

Two diagnostics were built to characterise the failure. Collecting bid logits
across many hands and taking the SVD of the column-centred matrix gives the
**rank ratio** $s_0 / \sum s$ — the share of output variation living in a single
direction. It rises from 0.26 to 0.77: three players holding entirely different
hands produce near-identical logits, differing by roughly a scalar that tracks
round size.

The second diagnostic is the **per-class logit standard deviation across hands**.
It separates cleanly: classes 1–6 vary with the hand ($\sigma \approx 2$–$10$),
classes 7–20 sit at $\sigma \approx 1.2$, the level of unlearned pass-through.
The head bias is near-uniform across all 21 classes, so the upper classes are not
being penalised — they are **frozen**. The reason is in the gradient itself. For a
class that was not sampled,

$$
\frac{\partial L}{\partial z_k} = A \cdot p_k
$$

which is proportional to the class's *own* probability. Once
$p(\text{bid } 7) \approx e^{-20}$, the gradient is numerically zero in both
directions and no learning rate or entropy weight can bring the class back.

![collapse](docs/figures/fig2_collapse.png)

### B — Value baseline

High Monte-Carlo return variance was the next hypothesis: round returns range
from −200 to +220, dominated by card luck rather than decision quality. A value
head replaces the batch-mean baseline with a state-conditional one,
$A = G - V(s)$.

It reduced variance and improved card play, but did not touch the ceiling —
consistent with the frozen-class diagnosis, since a better baseline still cannot
produce gradient for an action that is never sampled. The value head did confirm
the pathology from another angle: it learns to predict a *negative* return for
strong hands, which is correct under a policy that cannot bid high enough to use
them.

### C — Weighted round sampling

With ~1,200 bids per update spread across 20 round sizes, each size receives
about 60 samples. And the bid classes are **absolute**: what is learned about
bidding 1 in 3-card rounds does not transfer to bidding 7 in 20-card rounds.

This was tested directly by training exclusively on rounds of 18+ cards. The
ceiling disappeared immediately and bids landed in the 7–9 range, isolating
**data availability** rather than optimisation as the binding constraint.

Sampling round sizes $\propto r^2$ instead of cycling 1→20 raised the ceiling from
5 to 6 and roughly halved the rate of logit growth — but $r^2$ gives round 20 only
2.8× more samples where the isolating experiment had 20×. Small rounds were
*not* degraded by the reweighting, contrary to expectation.

![by round size](docs/figures/fig4_by_round_size.png)

### D and E — Reframing the bid as a prediction

Policy gradient only ever learns about the action it took. But Wizard's scoring
function is **known analytically** — so the only unknown is how many tricks a hand
will take, and that is *observed every round, regardless of what was bid*.

A fourth head predicts $p(w = k \mid s)$, trained by cross-entropy against the
observed count. Its gradient carries no $p_k$ factor:

$$
\frac{\partial L}{\partial z_k} = q_k - \mathbb{1}[k = w]
$$

so an improbable but correct class receives the *largest* update rather than
none. The label is available for every transition in the round, not just the
bidding one — about 13,800 training signals per update instead of 1,200.

In **D** this runs as an auxiliary loss while the policy head still decides. In
**E** the bid becomes arithmetic: maximise expected points under the predicted
distribution and the known scoring rule,

$$
\text{EV}(b) = q_b \,(20 + 10b) \;-\; 10 \sum_k q_k \,|b - k|
$$

The rule can output a bid of 10 without ever having played one, because the class
is *computed*, not learned.

![ceiling](docs/figures/fig1_bid_ceiling.png)

---

## Results

> Numbers below are from single runs. A 5 × 3 ablation grid (configurations
> A–E × three seeds) is in progress; see [Planned work](#planned-work).

| Config | Score vs. random | Bid accuracy | Bias @ r=20 | MAE @ r=20 | Max bid @ r=20 |
|---|---|---|---|---|---|
| A | | | | | |
| B | | | | | |
| C | | | | | |
| D | | | | | |
| E | | | | | |

*Bias* is mean(bid − tricks won): negative means systematic under-bidding.
*MAE* separates precision from bias — an agent can be unbiased and imprecise.

### The evaluation opponent determines the conclusion

The single most consequential finding was not about the agent but about how it
was measured.

![opponent dependence](docs/figures/fig3_opponent_dependence.png)

Same checkpoints, two evaluation protocols, opposite stories. Against the frozen
opponent, score climbs monotonically to +236 per game and bid calibration at
*r* = 20 **improves** from −1.36 to −0.69. Against random opponents, score peaks
near update 2,000 and returns to zero while calibration **degrades** from −1.21 to
−3.17.

Both measurements are correct. The agent has converged to bidding low and
avoiding tricks, and whether that is well calibrated depends on how aggressively
the opponents compete — the $r$ tricks in a round get distributed either way. The
frozen opponent bids around 6.5 and competes for them; random agents do not.

The frozen opponent never appears in training, so this is not opponent-specific
learning. It is a measurement artefact: the metric increasingly reflects the
*opponent's* weakness rather than the agent's strength. Read `score_vs_rl` alone
and this project looks like a success story.

### What the trick-count head fixed, and what it did not

![tricks head](docs/figures/fig5_tricks_head.png)

Prediction error at bid time falls to ~1.3 tricks and the bid ceiling disappears.
Calibration does not follow: the agent still under-bids by ~2 tricks in large
rounds, because the labels come from its own conservative play. The play head
observes the announced bid and plays toward it, so the training target depends on
the decision being evaluated — a confound the current design does not resolve.

---

## Lessons learned

- **Aggregate metrics hide selective failure.** Overall bid accuracy sat near 0.28
  throughout while small rounds were near-optimal and large rounds degraded
  steadily. Only the decomposition by round size showed either.
- **A collapsed action class is a gradient-flow problem, not a hyperparameter
  one.** This ruled out an entire class of interventions: no search over learning
  rate or entropy weight can revive a class whose gradient is proportional to its
  own vanishing probability.
- **Choose the evaluation opponent deliberately.** Two honest metrics disagreed
  about the same checkpoints for 10,000 updates.
- **One change per run.** The one time three things changed together, the run was
  uninterpretable and had to be repeated.
- **Verify what the analysis code measures.** One finding reported here was later
  retracted: an analysis concluding the agent ignored trump suit had been run
  against a *reconstructed* copy of the encoder rather than the project's own
  `encode()`. Re-run correctly, the trump effect was clearly present. The wrong
  conclusion had stood for two days and had already informed planned changes.
- **Probe the frozen model.** Synthetic-hand queries, linear probes on the trunk
  and gradient attribution over the 60 card inputs located blind spots that no
  training curve indicated.

## Planned work

- **5 × 3 ablation grid**, configurations A–E over three seeds, run in parallel,
  to attach confidence intervals to the table above. Two runs of an identical
  configuration once differed by 40 points, so effects below that are currently
  not measurable.
- **An opponent pool** sampled from checkpoints across training, replacing the
  single frozen reference.
- **$p(w \mid s, b)$ instead of $p(w \mid s)$** — conditioning the predictor on the
  bid and marginalising at decision time, to remove the confound above.
- **Permutation-invariant opponent encoding** (DeepSets / attention) for parameter
  sharing across opponents and variable player counts.
- **Card embeddings.** "Red 9 in hand" and "red 9 in the current trick" are
  currently unrelated input dimensions.

---

## Repository Structure

```
wizard-rl/
├── cards.py                  # deck, card representation, ordering
├── game.py                   # rules engine, round loop, scoring
├── tricks.py                 # trick resolution (wizard / fool / trump / follow suit)
├── observations.py           # 317-dim encoding, relative to the acting player
├── model.py                  # shared trunk + bid / play / value / tricks heads
├── player.py                 # RLAgent, RandomAgent, HeuristicAgent, HumanAgent
├── heuristic_agent.py        # bidding tables + trick helpers for HeuristicAgent
├── train.py                  # self-play loop, loss, evaluation, TensorBoard logging
├── play.py                   # CLI: play a game against a trained checkpoint
├── model_analysis.ipynb      # probing, gradient attribution, rank test
├── relevant_checkpoints/     # trained weights
├── docs/
│   ├── investigation.md      # full derivations behind the Experiments section
│   └── figures/
├── tests/                    # pytest invariants for the rules engine
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/JonathanSierks/wizard-rl.git
cd wizard-rl
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## How to Run

### Play against the agent

```bash
python main.py
```

Deals a full game against two copies of the trained agent. Your hand is shown
with card indices; you enter a bid and then an index per trick. Illegal choices
are rejected with the reason.

### Train from scratch

```bash
python train.py
tensorboard --logdir rl_runs
```

| Flag | Default | Purpose |
|---|---|---|
| `--config` | `ev` | Configuration A–E (`baseline`, `value`, `sampling`, `aux`, `ev`) |
| `--seed` | `0` | Random seed |
| `--updates` | `3000` | Number of self-play updates |
| `--tag` | — | Suffix appended to the run directory name |


## References

- Williams, R. J. (1992). *Simple statistical gradient-following algorithms for
  connectionist reinforcement learning*. Machine Learning 8, 229–256.
- Mnih, V., et al. (2016). *Asynchronous methods for deep reinforcement learning*.
  ICML, 1928–1937.
- Alain, G., & Bengio, Y. (2016). *Understanding intermediate layers using linear
  classifier probes*. arXiv:1610.01644.
- Ribeiro, M. T., et al. (2020). *Beyond accuracy: Behavioral testing of NLP models
  with CheckList*. ACL, 4902–4912.
- Jaderberg, M., et al. (2017). *Reinforcement learning with unsupervised auxiliary
  tasks*. ICLR.
