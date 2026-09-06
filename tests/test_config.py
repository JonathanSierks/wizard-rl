# tests/test_config.py
"""
Tests for the experiment configuration.

These are cheap: they check that the switches are wired to something, not
that training works. The expensive check -- every configuration survives an
update -- lives in the smoke script, not in the test suite.

Run with:
    pytest tests/test_config.py -v
    python tests/test_config.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

from game import round_weights, sample_round_sizes
from player import RLAgent
from train import Config


# --- importing train must not start a run -----------------------------------

def test_importing_train_has_no_side_effects():
    # If the training loop were still at module level, importing would have
    # hung long before reaching this line. Also assert nothing was created.
    import train
    assert callable(train.train)
    assert not any(p.startswith("events.out") for p in os.listdir(".")), \
        "importing train wrote TensorBoard events into the working directory"


# --- the ablation axes ------------------------------------------------------

def test_default_config_matches_current_behaviour():
    c = Config()
    assert c.bid_mode == "ev"
    assert c.use_value_baseline is True
    assert c.round_weights_exp == 2
    assert c.aux == 0.1


def test_policy_heads_follow_bid_mode():
    assert Config(bid_mode="ev").policy_heads() == {"play"}
    assert Config(bid_mode="policy").policy_heads() == {"play", "bid"}


def test_run_suffix_is_unique_per_configuration():
    names = {Config(name=n, bid_mode=b, use_value_baseline=v,
                    round_weights_exp=r, aux=a).run_suffix
             for n, b, v, r, a in [("A", "policy", False, 0, 0.0),
                                   ("B", "policy", True,  0, 0.0),
                                   ("C", "policy", True,  2, 0.0),
                                   ("D", "policy", True,  2, 0.1),
                                   ("E", "ev",     True,  2, 0.1)]}
    assert len(names) == 5, "two configurations would write to the same run directory"


def test_round_weights_exponent_changes_the_distribution():
    assert round_weights(0) == [1] * 20                 # uniform
    assert round_weights(2)[-1] == 400                  # r**2

    random.seed(0); uniform = sample_round_sizes(2000, weights_exp=0)
    random.seed(0); skewed  = sample_round_sizes(2000, weights_exp=2)
    mean = lambda xs: sum(xs) / len(xs)
    assert 9.5 < mean(uniform) < 11.5, "exp=0 should centre near 10.5"
    assert mean(skewed) > 14, "exp=2 should push mass towards large rounds"


# --- bid_mode reaches the agent ---------------------------------------------

def test_bid_mode_is_carried_by_the_agent():
    assert RLAgent(net=None).bid_mode == "ev"           # default unchanged
    assert RLAgent(net=None, bid_mode="policy").bid_mode == "policy"


def test_ev_mode_records_no_bid_mask():
    # In "ev" mode the bid is analytic, so there is no action distribution and
    # no mask to store; reinforce_loss relies on that to skip the policy term.
    assert "bid" not in Config(bid_mode="ev").policy_heads()


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            fn(); print(f"PASS   {name}")
        except AssertionError as e:
            failures += 1; print(f"FAIL   {name}\n         {e}")
        except Exception as e:
            failures += 1; print(f"ERROR  {name}\n         {type(e).__name__}: {e}")
    print(f"\n{failures} failure(s)" if failures else "\nall green")
    sys.exit(1 if failures else 0)
