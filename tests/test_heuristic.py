# tests/test_heuristic.py
"""
Tests for HeuristicAgent.

Layered from cheap to sharp:
  1. Smoke        -- does a full game run without breaking an invariant
                     asserted in game.py?
  2. Determinism  -- same observation, same card?
  3. Consistency  -- does the agent's notion of who wins a trick agree with
                     the ground truth in tricks.py?
  4. Regression   -- concrete situations that were once decided wrongly.
  5. Contract     -- does the agent keep the promises of the agent interface?

Run with:
    pytest tests/test_heuristic.py -v      # if pytest is installed
    python tests/test_heuristic.py         # without pytest, same effect
"""

# Only needed for the direct call: pytest locates the project root via
# conftest.py, "python tests/test_heuristic.py" does not.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

from cards import Card, COLORS, WIZARD, NARR
from game import Game
from heuristic_agent import _beats, _current_best
from observations import BidObservation, PlayObservation
from player import HeuristicAgent, Player
from tricks import resolve_winner


DECK = [Card(color, value) for color in COLORS for value in range(15)]
TRUMPS = COLORS + ["none"]


def _heuristic_game(n_players=3, sampled=False):
    """A full game played by heuristic agents only."""
    game = Game()
    for i in range(n_players):
        game.add_player(Player(f"h{i}", i, HeuristicAgent()))
    if sampled:
        game.start_sample()
    else:
        game.start()
    return game


def _play_obs(hand, trump="red", trick_so_far=(), called=2, won=0, round_nr=3):
    """A PlayObservation from my seat; index 0 of bids_and_wins is me."""
    return PlayObservation(
        hand=list(hand),
        trump=trump,
        trick_so_far=list(trick_so_far),
        played_cards=[],
        bids_and_wins=[(called, won), (1, 0), (1, 0)],
        round_nr=round_nr,
    )


# --- 1. Smoke ---------------------------------------------------------------
# No expected value. The claim is only "it runs through" -- what makes it
# sharp are the asserts inside game.py that run along with every game.

def test_full_game_runs_without_breaking_invariants():
    random.seed(1234)                    # reproducible if it ever blows up
    for _ in range(100):
        _heuristic_game()


def test_sampled_round_sizes_run_through():
    # start_sample() draws round sizes with repetition, covering cases
    # start() never sees (e.g. round 20 three times in a row).
    random.seed(1234)
    for _ in range(50):
        _heuristic_game(sampled=True)


def test_mixed_table_runs_through():
    # The agent must also work next to other agent types -- that is how it
    # will actually be used in evaluate() and main.py.
    from player import RandomAgent

    random.seed(1234)
    for _ in range(50):
        game = Game()
        game.add_player(Player("h0", 0, HeuristicAgent()))
        game.add_player(Player("r1", 1, RandomAgent()))
        game.add_player(Player("h2", 2, HeuristicAgent()))
        game.start()


# --- 2. Determinism ---------------------------------------------------------
# This is the core property that makes the agent usable as a reference
# opponent: its score must not drift between runs.

def test_card_choice_is_deterministic():
    agent = HeuristicAgent()
    hand = [Card("red", 14), Card("blue", 3), Card("green", 7)]
    obs = _play_obs(hand)
    first = agent.choose_card(obs, list(hand))
    for _ in range(10):
        assert agent.choose_card(obs, list(hand)) == first


def test_bid_is_deterministic():
    agent = HeuristicAgent()
    hand = [Card("red", 14), Card("blue", 3), Card("green", 7)]
    obs = BidObservation(hand, "red", [(None, 0)] * 3, 0, 3)
    first = agent.choose_bid(obs, [0, 1, 2, 3])
    for _ in range(10):
        assert agent.choose_bid(obs, [0, 1, 2, 3]) == first


# --- 3. Consistency with tricks.py ------------------------------------------
# The agent carries its own notion of who is winning a trick. It has to agree
# with resolve_winner(), otherwise it plays by rules the game does not have.

def test_current_best_agrees_with_resolve_winner():
    rng = random.Random(0)
    for _ in range(3000):
        cards = rng.sample(DECK, 3)
        trump = rng.choice(TRUMPS)
        plays = list(enumerate(cards))        # [(pid, Card)] == trick_so_far shape

        winner_pid = resolve_winner(plays, trump)
        best, _lead = _current_best(plays, trump)

        assert best == cards[winner_pid], (
            f"trick={cards} trump={trump}: "
            f"_current_best says {best}, resolve_winner says {cards[winner_pid]}"
        )


def test_current_best_agrees_on_partial_tricks():
    # trick_so_far is incomplete while the trick is being played (0, 1 or 2 cards).
    rng = random.Random(1)
    for n in (1, 2):
        for _ in range(2000):
            cards = rng.sample(DECK, n)
            trump = rng.choice(TRUMPS)
            plays = list(enumerate(cards))
            best, _lead = _current_best(plays, trump)
            assert best == cards[resolve_winner(plays, trump)]


def test_empty_trick_has_no_best_card():
    best, lead = _current_best([], "red")
    assert best is None
    assert lead is None


def test_beats_agrees_with_resolve_winner():
    # Ground truth for "does my card take this trick": play it and ask
    # resolve_winner whether I am the winner.
    rng = random.Random(2)
    for _ in range(4000):
        n = rng.randint(1, 2)                     # cards already on the table
        cards = rng.sample(DECK, n + 1)
        trick, my_card = list(enumerate(cards[:n])), cards[n]
        trump = rng.choice(TRUMPS)

        best, lead = _current_best(trick, trump)
        expected = resolve_winner(trick + [(n, my_card)], trump) == n

        assert _beats(my_card, best, trump, lead) == expected, (
            f"trick={cards[:n]} my_card={my_card} trump={trump}: "
            f"_beats says {_beats(my_card, best, trump, lead)}, truth is {expected}"
        )


# --- 4. Regression ----------------------------------------------------------
# Concrete situations with one right answer. Each of these stands for a bug
# that was in the code at some point.

def test_leading_with_tricks_needed_plays_the_strongest_card():
    # Bug: when leading, best is None, so EVERY card counted as a winner and
    # "win as cheaply as possible" picked the fool.
    agent = HeuristicAgent()
    hand = [Card("red", WIZARD), Card("blue", 3), Card("green", NARR)]
    obs = _play_obs(hand, trump="red", called=2, won=0)      # need = 2
    card = agent.choose_card(obs, list(hand))
    assert card.value != NARR, "never lead the fool when tricks are still needed"
    assert card.value == WIZARD


def test_leading_without_tricks_needed_plays_the_weakest_card():
    agent = HeuristicAgent()
    hand = [Card("red", WIZARD), Card("blue", 3), Card("green", NARR)]
    obs = _play_obs(hand, trump="red", called=0, won=0)      # need = 0
    assert agent.choose_card(obs, list(hand)).value == NARR


def test_bid_already_met_means_ducking():
    # need == 0: the lowest card that does NOT take the trick.
    agent = HeuristicAgent()
    hand = [Card("blue", 13), Card("blue", 2)]
    obs = _play_obs(hand, trump="red", trick_so_far=[(1, Card("blue", 9))],
                    called=1, won=1)                          # need = 0
    assert agent.choose_card(obs, list(hand)) == Card("blue", 2)


def test_overshot_bid_also_ducks():
    # need < 0 does occur: more tricks won than called.
    agent = HeuristicAgent()
    hand = [Card("blue", 13), Card("blue", 2)]
    obs = _play_obs(hand, trump="red", trick_so_far=[(1, Card("blue", 9))],
                    called=1, won=2)                          # need = -1
    assert agent.choose_card(obs, list(hand)) == Card("blue", 2)


def test_wins_as_cheaply_as_possible():
    # need > 0 with several winning cards -> take the weakest of them.
    agent = HeuristicAgent()
    hand = [Card("blue", 13), Card("blue", 10), Card("blue", 2)]
    obs = _play_obs(hand, trump="red", trick_so_far=[(1, Card("blue", 9))],
                    called=2, won=0)                          # need = 2
    assert agent.choose_card(obs, list(hand)) == Card("blue", 10)


# --- 5. Interface contract --------------------------------------------------

def test_bid_is_always_in_valid_bids():
    # If the agent violated this, assert 3 in game.py would fire
    # (screw-the-dealer).
    rng = random.Random(0)
    agent = HeuristicAgent()
    for _ in range(2000):
        r = rng.randint(1, 20)
        hand = rng.sample(DECK, r)
        trump = rng.choice(TRUMPS)

        valid = list(range(r + 1))
        valid.remove(rng.randint(0, r))          # emulate screw-the-dealer

        obs = BidObservation(hand, trump, [(None, 0)] * 3, 2, r)
        assert agent.choose_bid(obs, valid) in valid


def test_played_card_is_always_legal():
    # If it were not, Player.play_card would fail on hand.remove() with a
    # ValueError that is hard to read. Catch it here instead.
    rng = random.Random(0)
    agent = HeuristicAgent()
    for _ in range(2000):
        r = rng.randint(1, 8)
        hand = rng.sample(DECK, r)
        trump = rng.choice(TRUMPS)
        n_played = rng.randint(0, 2)
        trick = [(i + 1, c) for i, c in enumerate(rng.sample(DECK, n_played))]

        obs = _play_obs(hand, trump=trump, trick_so_far=trick,
                        called=rng.randint(0, r), won=rng.randint(0, r),
                        round_nr=r)
        assert agent.choose_card(obs, list(hand)) in hand


def test_agent_satisfies_the_training_interface():
    # train.py calls drain_buffer() on every agent; observe_reward arrives
    # from Player.observe_reward with two arguments.
    agent = HeuristicAgent()
    agent.observe_reward(30, 2)
    assert agent.drain_buffer() == []


# --- direct call without pytest ---------------------------------------------
if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            fn()
            print(f"PASS   {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL   {name}\n         {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR  {name}\n         {type(e).__name__}: {e}")
    print(f"\n{failures} failure(s)" if failures else "\nall green")
    sys.exit(1 if failures else 0)
