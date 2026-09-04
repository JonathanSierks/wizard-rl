"""
Deterministic rule-based Wizard agent.

Purpose: a fixed-strength reference opponent for evaluation. Unlike a frozen
learned checkpoint, its behaviour is transparent, does not drift between
experiments, and can be described in one page — so a score measured against it
is interpretable on its own terms.

Bidding
-------
Each card is assigned a win probability from a static table, and the bid is the
rounded sum. The tables were calibrated by simulation so that the mean bid over
random hands matches the structural expectation r / n_players to within ~1% for
every round size (see `calibration_check()` at the bottom of this file).

Playing
-------
`need` is the number of tricks still missing (called minus won so far).

When leading there is no card to beat yet, so the agent leads its strongest
card if need > 0 and its weakest one otherwise. When following:
  need > 0   -> try to win the trick as cheaply as possible; if no legal card
                can win, discard the weakest one
  need == 0  -> duck; discard the lowest card that cannot win
  need < 0   -> overshot the bid; duck as well

The agent is fully deterministic: identical observations produce identical
actions, with ties broken by a fixed card ordering.
"""

from cards import Card, COLORS, WIZARD, NARR
from tricks import card_rank

# ---------------------------------------------------------------------------
# Bidding tables. Values are P(this card takes a trick), calibrated by
# simulation (K = 1.55 scaling on hand-set relative strengths, capped at 0.95).
# ---------------------------------------------------------------------------

TRUMP_W = {1: .124, 2: .186, 3: .279, 4: .372, 5: .465, 6: .589, 7: .713,
           8: .837, 9: .950, 10: .950, 11: .950, 12: .950, 13: .950}

PLAIN_W = {1: .000, 2: .016, 3: .016, 4: .031, 5: .047, 6: .062, 7: .093,
           8: .155, 9: .233, 10: .341, 11: .481, 12: .651, 13: .853}


def card_strength(card, trump):
    """Estimated probability that `card` takes a trick, given the trump suit."""
    if card.value == WIZARD:
        return 1.0
    if card.value == NARR:
        return 0.0
    if trump is not None and trump != "none" and card.color == trump:
        return TRUMP_W[card.value]
    return PLAIN_W[card.value]


def hand_strength(hand, trump):
    """Expected number of tricks this hand takes."""
    return sum(card_strength(c, trump) for c in hand)


# ---------------------------------------------------------------------------
def _beats(card, best, trump, lead_color):
    """True if `card` takes the trick away from `best`. `best` may be None.

    Ranking is delegated to tricks.card_rank so the agent cannot drift away
    from the rules the game actually resolves tricks with.
    """
    if best is None:                              # nothing on the table yet
        return True
    # Only wizards and fools have been played so far, so my card is the one
    # that sets the lead colour (card_rank ignores it for wizards/fools).
    lead = lead_color if lead_color is not None else card.color
    return card_rank(card, trump, lead) > card_rank(best, trump, lead)


def _current_best(trick_so_far, trump):
    """(card currently winning the trick, lead colour) for a partial trick.

    `trick_so_far` has the [(rel_pos, Card)] shape of PlayObservation.
    """
    cards = [c for _, c in trick_so_far]
    lead = next((c.color for c in cards if c.value not in (WIZARD, NARR)), None)
    best, best_rank = None, -1
    for c in cards:
        r = card_rank(c, trump, lead)
        if r > best_rank:                         # strictly greater: first card wins ties
            best, best_rank = c, r
    return best, lead


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def calibration_check(n_hands=4000, seed=0):
    """Mean heuristic bid vs. the structural expectation r / 3, per round size."""
    import random
    rng = random.Random(seed)
    deck = [Card(color=c, value=v) for c in COLORS for v in range(15)]
    print(f"{'r':>3} {'r/3':>7} {'mean bid':>10} {'ratio':>7}")
    for r in (1, 3, 5, 8, 10, 14, 17, 20):
        tot = 0.0
        for _ in range(n_hands):
            hand = rng.sample(deck, r)
            trump = rng.choice(COLORS + ["none"])
            tot += hand_strength(hand, trump)
        m = tot / n_hands
        print(f"{r:3d} {r/3:7.2f} {m:10.2f} {m/(r/3):7.3f}")


if __name__ == "__main__":
    calibration_check()
