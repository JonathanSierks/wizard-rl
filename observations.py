from dataclasses import dataclass
from cards import Card

@dataclass(frozen=True)
class BidObservation:
    hand: list[Card]
    trump: str | None
    bids_and_wins: list[tuple[int, int]]   # rotiert, Index 0 = ich; (gebot, stiche)
    n_players_bid_before_me: int           # wie viele vor mir schon geboten haben
    round_nr: int

@dataclass(frozen=True)
class PlayObservation:
    hand: list[Card]
    trump: str | None
    trick_so_far: list[tuple[int, Card]]   # geordnet; Spieler relativ zu mir kodiert
    played_cards: list[Card]               # Historie der ganzen Runde (Markov)
    bids_and_wins: list[tuple[int, int]]   # rotiert, Index 0 = ich
    round_nr: int