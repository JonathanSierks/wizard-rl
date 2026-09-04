import random
import numpy as np
import torch
from termcolor import colored

from cards import Card, COLORS, TRUMP
from observations import BidObservation, PlayObservation
from heuristic_agent import card_strength, hand_strength, _current_best, _beats

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def card_index(card):
    return COLORS.index(card.color) * 15 + card.value

def multi_hot(cards):
    vec = np.zeros(60, dtype=np.float32)
    for card in cards:
        vec[card_index(card)] = 1.0
    return vec

def one_hot_trump(trump):
    trump_enc = np.zeros(5, dtype=np.float32)         # ["red", "yellow", "blue", "green", "none"]
    trump_enc[TRUMP.index(trump)] = 1.0
    return trump_enc

def encode_trick(plays, n_players=3):
    vec = np.zeros(60 * n_players, dtype=np.float32)   # ← feste Länge, immer
    for rel_pos, card in plays:
        vec[rel_pos * 60 + card_index(card)] = 1.0     # Block pro relativer Position
    return vec

def encode_bids(bids_and_wins, n_players=3):
    vec = np.zeros(3 * n_players, dtype=np.float32)
    for i, (bid, won) in enumerate(bids_and_wins):
        vec[i*3 + 1] = won
        if bid is not None:
            vec[i*3]     = bid
            vec[i*3 + 2] = 1.0        # Flag: hat geboten
    return vec
    
def encode(obs) -> np.ndarray:
    is_bid = isinstance(obs, BidObservation)        # flag to determine if its an BID or PLAY observation
    hand    = multi_hot(obs.hand)                # 60
    played  = multi_hot([] if is_bid else obs.played_cards)        # 60
    trump   = one_hot_trump(obs.trump)           # 5
    trick   = encode_trick([] if is_bid else obs.trick_so_far)     # 60 * n
    bids    = encode_bids(obs.bids_and_wins)     # 2 * n
    meta    = np.array([obs.round_nr, obs.n_players_bid_before_me if is_bid else len(obs.trick_so_far), is_bid], dtype=np.float32)
    return np.concatenate([hand, played, trump, trick, bids, meta])

def ev_bid(q, valid_bids):
    """q: [r+1] Wahrscheinlichkeiten ueber gewonnene Stiche (summiert zu 1).
       Gibt das Gebot mit dem hoechsten Erwartungsnutzen zurueck."""
    k = torch.arange(len(q), dtype=torch.float32)
    best, best_ev = valid_bids[0], -1e9
    for b in valid_bids:
        hit  = q[b] * (20 + 10 * b)
        miss = -10 * (q * (k - b).abs()).sum()      # bei k==b ist |k-b|=0, stoert nicht
        ev = float(hit + miss)
        if ev > best_ev:
            best, best_ev = b, ev
    return best

class RLAgent:
    def __init__(self, net, greedy=False, debug=False):
        self.net = net
        self.greedy = greedy
        self.debug = debug
        self.bid_logits_log = []
        self.bid_compare_log = []
        self.pending = []
        self.buffer = []
        self.gamma = 1.0        # we choose NOT to to discount reward for actions further back in time


    def choose_bid(self, observation, valid_bids):
        enc = encode(observation)
        x = torch.from_numpy(enc)
        r = observation.round_nr

        with torch.no_grad():
            raw_bid_logits, _, _, tricks_logits = self.net(x)          # [21], noch ohne -inf

        q = torch.softmax(tricks_logits[:r+1], 0)
        idx = ev_bid(q, valid_bids)

        if self.debug:
            mask = torch.zeros(21, dtype=torch.bool)
            mask[valid_bids] = True
            head_choice = int(raw_bid_logits.masked_fill(~mask, float('-inf')).argmax())
            self.bid_compare_log.append((r, head_choice, idx, int(q.argmax())))

        if not self.greedy:
            self.pending.append((enc, idx, None, "bid"))
        return idx
    
    
    def choose_card(self, observation, legal_cards: list[Card]):
        enc = encode(observation)
        x = torch.from_numpy(enc)

        with torch.no_grad():                       # Inference — kein Gradient nötig
            _, raw_play_logits, _, _ = self.net(x)             # net(x), nicht net.forward(x)

        mask = torch.from_numpy(multi_hot(legal_cards)).bool()
        play_logits = raw_play_logits.masked_fill(~mask, float('-inf'))  

        dist = torch.distributions.Categorical(logits=play_logits)

        # print play model states; only activate for jupyter debugging    
        # print("logits:", play_logits)
        # print("probs :", dist.probs)
        # print("argmax:", dist.probs.argmax().item())
        # print("mask  :", mask)

        if self.greedy:
            idx = int(play_logits.argmax())      # deterministisch, bester Zug
        else:
            idx = int(dist.sample())             # sampeln, Exploration

        if not self.greedy:
            self.pending.append((enc, idx, mask.numpy(), "play"))
        return Card(color=COLORS[idx // 15], value=idx % 15) 

    # calculate reward back over all actions of 1 round to obtain G's
    def observe_reward(self, reward, won_tricks=0):
        n = len(self.pending)
        for t, (enc, action, mask, head) in enumerate(self.pending):
            G = (self.gamma ** (n - 1 - t)) * reward     # rückwärts diskontiert für jeweilige action aus pending
            self.buffer.append((enc, action, mask, head, G, won_tricks, int(enc[314])))        # buffer speichert observation tuple (enc_obs, action, mask, head, G) jeder runde, inkl. discontinued reward G
        self.pending = []

    def drain_buffer(self):
        batch, self.buffer = self.buffer, []
        return batch


class RandomAgent:
    def choose_bid(self, observation, valid_bids: list[int]):
        return random.choice(valid_bids)

    def choose_card(self, observation, legal_cards: list[Card]):
        return random.choice(legal_cards)
    
    def observe_reward(self, reward, won_tricks=0):
        pass

class HeuristicAgent:
    """Deterministic reference opponent. Same interface as RLAgent."""

    def __init__(self):
        self.buffer = []          # never filled; kept for interface compatibility

    # -- bidding ------------------------------------------------------------
    def choose_bid(self, observation, valid_bids):
        est = hand_strength(observation.hand, observation.trump)
        target = int(round(est))
        # nearest legal bid, preferring the lower one on ties
        return min(valid_bids, key=lambda b: (abs(b - target), b))

    # -- playing ------------------------------------------------------------
    def choose_card(self, observation, legal_cards):
        trump = observation.trump
        trick = list(observation.trick_so_far)
        best, lead = _current_best(trick, trump)

        called, won = observation.bids_and_wins[0]
        need = (called or 0) - won

        def weakest_first(c):                     # fixed tie-break: keeps ties deterministic
            return card_strength(c, trump), c.value

        if best is None:
            # leading: no card to beat yet, so "cheapest winner" is meaningless.
            # Lead the strongest card when tricks are still needed, the weakest
            # one otherwise.
            return (max if need > 0 else min)(legal_cards, key=weakest_first)

        winners = [c for c in legal_cards if _beats(c, best, trump, lead)]

        if need > 0 and winners:
            # win as cheaply as possible
            return min(winners, key=weakest_first)

        if need <= 0:
            losers = [c for c in legal_cards if c not in winners]
            pool = losers if losers else legal_cards
            return min(pool, key=weakest_first)

        # need > 0 but cannot win: discard the weakest card
        return min(legal_cards, key=weakest_first)

    # -- interface ----------------------------------------------------------
    def observe_reward(self, reward, won_tricks=0):
        pass

    def drain_buffer(self):
        return []


class HumanAgent:
    def choose_bid(self, obs, valid_bids):
        print("What bid would you like to make?")
        while True:
            try:
                bid = int(input("Gebot: "))
            except ValueError:
                print("Bitte eine Zahl.")
                continue
            if bid in valid_bids:
                return bid
            print(f"Ungültig. Erlaubt: {valid_bids}")

    def choose_card(self, obs, legal_cards):
        hand_str = ", ".join(colored(str(c.value), c.color) for c in obs.hand)
        print(f"Your hand is: [{hand_str}]")
        print(f"Trumpf: {obs.trump}")
        print("What card index would you like to play?")
        while True:
            try:
                i = int(input())
            except ValueError:
                print("Bitte eine Zahl.")
                continue
            if not 0 <= i < len(obs.hand):
                print(f"Index außerhalb der Hand (0..{len(obs.hand)-1}).")
                continue
            if obs.hand[i] in legal_cards:
                return obs.hand[i]
            print("Karte nicht spielbar (Farbzwang).")

    def observe_reward(self, reward, won_tricks=0):
        pass


class Player:
    def __init__(self, name, id, agent=None):
        self.agent = agent
        self.name = name
        self.id = id
        self.hand = []
        self.points = 0
        self.called_tricks = 0
        self.won_tricks = 0 
        self.bid_hits = 0
        self.rounds_played = 0
        self.round_log = []         # used in calculate_points() to create a history over called and won tricks per round, to access that info in evaluate and create another metric out of it

    def call_tricks(self, observation, valid_bids):
        bid = self.agent.choose_bid(observation, valid_bids)
        self.called_tricks = bid
        return bid

    # the actual playing of the card
    # decision on what card to be played is made by agent.choose_card()
    def play_card(self, observation, legal_cards):
        card = self.agent.choose_card(observation, legal_cards)
        self.hand.remove(card)
        return card
    
    def observe_reward(self, reward):
        self.agent.observe_reward(reward, self.won_tricks)



