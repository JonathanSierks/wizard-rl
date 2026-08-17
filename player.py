import random
from cards import Card, COLORS, TRUMP
import numpy as np
from observations import BidObservation, PlayObservation
import torch
import argparse
from termcolor import colored

import random
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


class RLAgent:
    def __init__(self, net, greedy=False, debug=False):
        self.net = net
        self.greedy = greedy
        self.debug = debug
        self.bid_logits_log = []
        self.pending = []
        self.buffer = []
        self.gamma = 1.0        # we choose NOT to to discount reward for actions further back in time

    def choose_bid(self, observation, valid_bids):
        enc = encode(observation)
        x = torch.from_numpy(enc)

        with torch.no_grad():
            raw_logits, _, _ = self.net(x)          # [21], noch ohne -inf

        if self.debug:
            self.bid_logits_log.append((raw_logits.clone(), observation.round_nr))

        mask = torch.zeros(raw_logits.shape[0], dtype=torch.bool)
        mask[valid_bids] = True
        bid_logits = raw_logits.masked_fill(~mask, float('-inf'))

        dist = torch.distributions.Categorical(logits=bid_logits)

        # print bidding model states; only activate for jupyter debugging
        # print("logits:", bid_logits)
        # print("probs :", dist.probs)
        # print("argmax:", dist.probs.argmax().item())
        # print("mask  :", mask)

        if self.greedy:
            idx = int(bid_logits.argmax())
        else:
            idx = int(dist.sample())
            self.pending.append((enc, idx, mask.numpy(), "bid"))
        return idx
    
    
    def choose_card(self, observation, legal_cards: list[Card]):
        enc = encode(observation)
        x = torch.from_numpy(enc)

        with torch.no_grad():                       # Inference — kein Gradient nötig
            _, play_logits, _ = self.net(x)             # net(x), nicht net.forward(x)

        mask = torch.from_numpy(multi_hot(legal_cards)).bool()
        play_logits = play_logits.masked_fill(~mask, float('-inf'))  

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
    def observe_reward(self, reward):
        n = len(self.pending)
        for t, (enc, action, mask, head) in enumerate(self.pending):
            G = (self.gamma ** (n - 1 - t)) * reward     # rückwärts diskontiert für jeweilige action aus pending
            self.buffer.append((enc, action, mask, head, G))        # buffer speichert observation tuple (enc_obs, action, mask, head, G) jeder runde, inkl. discontinued reward G
        self.pending = []

    def drain_buffer(self):
        batch, self.buffer = self.buffer, []
        return batch


class RandomAgent:
    def choose_bid(self, observation, valid_bids: list[int]):
        return random.choice(valid_bids)

    def choose_card(self, observation, legal_cards: list[Card]):
        return random.choice(legal_cards)
    
    def observe_reward(self, reward):
        pass


class HumanAgent:
    def choose_bid(self, obs, valid_bids):
        print("What bid would you like to make?")
        while True:
            try:
                bid = int(input(f"Gebot: "))
                if bid in valid_bids: return bid
            except ValueError: pass


    def choose_card(self, obs, legal_cards):
        hand_str = ", ".join(colored(str(c.value), c.color) for c in obs.hand)
        print(f"Your hand is: [{hand_str}]")
        print(f"Trumpf: {obs.trump}")
        print("What card index would you like to play?")
        x = int(input())
    
        while obs.hand[x] not in legal_cards:
            try:
                print("Invalid card. Try again")
                x = int(input())
            except ValueError: pass
        return obs.hand[x]


    def observe_reward(self, reward):
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
        self.agent.observe_reward(reward)



