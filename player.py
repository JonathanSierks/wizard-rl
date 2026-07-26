import random
from cards import Card, COLORS, TRUMP
import numpy as np
from observations import BidObservation, PlayObservation
import torch

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
    def __init__(self, net, greedy=False):
        self.net = net
        self.greedy = greedy
        self.pending = []
        self.buffer = []
        self.gamma = 0.95

    def choose_bid(self, observation, valid_bids):
        enc = encode(observation)
        x = torch.from_numpy(enc)

        with torch.no_grad():                       # Inference — kein Gradient nötig
            bid_logits, _ = self.net(x)             # net(x), nicht net.forward(x)

        mask = torch.zeros(bid_logits.shape[0], dtype=torch.bool)
        mask[valid_bids] = True                     # valid_bids sind direkt die Indizes
        bid_logits = bid_logits.masked_fill(~mask, float('-inf'))

        dist = torch.distributions.Categorical(logits=bid_logits)
        if self.greedy:
            idx = int(bid_logits.argmax())
        else:
            idx = int(dist.sample())

        if not self.greedy:
            self.pending.append((enc, idx, mask.numpy(), "bid"))
        return idx        # beim Bid ist der Index direkt das Gebot
    
    def choose_card(self, observation, legal_cards: list[Card]):
        enc = encode(observation)
        x = torch.from_numpy(enc)

        with torch.no_grad():                       # Inference — kein Gradient nötig
            _, play_logits = self.net(x)             # net(x), nicht net.forward(x)

        mask = torch.from_numpy(multi_hot(legal_cards)).bool()
        play_logits = play_logits.masked_fill(~mask, float('-inf'))

        dist = torch.distributions.Categorical(logits=play_logits)
        if self.greedy:
            idx = int(play_logits.argmax())      # deterministisch, bester Zug
        else:
            idx = int(dist.sample())             # sampeln, Exploration

        if not self.greedy:
            self.pending.append((enc, idx, mask.numpy(), "play"))
        return Card(color=COLORS[idx // 15], value=idx % 15) 

    def observe_reward(self, reward):
        n = len(self.pending)
        for t, (enc, action, mask, head) in enumerate(self.pending):
            G = (self.gamma ** (n - 1 - t)) * reward     # rückwärts diskontiert über die round
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


class Player:
    def __init__(self, name, id, agent=None):
        self.agent = agent
        self.name = name
        self.id = id
        self.hand = []
        self.points = 0
        self.called_tricks = 0
        self.won_tricks = 0  

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
