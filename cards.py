from random import shuffle, choice
from dataclasses import dataclass

NUM_CARDS = 60
COLORS = ["green", "red", "blue", "yellow"]
VALUES = range(0,15)

WIZARD = 14
NARR = 0

@dataclass
class Card:
    color: str
    value: int

class CardDeck:
    
    def __init__(self):
        self.cards = []
        self.initialize_deck()
    
    def initialize_deck(self):
        for color in COLORS:
            for value in VALUES:        
                self.cards.append(Card(color, value))
        shuffle(self.cards)

    def draw_card(self):
        return self.cards.pop()