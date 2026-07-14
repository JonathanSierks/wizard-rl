'''PLAYER
Player sollte die legalen Aktionen berechnen (Farbzwang etc.) und Agent nur aus dieser Menge wählen lassen 
— nicht der Agent selbst muss Regelwissen haben. 
Macht Regeln testbar (wie jetzt schon bei resolve_winner) und hält Agent puristisch: Observation rein, Action raus.
'''

'''AGENT
hier wichtig: daten-sichtbarkeit für den agenten beachten; 
sauber von global-wissen trennen

Agent.choose_card/choose_bid dürfen nur die Information bekommen, die ein echter Spieler an diesem Punkt hätte 
(eigene Hand, Trumpf, bisher gespielte Karten des Stichs, Gebote/Punktestand aller Spieler, Rundennummer) 
— nicht die Hände der Gegner. Das ist der klassische Fehler bei Kartenspiel-RL: wenn beim Training versehentlich 
Gegnerinformation durchsickert, lernt der Agent etwas, das er im echten Spiel (Inferenz/Self-Play gegen andere) nicht nutzen kann.
'''
import random

from cards import Card

class Agent:
    def choose_bid(self, observation, valid_bids: list[int]):
        pass

    def choose_card(self, observation, legal_cards: list[Card]):
        pass

class RandomAgent:
    def choose_bid(self, observation, valid_bids: list[int]):
        return random.choice(valid_bids)

    def choose_card(self, observation, legal_cards: list[Card]):
        return random.choice(legal_cards)


class Player:
    def __init__(self, name, id, agent):
        self.agent = agent      # connection in main.py muss noch implementiert werden
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


  