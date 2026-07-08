'''
Player sollte die legalen Aktionen berechnen (Farbzwang etc.) und Agent nur aus dieser Menge wählen lassen 
— nicht der Agent selbst muss Regelwissen haben. 
Macht Regeln testbar (wie jetzt schon bei resolve_winner) und hält Agent puristisch: Observation rein, Action raus.
'''

class Player:
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.hand = []
        self.points = 0
        self.called_tricks = 0
        self.won_tricks = 0     

    # only card PLAYING, decision on what card to be played made in agent.py
    def play_card(self):
        card_to_play = self.hand.pop()      # this is what we change to AGENT later on
        return card_to_play
    


class Agent:
    def choose_bid(self, observation, valid_bids):
        pass

    def choose_card(self, observation, legal_cards):
        pass