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

class Strategy:
    pass