from cards import CardDeck
from tricks import Trick
import random
import math

NUMBER_OF_ROUNDS = 3

def determine_trump(deck):
    potential_trump_card = random.choice(deck.cards)

    if potential_trump_card.value == 14:
        return random.choice(["red", "yellow", "blue", "green"])    # stub; später kommt hier ein agent aufruf rein, der dann entscheidet, was er als trumpf haben will
    elif potential_trump_card.value == 0:
        return "no trump"       # ob das so funktioniert weiß ich nicht; sonderfall, auch für den agenten; vielleicht lieber erstmal übergehen?
    else:
        return potential_trump_card.color

def calculate_points(players):
    for player in players:
        wanted = player.called_tricks
        achieved = player.won_tricks

        # was ist wenn der agent so schlecht ist, dass er negativ punkte erreicht?
        if wanted == achieved:
            player.points += 20 + wanted*10
        else:
            player.points += abs(wanted - achieved) * -10

class Game:
    
    def __init__(self):
        self.players = []
        self.round = 1
        self.number_of_rounds = NUMBER_OF_ROUNDS

    def add_player(self, player):
        self.players.append(player)

    def start(self):
        lead_player = 3
        for round_nr in range(1, self.number_of_rounds + 1):
            #print(f"Round nr: {round_nr}")
            # -1) initialize CardDeck
            deck = CardDeck()

            # 1) deal cards to players
            for player in self.players:
                for i in range(round_nr):
                    player.hand.append(deck.draw_card())
            
            # 0) create initial trick-object. lead_player will change, trump will stay the same for this round
            trump = determine_trump(deck)
            #print(f"Trump is: {trump}")

            lead_player = lead_player % len(self.players)
            lead_player += 1      
            

            # 2) call tricks
            for player in self.players:
                player.called_tricks = 1

            # 3) play cards and write won tricks to players
            # repeat as many times as tricks exist
            for _ in range(round_nr):
                trick = Trick(trump, lead_player)
                for player in self.players:
                    trick.add((player, player.play_card()))
                #print(trick.winner().id)
                trick.lead_player = trick.winner().id
                self.players[trick.winner().id].won_tricks += 1

            # 4) count points, write points
            calculate_points(self.players)
    
    # 5) determine GAME WINNER
        


