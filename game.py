from cards import CardDeck
from tricks import Trick

NUM_PLAYERS = 3

class Game:
    
    def __init__(self):
        self.players = []
        self.round = 1
        self.number_of_rounds = 3

    def add_player(self, player):
        self.players.append(player)

    def start(self):
        

        for round_nr in range(1, self.number_of_rounds + 1):
            # -1) initialize CardDeck
            deck = CardDeck()

            # 1) deal cards to players
            for player in self.players:
                for i in range(round_nr):
                    player.hand.append(deck.draw_card())
            
            # 0) create initial trick-object. lead_player will change, trump will stay the same for this round
            trump = "green"
            lead_player = 1
            trick = Trick(trump, lead_player)

            # 2) call tricks
            for player in self.players:
                player.called_tricks = 1

            # 3) play cards and write won tricks to players
            # repeat as many times as tricks exist
            for t in range(round_nr):
                for player in self.players:
                    trick.add((player, player.play_card()))

                trick.lead_player = trick.winner()
                self.players[trick.winner()].won_tricks += 1

            # 4) count points, write points
            for player in self.players:
                wanted = player.called_tricks
                achieved = player.won_tricks

                # vereinfacht für jetzt
                if wanted == achieved:
                    player.points += 5

                elif achieved < wanted:
                    player.points -= 10
                elif achieved > wanted:
                    player.points -= 20
    
    # 5) determine GAME WINNER
        


