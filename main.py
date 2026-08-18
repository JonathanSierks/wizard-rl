from model import WizNet
from player import Player, RLAgent, HumanAgent
from game import Game
import torch


def start_game():

    obs_dim = 60 + 60 + 180 + 5 + 9 + (1 + 1 + 1)       # dim = 317
    max_bid = 20
    hidden_dim = 256

    net_1000 = WizNet(obs_dim, max_bid, hidden_dim)
    #net_1000.load_state_dict(torch.load("relevant_checkpoints/up_20260816_160023_750.pt"))
    net_1000.load_state_dict(torch.load("relevant_checkpoints/up_20260815_161453_6000.pt"), strict=False)

    player1 = Player("human1", 0, agent=HumanAgent())
    player2 = Player("rl1", 1, agent=RLAgent(net_1000, greedy=True))
    player3 = Player("rl2", 2, agent=RLAgent(net_1000, greedy=True))

    game = Game()

    game.add_player(player1)
    game.add_player(player2)
    game.add_player(player3)

    game.start_verbose()
    

if __name__== "__main__":
    start_game()
