from model import WizNet
from player import Player, RLAgent, HumanAgent, HeuristicAgent
from game import Game
import torch


def start_game():

    obs_dim = 60 + 60 + 180 + 5 + 9 + (1 + 1 + 1)       # dim = 317
    max_bid = 20
    hidden_dim = 256

    pro_net = WizNet(obs_dim, max_bid, hidden_dim)
    #pro_net.load_state_dict(torch.load("relevant_checkpoints/up_20260816_160023_750.pt"))
    #pro_net.load_state_dict(torch.load("relevant_checkpoints/up_20260815_161453_12500.pt"), strict=False)
    #pro_net.load_state_dict(torch.load("relevant_checkpoints/up_20260819_000156_12500.pt"), strict=False)
    pro_net.load_state_dict(torch.load("relevant_checkpoints/F_seed2_20260907_190730_up2750.pt"), strict=False)
    player1 = Player("human1", 0, agent=HumanAgent())
    player2 = Player("rl1", 1, agent=RLAgent(pro_net, greedy=True))
    player3 = Player("rl2", 2, agent=RLAgent(pro_net, greedy=True))
    #player2 = Player("heuristic1", 1, agent=HeuristicAgent())   # explainable opponent
    #player3 = Player("heuristic2", 2, agent=HeuristicAgent())   # explainable opponent

    game = Game()

    game.add_player(player1)
    game.add_player(player2)
    game.add_player(player3)

    game.start_verbose()
    

if __name__== "__main__":
    start_game()
