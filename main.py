from game import Game
from player import Player, RandomAgent


def start_game():
    game = Game()
    
    agent = RandomAgent()



    p = Player("justus", 0, agent)
    game.add_player(p)
    p = Player("peter", 1, agent)
    game.add_player(p)
    p = Player("bob", 2, agent)
    game.add_player(p)

    game.start()
    

if __name__== "__main__":
    start_game()
