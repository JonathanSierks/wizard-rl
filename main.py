from game import Game
from player import Player

def start_game():
    game = Game()
    
    player_num = 3

    p = Player("peter", 0)
    game.add_player(p)
    p = Player("bob", 1)
    game.add_player(p)
    p = Player("hank", 2)
    game.add_player(p)

    game.start()



if __name__== "__main__":
    start_game()
