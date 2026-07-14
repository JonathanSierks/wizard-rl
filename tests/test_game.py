# tests/test_game.py
from game import Game
from player import Player
from player import RandomAgent          # oder wie dein Zufalls-Agent heißt

def _run_random_game(n_players=3):
    game = Game()
    for i in range(n_players):
        game.add_player(Player(f"p{i}", i, RandomAgent()))
    game.start()                        # läuft; die asserts drin wachen
    return game

def test_zufallsspiel_laeuft_ohne_invariantenbruch():
    for _ in range(300):                # 300 Spiele → viele Zufallssituationen
        _run_random_game()