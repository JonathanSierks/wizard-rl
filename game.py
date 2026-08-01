from dataclasses import dataclass
from cards import CardDeck, Card
from tricks import Trick, legal_cards, resolve_winner
import random
from observations import BidObservation, PlayObservation
from termcolor import colored

NUMBER_OF_ROUNDS = 20

def determine_trump(deck, round_nr):
    if round_nr == NUMBER_OF_ROUNDS:
        return "none"
    
    potential_trump_card = random.choice(deck.cards)
    if potential_trump_card.value == 14:
        return random.choice(["red", "yellow", "blue", "green"])    # stub; später kommt hier ein agent aufruf rein, der dann entscheidet, was er als trumpf haben will
    elif potential_trump_card.value == 0:
        return "none"       # ob das so funktioniert weiß ich nicht; sonderfall, auch für den agenten; vielleicht lieber erstmal übergehen?
    else:
        return potential_trump_card.color

def calculate_points(players):
    round_points = {}
    for player in players:
        pts = (20 + player.called_tricks * 10
               if player.called_tricks == player.won_tricks
               else abs(player.called_tricks - player.won_tricks) * -10)
        player.points += pts
        round_points[player.id] = pts

        # in calculate_points, pro Spieler:
        player.rounds_played += 1
        
        if player.called_tricks == player.won_tricks:
            player.bid_hits += 1

    return round_points

class Game:
    def _rotate_to(self, my_id):                 # Spieler in MEINER Sicht: Index 0 = ich
        n = len(self.players)
        return [self.players[(my_id + k) % n] for k in range(n)]
    
    def __init__(self):
        self.players = []
        self.number_of_rounds = NUMBER_OF_ROUNDS

    def add_player(self, player):
        self.players.append(player)

    def start(self):

        # play 1 game = iterate over ROUNDS
        for round_nr in range(1, self.number_of_rounds + 1):
            
            #print(f"=========== ROUND: {round_nr} ===========")
            # -1) initialize CardDeck
            deck = CardDeck()
            
            # ── ASSERT 1: vor dem Austeilen ── Hände müssen leer sein
            assert all(not p.hand for p in self.players), \
                f"Hand nicht leer zu Beginn von Runde {round_nr}"
            
            # 1) deal cards to players    
            for player in self.players:
                for i in range(round_nr):
                    player.hand.append(deck.draw_card())
                    #print(f"Anzahl Karten in Hand von {player.id}: {len(player.hand)}")
                    #print(f"Hand von Spieler {player.id}: {player.hand}")
            
            # ── ASSERT 2: nach dem Austeilen ── Karten-Erhaltung
                assert sum(len(p.hand) for p in self.players) + len(deck.cards) == 60, \
                    "Karten sind verloren gegangen oder doppelt"

            # 0) create initial trick-object. lead_player will change, trump will stay the same for this round
            trump = determine_trump(deck, round_nr)
            #print(f"Trump: {trump}")

            first_player = (round_nr - 1) % len(self.players)
            #print(first_player)
            
            # 2) bidding
            # Rundenstart, einmal vor dem Bieten:
            for p in self.players:
                p.called_tricks = None       # None = hat noch nicht geboten
                p.won_tricks   = 0           # neue Runde

            n = len(self.players)
            order = [(first_player + i) % n for i in range(n)]   # Gebotsreihenfolge

            for n_bid_before, pid in enumerate(order):
                player = self.players[pid]

                bids_and_wins = [(p.called_tricks, p.won_tricks)   # Index 0 = ich, rotiert
                                for p in self._rotate_to(pid)]

                obs = BidObservation(player.hand, trump, bids_and_wins, n_bid_before, round_nr)
                #print(f"Bid obs: {obs}")

                valid_bids = list(range(round_nr + 1))             # 0..round_nr
                if n_bid_before == n - 1:                          # letzter Bieter: "screw the dealer"
                    schon    = sum(p.called_tricks for p in self.players if p.called_tricks is not None)
                    verboten = round_nr - schon                    # Summe darf nicht = round_nr werden
                    if verboten in valid_bids:
                        valid_bids.remove(verboten)

                player.called_tricks = player.call_tricks(obs, valid_bids)
                #print(f"Spieler {player.id} hat {player.called_tricks} Stiche angesagt.")

        # ── ASSERT 3: nach dem Bieten ── Gebote gültig + screw-the-dealer
            assert all(p.called_tricks is not None for p in self.players), \
                "Nicht jeder hat geboten"
            assert all(0 <= p.called_tricks <= round_nr for p in self.players), \
                "Gebot außerhalb 0..round_nr"
            assert sum(p.called_tricks for p in self.players) != round_nr, \
                "Gebotssumme = round_nr — screw-the-dealer wurde verletzt"

            # 3) playing
            # --- auf Rundenebene, VOR der Stich-Schleife ---
            played_cards = []
            current_lead = first_player          # erster Stich: Anspieler = erster Bieter
            # n = len(self.players)  ist aus dem Bid-Block schon gesetzt

            # play 1 round = iterate over TRICKS
            for _ in range(round_nr):
                trick = Trick(trump, current_lead)
                order = [(current_lead + i) % n for i in range(n)]    # Spielreihenfolge dieses Stichs

                # play 1 trick = iterate over PLAYERS/CARDS
                for pid in order:
                    player = self.players[pid]

                    # --- Observation dieses Spielers ---
                    trick_so_far  = [((p_id - pid) % n, card) for p_id, card in trick.plays]
                    bids_and_wins = [(p.called_tricks, p.won_tricks) for p in self._rotate_to(pid)]
                    obs = PlayObservation(
                        hand=list(player.hand),
                        trump=trump,
                        trick_so_far=trick_so_far,
                        played_cards=list(played_cards),
                        bids_and_wins=bids_and_wins,
                        round_nr=round_nr
                    )
                    #print(f"Play obs: {obs}")

                    # --- legale Karten, entscheiden, spielen ---
                    legal = legal_cards(player.hand, trick.plays)
                    card  = player.play_card(obs, legal)     # entscheidet, entfernt aus Hand, gibt Karte
                    #print(f"{player.id} spielt {card}")
                    trick.add((pid, card))
                    played_cards.append(card)
                
                # ── ASSERT 4a: nach jedem Stich ── jeder hat genau eine Karte gelegt
                assert len(trick.plays) == n, "Nicht jeder hat eine Karte gespielt"

                # --- Stich auswerten ---
                winner_id = resolve_winner(trick.plays, trump)   # reine Funktion, gibt pid
                self.players[winner_id].won_tricks += 1
                current_lead = winner_id                          # Gewinner spielt nächsten Stich an

            # ── ASSERT 4b: nach allen Stichen ── Stich-Erhaltung + Hände leer
            assert sum(p.won_tricks for p in self.players) == round_nr, \
                f"Stich-Erhaltung verletzt: {sum(p.won_tricks for p in self.players)} ≠ {round_nr}"
            assert all(not p.hand for p in self.players), \
                "Nach der Runde sind noch Karten übrig"

            # 4) count points, write points
            
            #after 1 full round: calculate points and calculate rewards backwards
            round_points = calculate_points(self.players)
            for p in self.players:
                p.observe_reward(round_points[p.id]) 
            #print(f"Nach Runde {round_nr} ist das Standing:")
            #for p in self.players:
            #    print(f"Spieler {p.id} hat {p.points}")

    def start_verbose(self):
        for round_nr in range(1, self.number_of_rounds + 1):

            print(f"=========== ROUND: {round_nr} ===========")

            deck = CardDeck()

            assert all(not p.hand for p in self.players), \
                f"Hand nicht leer zu Beginn von Runde {round_nr}"

            # 1) deal cards
            for player in self.players:
                for i in range(round_nr):
                    player.hand.append(deck.draw_card())
                assert sum(len(p.hand) for p in self.players) + len(deck.cards) == 60, \
                    "Karten sind verloren gegangen oder doppelt"

            trump = determine_trump(deck, round_nr)

            # --- HAND + TRUMP ausgeben ---
            for player in self.players:
                hand_str = ", ".join(colored(str(c.value), c.color) for c in player.hand)
                print(f"HAND{player.name}: {hand_str}")
            trump_str = "None" if trump == "none" else colored(trump, trump)
            print(f"Trump: {trump_str}")
            print("\n")

            first_player = (round_nr - 1) % len(self.players)

            # 2) bidding
            for p in self.players:
                p.called_tricks = None
                p.won_tricks = 0

            n = len(self.players)
            order = [(first_player + i) % n for i in range(n)]

            print("#### CALLS ####")
            for n_bid_before, pid in enumerate(order):
                player = self.players[pid]

                bids_and_wins = [(p.called_tricks, p.won_tricks)
                                for p in self._rotate_to(pid)]

                obs = BidObservation(player.hand, trump, bids_and_wins, n_bid_before, round_nr)

                valid_bids = list(range(round_nr + 1))
                if n_bid_before == n - 1:
                    schon = sum(p.called_tricks for p in self.players if p.called_tricks is not None)
                    verboten = round_nr - schon
                    if verboten in valid_bids:
                        valid_bids.remove(verboten)

                player.called_tricks = player.call_tricks(obs, valid_bids)
                print(f"Spieler {player.name} CALLED = {player.called_tricks}")
            print("\n")

            assert all(p.called_tricks is not None for p in self.players), \
                "Nicht jeder hat geboten"
            assert all(0 <= p.called_tricks <= round_nr for p in self.players), \
                "Gebot außerhalb 0..round_nr"
            assert sum(p.called_tricks for p in self.players) != round_nr, \
                "Gebotssumme = round_nr — screw-the-dealer wurde verletzt"

            # 3) playing
            print(f"#### Plays of ROUND {round_nr} ####")
            played_cards = []
            current_lead = first_player

            for trick_idx in range(round_nr):
                print(f"## TRICK {trick_idx} of ROUND {round_nr} ##")
                trick = Trick(trump, current_lead)
                order = [(current_lead + i) % n for i in range(n)]

                for pid in order:
                    player = self.players[pid]

                    trick_so_far = [((p_id - pid) % n, card) for p_id, card in trick.plays]
                    bids_and_wins = [(p.called_tricks, p.won_tricks) for p in self._rotate_to(pid)]
                    obs = PlayObservation(
                        hand=list(player.hand),
                        trump=trump,
                        trick_so_far=trick_so_far,
                        played_cards=list(played_cards),
                        bids_and_wins=bids_and_wins,
                        round_nr=round_nr
                    )

                    legal = legal_cards(player.hand, trick.plays)
                    card = player.play_card(obs, legal)
                    print(f"{player.name} spielt {colored(str(card.value), card.color)}")
                    trick.add((pid, card))
                    played_cards.append(card)

                assert len(trick.plays) == n, "Nicht jeder hat eine Karte gespielt"

                winner_id = resolve_winner(trick.plays, trump)
                self.players[winner_id].won_tricks += 1
                current_lead = winner_id
                print(f"Spieler {self.players[winner_id].name} hat den Stich gewonnen!")
                print("\n")

            assert sum(p.won_tricks for p in self.players) == round_nr, \
                f"Stich-Erhaltung verletzt: {sum(p.won_tricks for p in self.players)} ≠ {round_nr}"
            assert all(not p.hand for p in self.players), \
                "Nach der Runde sind noch Karten übrig"

            # 4) points
            round_points = calculate_points(self.players)
            for p in self.players:
                p.observe_reward(round_points[p.id])

            print(f"#### STATS (for ROUND {round_nr}) ####")
            for p in self.players:
                print(f"Spieler {p.name}: CALLED = {p.called_tricks}; "
                    f"WON = {p.won_tricks}; POINTS = {p.points}")
            print("\n")            
        
        # 5) determine GAME WINNER