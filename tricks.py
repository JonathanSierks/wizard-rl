from cards import WIZARD, NARR

def legal_cards(hand, trick_plays):
    if not trick_plays:
        return list(hand)                    # Anspieler: alles erlaubt

    if trick_plays[0][1].value == WIZARD:    # Zauberer ANGESPIELT → keine Bedienpflicht
        return list(hand)

    lead_color = None                        # sonst: erste echte Farbe = Anspielfarbe
    for _, card in trick_plays:
        if card.value not in (WIZARD, NARR):
            lead_color = card.color
            break
    if lead_color is None:                   # nur Narren bisher → alles erlaubt
        return list(hand)

    follow = [c for c in hand if c.color == lead_color]
    if not follow:                           # kann Farbe nicht bedienen → alles erlaubt
        return list(hand)

    return [c for c in hand                  # Bedienpflicht — aber Z/N immer erlaubt
            if c.color == lead_color or c.value in (WIZARD, NARR)]

def card_rank(card, trump, lead_color):
    if card.value == WIZARD:            return 3000
    if card.value == NARR:              return 0        # Narr verliert immer
    if card.color == trump:             return 2000 + card.value
    if card.color == lead_color:        return 1000 + card.value
    return 0   

def resolve_winner(plays, trump):
    lead_color = None
    for _, card in plays:                       # erste echte Farbe = Anspielfarbe
        if card.value not in (WIZARD, NARR):
            lead_color = card.color
            break

    best_player, best_rank = None, -1
    for player, card in plays:
        rank = card_rank(card, trump, lead_color)
        if rank > best_rank:                    # NUR bei echt größer wechseln
            best_rank = rank
            best_player = player
    return best_player  

class Trick:
    def __init__(self, trump, lead_player):
        self.trump = trump
        self.lead_player = lead_player
        self.plays = []

    def add(self, move):
        self.plays.append(move)

    '''
    # stub function
    def winner(self):
        winner_winner_chicken_dinner = 1    # stub für eine funktion, die sich trump, anfangs spieler und plays anschaut und dann entscheidet wer gewonnen hat; gibt player_ID zurück
        
        self.plays = []     # reset plays for the next trick of the round; wrong for later; this is what we need the RL agent to learn
        return winner_winner_chicken_dinner
    '''
    
    def winner(self):
        return resolve_winner(self.plays, self.trump)   # delegiert an die reine Funktion
    