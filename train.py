from model import WizNet
from player import RLAgent, RandomAgent, Player
from game import Game
import torch
import numpy as np

def evaluate(net, n_games=200):
    net.eval()
    totals = []

    hits = []
    with torch.no_grad():
        for _ in range(n_games):
            players = [Player("rl", 0, RLAgent(net, greedy=True)),
                    Player("r1", 1, RandomAgent()),
                    Player("r2", 2, RandomAgent())]
            game = Game()
            for p in players:
                game.add_player(p)
            game.start()
            totals.append(players[0].points)

            rl = players[0]
            hits.append(rl.bid_hits / rl.rounds_played)     # Trefferquote über ALLE 20 Runden
    net.train()
    
    return sum(totals) / len(totals), hits

# buffer = (enc, action, mask, head, G)             ; bids und alle aktionen mit G; einzelner buffer JE SPIELER; wächst in länge über Game
# pending = (enc, idx, mask.numpy(), "bid"/"play")  ; aktionen einen runde (je SPIELER) ohne G; wächst in länge über Runde
# batch = concatinierter buffer aller spieler; bei 3 spielern wäre das len(batch) = 690; eig. zu wenig für einen step; wir wollen lieber 10 - 20 games sammeln
def reinforce_loss(net, batch):
    losses = []
    for head_name, head_idx in (("bid", 0), ("play", 1)):
        group = [b for b in batch if b[3] == head_name]
        if not group:
            continue
        G = torch.tensor([b[4] for b in group], dtype=torch.float32)
        adv = (G - G.mean()) / (G.std() + 1e-8)          # Baseline JE Kopf

        enc = torch.from_numpy(np.stack([b[0] for b in group]))
        act = torch.tensor([b[1] for b in group])
        msk = torch.from_numpy(np.stack([b[2] for b in group])).bool()

        logits = net(enc)[head_idx].masked_fill(~msk, float('-inf'))
        dist   = torch.distributions.Categorical(logits=logits)
        losses.append(-(dist.log_prob(act) * adv))

    return torch.cat(losses).mean()

obs_dim = 60 + 60 + 180 + 5 + 9 + (1 + 1 + 1)       # dim = 317
max_bid = 20
hidden_dim = 256

net = WizNet(obs_dim, max_bid, hidden_dim)             # ONE architecture; different agents querry that architecture; game information is agent specific
#agents = [RLAgent(net) for _ in range(3)]   
#players = [Player(f"p{i}", i, agents[i]) for i in range(3)]   # SELF-PLAY: derselbe Agent
#player1, player2, player3 = players
#game = Game()
#game.add_player(player1)
#game.add_player(player2)
#game.add_player(player3)

opt = torch.optim.Adam(net.parameters(), lr=3e-4)

for update in range(50_000):

    # create a batch of 20 games to reduce impact of noise; 1 game = 690 Transitions, 20 games = 14k transitions
    batch = []
    for _ in range(20):
        agents = [RLAgent(net) for _ in range(3)]
        game = Game()
        for i, ag in enumerate(agents):
            game.add_player(Player(f"p{i}", i, ag))
        game.start()

        for ag in agents:
            batch.extend(ag.drain_buffer())
    
    loss = reinforce_loss(net, batch)
    opt.zero_grad(); loss.backward(); opt.step()

    if update % 50 == 0:        # every 50 * 20 games = 1000 games
        points, hits = evaluate(net)
        print(f"up {update}: {points:.1f} Points (RL vs. 2× Random) & occurences of bid=won_tricks:{sum(hits)/len(hits)}")
    if update % 100 == 0:
        torch.save(net.state_dict(), f"checkpoints/up{update}.pt")