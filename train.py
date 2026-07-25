from model import WizardNet
from player import RLAgent, RandomAgent, Player
from game import Game
import torch

def evaluate(net, n_games=200):
    net.eval()
    totals = []
    for _ in range(n_games):
        players = [Player("rl", 0, RLAgent(net, greedy=True)),
                   Player("r1", 1, RandomAgent()),
                   Player("r2", 2, RandomAgent())]
        game = Game()
        for p in players:
            game.add_player(p)
        game.start()
        totals.append(players[0].points)
    net.train()
    return sum(totals) / len(totals)

def reinforce_loss(net, batch):
    all_G = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    baseline, std = all_G.mean(), all_G.std() + 1e-8      # Baseline + Normierung

    losses = []
    for head_name, head_idx in (("bid", 0), ("play", 1)):
        group = [b for b in batch if b[3] == head_name]
        if not group:
            continue
        enc = torch.from_numpy(np.stack([b[0] for b in group]))
        act = torch.tensor([b[1] for b in group])
        msk = torch.from_numpy(np.stack([b[2] for b in group])).bool()
        adv = (torch.tensor([b[4] for b in group], dtype=torch.float32) - baseline) / std

        logits = net(enc)[head_idx].masked_fill(~msk, float('-inf'))   # GLEICHE Maske
        dist   = torch.distributions.Categorical(logits=logits)
        losses.append(-(dist.log_prob(act) * adv))

    return torch.cat(losses).mean()

obs_dim = 60 + 60 + 180 + 5 + 9 + (1 + 1 + 1)       # dim = 317
max_bid = 20
hidden_dim = 256

net = WizardNet(obs_dim, max_bid, hidden_dim)             # ONE architecture; different agents querry that architecture; game information is agent specific
agents = [RLAgent(net) for _ in range(3)]   
players = [Player(f"p{i}", i, agents[i]) for i in range(3)]   # SELF-PLAY: derselbe Agent
game = Game(players)

opt = torch.optim.Adam(net.parameters(), lr=1e-3)

for episode in range(50_000):
    agents = [RLAgent(net) for _ in range(3)]
    game = Game()
    for i, ag in enumerate(agents):
        game.add_player(Player(f"p{i}", i, ag))
    game.start()

    batch = [t for ag in agents for t in ag.drain_buffer()]
    loss = reinforce_loss(net, batch)
    opt.zero_grad(); loss.backward(); opt.step()

    if episode % 3000 == 0:
        print(f"ep {episode}: {evaluate(net):.1f} Points (RL vs. 2× Random)")
    if episode % 5000 == 0:
        torch.save(net.state_dict(), f"checkpoints/ep{episode}.pt")