from model import WizNet
from player import RLAgent, RandomAgent, Player
from game import Game
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_improved_metrics.01"
writer = SummaryWriter(log_dir=f"/home/ipv577/rl_runs/{run_name}")

torch.set_printoptions(precision=3, sci_mode=False)

# hyperparams
BETA = 0.01
LR = 3e-4

def rank_ratio(M):
    if M.shape[0] < 30:                  # zu wenige Zeilen → nicht aussagekräftig
        return float('nan')
    M = M - M.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(M)
    return (s[0] / s.sum()).item()

def evaluate(net, n_games=200, collect=False):
    net.eval()
    totals, hits = [], []
    logits_log = []

    with torch.no_grad():
        for _ in range(n_games):
            players = [Player("rl", 0, RLAgent(net, greedy=True, debug=collect)),
                    Player("r1", 1, RandomAgent()),
                    Player("r2", 2, RandomAgent())]
            game = Game()
            for p in players:
                game.add_player(p)
            game.start()

            totals.append(players[0].points)
            rl = players[0]
            hits.append(rl.bid_hits / rl.rounds_played)     # Trefferquote über ALLE 20 Runden

            if collect:
                logits_log.extend(rl.agent.bid_logits_log)
    net.train()

    if collect: 
        M = torch.stack([lg for lg, _ in logits_log])              # [n, 21]
        rsizes = torch.tensor([r for _, r in logits_log])           # [n]
        return sum(totals)/len(totals), hits, M, rsizes

    return sum(totals)/len(totals), hits    
    

# buffer = (enc, action, mask, head, G)             ; bids und alle aktionen mit G; einzelner buffer JE SPIELER; wächst in länge über Game
# pending = (enc, idx, mask.numpy(), "bid"/"play")  ; aktionen einen runde (je SPIELER) ohne G; wächst in länge über Runde
# batch = concatinierter buffer aller spieler; bei 3 spielern wäre das len(batch) = 690; eig. zu wenig für einen step; wir wollen lieber 10 - 20 games sammeln. update: machen wir jetzt; ein batch hält 20 * buffer (=20 games)
def reinforce_loss(net, batch):
    losses = []
    stats = {}
    for head_name, head_idx in (("bid", 0), ("play", 1)):
        group = [b for b in batch if b[3] == head_name]

        '''
        b = (enc, action, mask, head, G)
        #    b[0]  b[1]   b[2]  b[3]  b[4]

        b[0]  np.ndarray float32, shape (317,)   # die encodierte Observation zum Zeitpunkt der Entscheidung
        b[1]  int                                 # welcher Index gesampelt wurde, z.B. 5
        b[2]  np.ndarray bool, shape (21,)/(60,)  # welche Indizes legal waren
        b[3]  str, "bid" oder "play"
        b[4]  float                               # der Return, den du beim Drainen zugewiesen hast
        '''    

        if not group:
            continue
        G = torch.tensor([b[4] for b in group], dtype=torch.float32)
        adv = (G - G.mean()) / (G.std() + 1e-8)          # Baseline JE Kopf
        #print(adv)

        enc = torch.from_numpy(np.stack([b[0] for b in group]))
        act = torch.tensor([b[1] for b in group])
        msk = torch.from_numpy(np.stack([b[2] for b in group])).bool()

        logits = net(enc)[head_idx].masked_fill(~msk, float('-inf'))
        dist   = torch.distributions.Categorical(logits=logits)

        loss_return  = -(dist.log_prob(act) * adv).mean()   # <- .mean() hier dazu → Skalar pro Kopf
        loss_entropy = - BETA * dist.entropy().mean()
        losses.append(loss_return + loss_entropy)           # 0-D
        real = msk.sum(dim=1) > 1                           # was there more than 1 option within the mask? only than we have a "real" decision and should measure
        n_legal = msk.sum(dim=1).float()
        # pack values and send them out of function to log them in main train loop
        stats[f"{head_name}_entropy_real"]      = dist.entropy()[real].mean().item()
        stats[f"{head_name}_entropy"]      = dist.entropy().mean().item()
        stats[f"{head_name}_logit_absmax"] = logits[real][msk[real]].abs().max().item()
        stats[f"{head_name}_entropy_norm"] = (dist.entropy()[real] / n_legal[real].log()).mean().item()
        stats[f"{head_name}_n_decisions"] = real.sum().item()

    return torch.stack(losses).mean(), stats

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

opt = torch.optim.Adam(net.parameters(), lr=LR)

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
    
    loss, stats = reinforce_loss(net, batch)
    opt.zero_grad(); loss.backward(); opt.step()

    writer.add_scalar("loss/total",                     loss.item(),                    update)
    writer.add_scalar("policy/bid_entropy",             stats["bid_entropy"],           update)
    writer.add_scalar("policy/bid_entropy_real",        stats["bid_entropy_real"],      update)
    writer.add_scalar("policy/play_entropy",            stats["play_entropy"],          update)
    writer.add_scalar("policy/play_entropy_real",       stats["play_entropy_real"],     update)
    writer.add_scalar("policy/bid_logit_absmax",        stats["bid_logit_absmax"],      update)
    writer.add_scalar("policy/play_logit_absmax",       stats["play_logit_absmax"],     update)
    writer.add_scalar("policy/play_entropy_norm",       stats["play_entropy_norm"],     update)
    writer.add_scalar("policy/bid_entropy_norm",        stats["bid_entropy_norm"],      update)
    writer.add_scalar("policy/bid_n_decisions",         stats["bid_n_decisions"],       update)
    writer.add_scalar("policy/play_n_decisions",        stats["play_n_decisions"],      update)

    if update % 50 == 0:
        collect = (update % 500 == 0)        # Rang seltener, ist teurer

        if collect:
            points, hits, M, rsizes = evaluate(net, collect=True)
            writer.add_scalar("rank/bid_all", rank_ratio(M), update)
            for r in (5, 10, 20):
                writer.add_scalar(f"rank/bid_r{r}", rank_ratio(M[rsizes == r]), update)
        else:
            points, hits = evaluate(net)

        writer.add_scalar("eval/score_vs_random", points, update)
        writer.add_scalar("eval/bid_accuracy", sum(hits)/len(hits), update)
        writer.flush()
        print(f"up {update}: {points:.1f} Points & bid=won {sum(hits)/len(hits):.3f}")

    if update % 300 == 0:
        torch.save(net.state_dict(), f"checkpoints/up_{run_time}_{update}.pt")