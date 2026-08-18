from model import WizNet
from player import RLAgent, RandomAgent, Player
from game import Game
import torch
import numpy as np


from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import random
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_r**2_round_sampling"
writer = SummaryWriter(log_dir=f"/home/ipv577/rl_runs/{run_name}")

torch.set_printoptions(precision=3, sci_mode=False)

# hyperparams
BETA_BID = 0.05
BETA_PLAY = 0.01
LR = 3e-4
G_SCALE = 50

def rank_ratio(M):
    if M.shape[0] < 30:                  # zu wenige Zeilen → nicht aussagekräftig
        return float('nan')
    M = M - M.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(M)
    return (s[0] / s.sum()).item()

def evaluate(net, net_opp=None, n_games=200, collect=False, eval_seed=42):
    py_state = random.getstate()
    th_state = torch.get_rng_state()
    random.seed(eval_seed)
    torch.manual_seed(eval_seed)

    net.eval()
    if net_opp is not None:
        net_opp.eval()
    totals, hits = [], []
    logits_log = []

    with torch.no_grad():
        rows = []               # history to store player.round_logs over multiple games
        for _ in range(n_games):
            player = [Player("rl1", 0, RLAgent(net, greedy=True, debug=collect))]
            
            if net_opp is not None:
                others = [Player("other1", 1, RLAgent(net_opp, greedy=True)),
                            Player("other2", 2, RLAgent(net_opp, greedy=True))]
            else:
                others = [Player("r1", 1, RandomAgent()),
                            Player("r2", 2, RandomAgent())]
            players = player + others

            game = Game()
            for p in players:
                game.add_player(p)
            game.start()

            totals.append(players[0].points)
            rl = players[0]
            hits.append(rl.bid_hits / rl.rounds_played)     # Trefferquote über ALLE 20 Runden

            rows.extend(rl.round_log)
                
                
            if collect:
                logits_log.extend(rl.agent.bid_logits_log)

        r  = torch.tensor([x[0] for x in rows])
        bd = torch.tensor([x[1] for x in rows])
        wn = torch.tensor([x[2] for x in rows])

        for size in (3, 8, 14, 20):
            m = r == size
            if m.sum() == 0: continue
            d = (bd[m] - wn[m]).float()
            writer.add_scalar(f"bias/mean_r{size}", d.mean().item(), update)
            writer.add_scalar(f"bias/mae_r{size}",  d.abs().mean().item(), update)
            writer.add_scalar(f"acc/r{size}",       (d == 0).float().mean().item(), update)
            
    net.train()

    if collect: 
        M = torch.stack([lg for lg, _ in logits_log])              # [n, 21]
        rsizes = torch.tensor([r for _, r in logits_log])           # [n]

        random.setstate(py_state)
        torch.set_rng_state(th_state)
        return sum(totals)/len(totals), hits, M, rsizes


    random.setstate(py_state)
    torch.set_rng_state(th_state)
    return sum(totals)/len(totals), hits

         
    

# buffer = (enc, action, mask, head, G)             ; bids und alle aktionen mit G; einzelner buffer JE SPIELER; wächst in länge über Game
# pending = (enc, idx, mask.numpy(), "bid"/"play")  ; aktionen einen runde (je SPIELER) ohne G; wächst in länge über Runde
# batch = concatinierter buffer aller spieler; bei 3 spielern wäre das len(batch) = 690; eig. zu wenig für einen step; wir wollen lieber 10 - 20 games sammeln. update: machen wir jetzt; ein batch hält 20 * buffer (=20 games)
def reinforce_loss(net, batch):
    losses = []
    stats = {}
    for head_name, head_idx in (("bid", 0), ("play", 1)):
        group = [b for b in batch if b[3] == head_name]
        if not group:
            continue
        '''
        b = (enc, action, mask, head, G)
        #    b[0]  b[1]   b[2]  b[3]  b[4]

        b[0]  np.ndarray float32, shape (317,)   # die encodierte Observation zum Zeitpunkt der Entscheidung
        b[1]  int                                 # welcher Index gesampelt wurde, z.B. 5
        b[2]  np.ndarray bool, shape (21,)/(60,)  # welche Indizes legal waren
        b[3]  str, "bid" oder "play"
        b[4]  float                               # der Return, den du beim Drainen zugewiesen hast
        '''    

        G = torch.tensor([b[4] for b in group], dtype=torch.float32) / G_SCALE
        adv = (G - G.mean()) / (G.std() + 1e-8)          # Baseline JE Kopf
        #print(adv)

        enc = torch.from_numpy(np.stack([b[0] for b in group]))
        act = torch.tensor([b[1] for b in group])
        msk = torch.from_numpy(np.stack([b[2] for b in group])).bool()

        out = net(enc)
        logits = out[head_idx].masked_fill(~msk, float('-inf'))
        V = out[2]

        adv = G - V.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        dist   = torch.distributions.Categorical(logits=logits)
        loss_return  = -(dist.log_prob(act) * adv).mean()   # <- .mean() hier dazu → Skalar pro Kopf
        beta = BETA_BID if head_name == "bid" else BETA_PLAY
        loss_entropy = - beta * dist.entropy().mean()
        loss_value = (V-G).pow(2).mean()

        losses.append(loss_return + loss_entropy + loss_value)           # 0-D

        real = msk.sum(dim=1) > 1                           # was there more than 1 option within the mask? only than we have a "real" decision and should measure
        n_legal = msk.sum(dim=1).float()

        with torch.no_grad():
            vc = np.corrcoef(V.detach().numpy(), G.numpy())[0, 1]
        stats[f"{head_name}_value_corr"] = float(vc)

        # pack values and send them out of function to log them in main train loop
        stats[f"{head_name}_loss_return"] = loss_return.item()
        stats[f"{head_name}_loss_entropy"] = loss_entropy.item()
        stats[f"{head_name}_loss_value"] = loss_value.item()
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
net_opp = WizNet(obs_dim, max_bid, hidden_dim)
net_opp.load_state_dict(torch.load("relevant_checkpoints/up_20260815_161453_6000.pt"), strict=False)

#agents = [RLAgent(net) for _ in range(3)]   
#players = [Player(f"p{i}", i, agents[i]) for i in range(3)]   # SELF-PLAY: derselbe Agent
#player1, player2, player3 = players
#game = Game()
#game.add_player(player1)
#game.add_player(player2)
#game.add_player(player3)

opt = torch.optim.Adam(net.parameters(), lr=LR)

for update in range(50_000):

    # create a batch of 20 games to reduce impact of noise; 1 game = 690 Transitions, 20 games = 13.800k transitions; 1.200 bid transistions, 12.600 play transitions
    batch = []
    for _ in range(20):
        agents = [RLAgent(net) for _ in range(3)]
        game = Game()
        for i, ag in enumerate(agents):
            game.add_player(Player(f"p{i}", i, ag))
        game.start_sample()

        for ag in agents:
            batch.extend(ag.drain_buffer())
    
    loss, stats = reinforce_loss(net, batch)
    opt.zero_grad()
    loss.backward()

    g = torch.cat([p.grad.flatten() for p in net.bid_head.parameters()])
    writer.add_scalar("grad/bid_head_norm", g.norm().item(), update)

    opt.step()
    for h in ("bid", "play"):
        writer.add_scalar(f"loss/{h}_return",  stats[f"{h}_loss_return"],  update)
        writer.add_scalar(f"loss/{h}_entropy", stats[f"{h}_loss_entropy"], update)
        writer.add_scalar(f"loss/{h}_value",   stats[f"{h}_loss_value"],   update)
        writer.add_scalar(f"value/{h}_corr",   stats[f"{h}_value_corr"],   update)

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
        collect = (update % 250 == 0)        # Rang seltener, ist teurer

        if collect:
            points, hits, M, rsizes = evaluate(net, n_games=1000, collect=True)         # rang: 1x rl vs. 2x random agents
            writer.add_scalar("rank/bid_all_random", rank_ratio(M), update)
            for r in (5, 10, 20):
                writer.add_scalar(f"rank/bid_r{r}_random", rank_ratio(M[rsizes == r]), update)

            bids20 = M[rsizes == 20].argmax(dim=1)
            writer.add_histogram("bids/round20", bids20, update)
            writer.add_scalar("bids/mean_r20", bids20.float().mean().item(), update)
            writer.add_scalar("bids/max_r20",  bids20.max().item(), update)

        else:
            points, hits = evaluate(net)                    # play against random agents

        points_opp, hits_opp = evaluate(net, net_opp)   # play against rl instance

        writer.add_scalar("eval/score_vs_random", points, update)
        writer.add_scalar("eval/score_vs_rl", points_opp, update)
        writer.add_scalar("eval/bid_accuracy_random", sum(hits)/len(hits), update)
        writer.add_scalar("eval/bid_accuracy_rl", sum(hits_opp)/len(hits_opp), update)
        writer.flush()
        print(f"up {update}: {points:.1f} Points & bid=won {sum(hits)/len(hits):.3f} [RANDOM]")
        print(f"up {update}: {points_opp:.1f} Points & bid=won {sum(hits_opp)/len(hits_opp):.3f} [RL]")

    if update % 250 == 0:
        torch.save(net.state_dict(), f"checkpoints/up_{run_time}_{update}.pt")