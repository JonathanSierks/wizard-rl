from model import WizNet
from player import RLAgent, RandomAgent, HeuristicAgent, Player
from game import Game
import torch
import numpy as np
import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

import random

torch.set_printoptions(precision=3, sci_mode=False)

# --- run setup -------------------------------------------------------------
SEED       = 0
RUN_SUFFIX = "_ev_bidding_heuristic_baseline"
UPDATES    = 50_000
LOG_ROOT   = "/home/ipv577/rl_runs"
CKPT_DIR   = "checkpoints"

# --- architecture ----------------------------------------------------------
OBS_DIM    = 60 + 60 + 180 + 5 + 9 + 3       # 317
MAX_BID    = 20
HIDDEN_DIM = 256
OPP_CKPT   = "relevant_checkpoints/up_20260815_161453_6000.pt"

# hyperparams
BETA_BID = 0.05
BETA_PLAY = 0.01
LR = 3e-4
G_SCALE = 50
AUX = 0.1
P_HEUR = 0.4

def rank_ratio(M):
    if M.shape[0] < 30:                  # zu wenige Zeilen → nicht aussagekräftig
        return float('nan')
    M = M - M.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(M)
    return (s[0] / s.sum()).item()

def evaluate(net, net_opp=None, n_games=200, collect=False, eval_seed=42, opponent=None):
    """Score `net` against a table of two opponents.

    opponent="heuristic" -> the rule-based reference agent (fixed strength,
                            deterministic, never trained against)
    net_opp given         -> a frozen learned checkpoint
    otherwise             -> random agents
    """
    py_state = random.getstate()
    th_state = torch.get_rng_state()
    random.seed(eval_seed); torch.manual_seed(eval_seed)

    net.eval()
    if net_opp is not None:
        net_opp.eval()

    try:
        totals, hits, rows, cmp_log = [], [], [], []

        with torch.no_grad():
            for _ in range(n_games):
                rl = Player("rl1", 0, RLAgent(net, greedy=True, debug=collect))
                if opponent == "heuristic":
                    others = [Player("h1", 1, HeuristicAgent()),
                              Player("h2", 2, HeuristicAgent())]
                elif net_opp is not None:
                    others = [Player("o1", 1, RLAgent(net_opp, greedy=True)),
                              Player("o2", 2, RLAgent(net_opp, greedy=True))]
                else:
                    others = [Player("r1", 1, RandomAgent()),
                              Player("r2", 2, RandomAgent())]

                game = Game()
                for p in [rl] + others:
                    game.add_player(p)
                game.start()

                totals.append(rl.points)
                hits.append(rl.bid_hits / rl.rounds_played)
                rows.extend(rl.round_log)
                if collect:
                    cmp_log.extend(rl.agent.bid_compare_log)

        metrics = {}
        bids20  = torch.tensor([])

        # --- Gebotsqualitaet pro Rundengroesse (misst die gespielten Gebote) ---
        rr = torch.tensor([x[0] for x in rows])
        bd = torch.tensor([x[1] for x in rows])
        wn = torch.tensor([x[2] for x in rows])
        for size in (3, 8, 14, 20):
            m = rr == size
            if m.sum() == 0: continue
            d = (bd[m] - wn[m]).float()
            metrics[f"bias/mean_r{size}"] = d.mean().item()
            metrics[f"bias/mae_r{size}"]  = d.abs().mean().item()
            metrics[f"acc/r{size}"]       = (d == 0).float().mean().item()

        # --- EV-Gebot vs. (ungenutzter) Bid-Head ---
        if cmp_log:
            C = torch.tensor(cmp_log)          # [n, 4]: r, bid_head, ev, mode
            bids20 = C[C[:, 0] == 20, 2]
            for size in (3, 8, 14, 20):
                m = C[:, 0] == size
                if m.sum() == 0: continue
                metrics[f"bid_ev/mean_r{size}"]     = C[m, 2].float().mean().item()
                metrics[f"bid_ev/max_r{size}"]      = C[m, 2].max().item()
                metrics[f"bid_head/mean_r{size}"]   = C[m, 1].float().mean().item()
                metrics[f"bid_head/max_r{size}"]    = C[m, 1].max().item()
                metrics[f"bid_ev/agree_r{size}"]    = (C[m, 1] == C[m, 2]).float().mean().item()

        score = sum(totals) / len(totals)
        if collect:
            return score, hits, metrics, bids20
        return score, hits, metrics

    finally:
        random.setstate(py_state)
        torch.set_rng_state(th_state)

# buffer = (enc, action, mask, head, G)             ; bids und alle aktionen mit G; einzelner buffer JE SPIELER; wächst in länge über Game
# pending = (enc, idx, mask.numpy(), "bid"/"play")  ; aktionen einen runde (je SPIELER) ohne G; wächst in länge über Runde
# batch = concatinierter buffer aller spieler; bei 3 spielern wäre das len(batch) = 690; eig. zu wenig für einen step; wir wollen lieber 10 - 20 games sammeln. update: machen wir jetzt; ein batch hält 20 * buffer (=20 games)
def reinforce_loss(net, batch):
    """
    Buffer-Tupel:
        b[0]  enc         np.float32 (317,)   Observation zum Entscheidungszeitpunkt
        b[1]  action      int                 gesampelter Index (nur bei "play" belegt)
        b[2]  mask        np.bool (60,)       legale Karten (nur bei "play" belegt)
        b[3]  head        str                 "bid" oder "play"
        b[4]  G           float               Rundenreward
        b[5]  won_tricks  int                 tatsaechlich gewonnene Stiche (CE-Label)
        b[6]  round_nr    int                 Rundengroesse

    Gebote entstehen seit dem Umbau aus der EV-Rechnung ueber den tricks_head,
    nicht mehr aus einer gesampelten Policy. Bid-Transitions liefern deshalb
    nur noch Value- und Tricks-Signal, keinen Policy-Gradienten.
    """
    losses, stats = [], {}

    for head_name in ("bid", "play"):
        group = [b for b in batch if b[3] == head_name]
        if not group:
            continue

        enc = torch.from_numpy(np.stack([b[0] for b in group]))
        G   = torch.tensor([b[4] for b in group], dtype=torch.float32) / G_SCALE
        won = torch.tensor([b[5] for b in group])                       # [B] long
        rs  = torch.tensor([b[6] for b in group])                       # [B]

        out = net(enc)
        V             = out[2]
        tricks_logits = out[3]                                          # [B, 21]

        # --- Tricks-Head: CE gegen beobachtete Stichzahl -----------------
        valid = torch.arange(21)[None, :] <= rs[:, None]                # [B, 21]
        tricks_logits = tricks_logits.masked_fill(~valid, float('-inf'))
        loss_tricks = F.cross_entropy(tricks_logits, won)

        # --- Value-Head --------------------------------------------------
        loss_value = (V - G).pow(2).mean()

        # --- Policy-Gradient: nur fuer Kartenentscheidungen --------------
        if head_name == "play":
            act = torch.tensor([b[1] for b in group])
            msk = torch.from_numpy(np.stack([b[2] for b in group])).bool()
            logits = out[1].masked_fill(~msk, float('-inf'))

            adv = G - V.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            dist = torch.distributions.Categorical(logits=logits)
            loss_return  = -(dist.log_prob(act) * adv).mean()
            loss_entropy = -BETA_PLAY * dist.entropy().mean()

            losses.append(loss_return + loss_entropy + loss_value + AUX * loss_tricks)

            real    = msk.sum(dim=1) > 1        # echte Wahl, kein Zwangszug
            n_legal = msk.sum(dim=1).float()
            ent     = dist.entropy()

            stats["play_loss_return"]   = loss_return.item()
            stats["play_loss_entropy"]  = loss_entropy.item()
            stats["play_entropy"]       = ent.mean().item()
            stats["play_entropy_real"]  = ent[real].mean().item()
            stats["play_entropy_norm"]  = (ent[real] / n_legal[real].log()).mean().item()
            stats["play_logit_absmax"]  = logits[real][msk[real]].abs().max().item()
            stats["play_n_decisions"]   = real.sum().item()
        else:
            losses.append(loss_value + AUX * loss_tricks)

        # --- gemeinsame Statistik ----------------------------------------
        with torch.no_grad():
            vc = np.corrcoef(V.numpy(), G.numpy())[0, 1]
        stats[f"{head_name}_value_corr"]  = float(vc)
        stats[f"{head_name}_loss_value"]  = loss_value.item()
        stats[f"{head_name}_loss_tricks"] = loss_tricks.item()

        pred_tricks = tricks_logits.argmax(dim=1)
        stats[f"tricks_mae_at_{head_name}"] = (pred_tricks - won).abs().float().mean().item()
        for r in (3, 8, 14, 20):
            m = rs == r
            if m.sum() > 0:
                stats[f"tricks_mae_r{r}_at_{head_name}"] = \
                    (pred_tricks[m] - won[m]).abs().float().mean().item()

    return torch.stack(losses).mean(), stats

def train(updates=UPDATES, seed=SEED, run_suffix=RUN_SUFFIX):
    """Run the training loop and return the trained network.

    Everything that used to happen at import time lives here now, so the
    module can be imported (from a notebook, a test, an analysis script)
    without starting a run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed + 10_000)

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_time + run_suffix
    writer = SummaryWriter(log_dir=f"{LOG_ROOT}/{run_name}")
    os.makedirs(CKPT_DIR, exist_ok=True)   # git cannot track the empty dir

    net = WizNet(OBS_DIM, MAX_BID, HIDDEN_DIM)
    net_opp = WizNet(OBS_DIM, MAX_BID, HIDDEN_DIM)
    net_opp.load_state_dict(torch.load(OPP_CKPT), strict=False)

    opt = torch.optim.Adam(net.parameters(), lr=LR)

    for update in range(updates):

        # 27 games per update: with P_HEUR of the opponents being heuristics, only
        # the RL agents contribute transitions, so more games are needed to keep the
        # batch at roughly the 13.800 transitions that 20 pure self-play games gave.
        batch = []
        for _ in range(27):
            others = [HeuristicAgent() if rng.random() < P_HEUR else RLAgent(net)
                 for _ in range(2)]
            rl = RLAgent(net)
            agents = [rl] + others
        
            game = Game()
            for i, ag in enumerate(agents):
                game.add_player(Player(f"p{i}", i, ag))
            game.start_sample()
            for ag in agents:
                if isinstance(ag, RLAgent):     # heuristics carry no transitions
                    batch.extend(ag.drain_buffer())

        loss, stats = reinforce_loss(net, batch)
        opt.zero_grad()
        loss.backward()

        for name, mod in (("play", net.card_head), ("value", net.value_head),
                          ("tricks", net.tricks_head), ("trunk", net.layer2)):
            g = torch.cat([p.grad.flatten() for p in mod.parameters()])
            writer.add_scalar(f"grad/{name}_norm", g.norm().item(), update)

        opt.step()

        # --- Tricks- und Value-Head: beide Transitionstypen ------------------
        for h in ("bid", "play"):
            writer.add_scalar(f"loss/{h}_value",            stats[f"{h}_loss_value"],  update)
            writer.add_scalar(f"value/{h}_corr",            stats[f"{h}_value_corr"],  update)
            writer.add_scalar(f"tricks/loss_tricks_at_{h}", stats[f"{h}_loss_tricks"], update)
            writer.add_scalar(f"tricks/mae_at_{h}",         stats[f"tricks_mae_at_{h}"], update)
            for r in (3, 8, 14, 20):
                k = f"tricks_mae_r{r}_at_{h}"
                if k in stats:
                    writer.add_scalar(f"tricks/mae_r{r}_at_{h}", stats[k], update)

        # --- Play-Policy: nur hier gibt es noch einen Policy-Gradienten ------
        writer.add_scalar("loss/play_return",         stats["play_loss_return"],  update)
        writer.add_scalar("loss/play_entropy",        stats["play_loss_entropy"], update)
        writer.add_scalar("policy/play_entropy",      stats["play_entropy"],      update)
        writer.add_scalar("policy/play_entropy_real", stats["play_entropy_real"], update)
        writer.add_scalar("policy/play_entropy_norm", stats["play_entropy_norm"], update)
        writer.add_scalar("policy/play_logit_absmax", stats["play_logit_absmax"], update)
        writer.add_scalar("policy/play_n_decisions",  stats["play_n_decisions"],  update)

        # ====================================================================
        if update % 50 == 0:
            collect = (update % 250 == 0)

            if collect:
                points, hits, metrics, bids20 = evaluate(net, n_games=1000, collect=True)
                writer.add_histogram("bids/round20", bids20, update)
            else:
                points, hits, metrics = evaluate(net)

            points_opp, hits_opp, metrics_opp = evaluate(net, net_opp)

            # Heuristics are now also training opponents (P_HEUR), so this score
            # is no longer a clean generalisation measure -- score_vs_random is the
            # axis that is never trained against. Kept because it is deterministic
            # and therefore low-variance, hence fewer games.
            points_heu, hits_heu, metrics_heu = evaluate(net, n_games=100, opponent="heuristic")

            writer.add_scalar("eval/score_vs_random",         points, update)
            writer.add_scalar("eval/score_vs_rl",             points_opp, update)
            writer.add_scalar("eval/score_vs_heuristic",      points_heu, update)
            writer.add_scalar("eval/bid_accuracy_random",     sum(hits)/len(hits), update)
            writer.add_scalar("eval/bid_accuracy_rl",         sum(hits_opp)/len(hits_opp), update)
            writer.add_scalar("eval/bid_accuracy_heuristic",  sum(hits_heu)/len(hits_heu), update)

            for k, v in metrics.items():
                writer.add_scalar(f"{k}_random", v, update)
            for k, v in metrics_opp.items():
                writer.add_scalar(f"{k}_rl", v, update)
            for k, v in metrics_heu.items():
                writer.add_scalar(f"{k}_heuristic", v, update)

            writer.flush()
            print(f"up {update}: {points:.1f} Points & bid=won {sum(hits)/len(hits):.3f} [RANDOM]")
            print(f"up {update}: {points_opp:.1f} Points & bid=won {sum(hits_opp)/len(hits_opp):.3f} [RL]")
            print(f"up {update}: {points_heu:.1f} Points & bid=won {sum(hits_heu)/len(hits_heu):.3f} [HEURISTIC]")

        if update % 250 == 0:
            torch.save(net.state_dict(), os.path.join(CKPT_DIR, f"up_{run_time}_{update}.pt"))

    writer.close()
    return net


if __name__ == "__main__":
    train()
