from model import WizNet
from player import RLAgent, RandomAgent, HeuristicAgent, Player
from game import Game
import torch
import numpy as np
import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataclasses import dataclass
import os

import random

torch.set_printoptions(precision=3, sci_mode=False)

LOG_ROOT   = "/home/ipv577/rl_runs"
CKPT_DIR   = "checkpoints"
OPP_CKPT   = "relevant_checkpoints/up_20260815_161453_6000.pt"


@dataclass
class Config:
    """One experiment. The defaults reproduce the current behaviour exactly.

    The five configurations of the ablation differ only in the four fields
    marked below; everything else is held fixed so a difference in the result
    is attributable to one axis.

        A  bid_mode="policy"  use_value_baseline=False  round_weights_exp=0  aux=0.0
        B  bid_mode="policy"  use_value_baseline=True   round_weights_exp=0  aux=0.0
        C  bid_mode="policy"  use_value_baseline=True   round_weights_exp=2  aux=0.0
        D  bid_mode="policy"  use_value_baseline=True   round_weights_exp=2  aux=0.1
        E  bid_mode="ev"      use_value_baseline=True   round_weights_exp=2  aux=0.1
    """
    name: str = "default"

    # --- the four ablation axes -------------------------------------------
    bid_mode: str = "ev"               # "ev" | "policy"
    use_value_baseline: bool = True    # False -> standardised return as baseline
    round_weights_exp: int = 2         # 0 = uniform round sizes, 2 = r**2
    aux: float = 0.1                   # weight of the tricks cross-entropy

    # --- held fixed across the ablation -----------------------------------
    beta_bid: float = 0.05
    beta_play: float = 0.01
    lr: float = 3e-4
    g_scale: float = 50
    p_heur: float = 0.4                # share of heuristic training opponents
    games_per_update: int = 27
    updates: int = 50_000
    seed: int = 0

    # --- architecture ------------------------------------------------------
    obs_dim: int = 60 + 60 + 180 + 5 + 9 + 3       # 317
    max_bid: int = 20
    hidden_dim: int = 256

    @property
    def run_suffix(self):
        return (f"_{self.name}_bid-{self.bid_mode}_vb-{int(self.use_value_baseline)}"
                f"_rw-{self.round_weights_exp}_aux-{self.aux}_ph-{self.p_heur}")

    def policy_heads(self):
        """Heads that receive a policy gradient under this configuration."""
        return {"play"} | ({"bid"} if self.bid_mode == "policy" else set())

def rank_ratio(M):
    if M.shape[0] < 30:                  # zu wenige Zeilen → nicht aussagekräftig
        return float('nan')
    M = M - M.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(M)
    return (s[0] / s.sum()).item()

def evaluate(net, net_opp=None, n_games=200, collect=False, eval_seed=42,
             opponent=None, bid_mode="ev"):
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
            for g_i in range(n_games):
                # Rotate the measured agent through all three seats. game.start()
                # derives the first bidder from the round index, so seat 0 would
                # otherwise be the "screw the dealer" player in EVERY round 20 --
                # worth ~8 points of bid accuracy and biasing every r20 metric.
                seat = g_i % 3
                opp_seats = [i for i in range(3) if i != seat]

                if opponent == "heuristic":
                    opp_agents = [HeuristicAgent(), HeuristicAgent()]
                elif net_opp is not None:
                    opp_agents = [RLAgent(net_opp, greedy=True, bid_mode=bid_mode),
                                  RLAgent(net_opp, greedy=True, bid_mode=bid_mode)]
                else:
                    opp_agents = [RandomAgent(), RandomAgent()]

                seats = [None, None, None]
                rl = Player("rl1", seat, RLAgent(net, greedy=True, debug=collect,
                                                 bid_mode=bid_mode))
                seats[seat] = rl
                for k, i in enumerate(opp_seats):
                    seats[i] = Player(f"o{k+1}", i, opp_agents[k])

                game = Game()
                for p in seats:
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
def reinforce_loss(net, batch, cfg):
    """
    Buffer-Tupel:
        b[0]  enc         np.float32 (317,)   Observation zum Entscheidungszeitpunkt
        b[1]  action      int                 gesampelter Index (nur bei "play" belegt)
        b[2]  mask        np.bool (60,)       legale Karten (nur bei "play" belegt)
        b[3]  head        str                 "bid" oder "play"
        b[4]  G           float               Rundenreward
        b[5]  won_tricks  int                 tatsaechlich gewonnene Stiche (CE-Label)
        b[6]  round_nr    int                 Rundengroesse

    Which heads get a policy gradient depends on cfg.bid_mode: with "policy"
    both heads are trained as policies (configs A-D), with "ev" the bid comes
    from the analytic rule and bid transitions carry only value and tricks
    signal (config E).
    """
    losses, stats = [], {}
    policy_heads = cfg.policy_heads()

    for head_name in ("bid", "play"):
        group = [b for b in batch if b[3] == head_name]
        if not group:
            continue

        enc = torch.from_numpy(np.stack([b[0] for b in group]))
        G   = torch.tensor([b[4] for b in group], dtype=torch.float32) / cfg.g_scale
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

        # --- Policy-Gradient: fuer jeden Kopf, der unter cfg eine Policy ist --
        terms = []
        if head_name in policy_heads:
            head_idx = 0 if head_name == "bid" else 1
            beta     = cfg.beta_bid if head_name == "bid" else cfg.beta_play

            act = torch.tensor([b[1] for b in group])
            msk = torch.from_numpy(np.stack([b[2] for b in group])).bool()
            logits = out[head_idx].masked_fill(~msk, float('-inf'))

            # Without the learned baseline the standardised return is used
            # instead -- that is the configuration A/B difference.
            adv = (G - V.detach()) if cfg.use_value_baseline else G
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            dist = torch.distributions.Categorical(logits=logits)
            loss_return  = -(dist.log_prob(act) * adv).mean()
            loss_entropy = -beta * dist.entropy().mean()
            terms += [loss_return, loss_entropy]

            real    = msk.sum(dim=1) > 1        # echte Wahl, kein Zwangszug
            n_legal = msk.sum(dim=1).float()
            ent     = dist.entropy()

            stats[f"{head_name}_loss_return"]  = loss_return.item()
            stats[f"{head_name}_loss_entropy"] = loss_entropy.item()
            stats[f"{head_name}_entropy"]      = ent.mean().item()
            stats[f"{head_name}_entropy_real"] = ent[real].mean().item()
            stats[f"{head_name}_entropy_norm"] = (ent[real] / n_legal[real].log()).mean().item()
            stats[f"{head_name}_logit_absmax"] = logits[real][msk[real]].abs().max().item()
            stats[f"{head_name}_n_decisions"]  = real.sum().item()

        if cfg.use_value_baseline:
            terms.append(loss_value)
        if cfg.aux > 0:
            terms.append(cfg.aux * loss_tricks)
        if terms:
            losses.append(sum(terms))

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

def train(cfg=None, updates=None):
    """Run one experiment and return the trained network.

    Everything that used to happen at import time lives here now, so the
    module can be imported (from a notebook, a test, an analysis script)
    without starting a run.
    """
    cfg = cfg or Config()
    updates = cfg.updates if updates is None else updates

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed + 10_000)

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_time + cfg.run_suffix
    writer = SummaryWriter(log_dir=f"{LOG_ROOT}/{run_name}")
    os.makedirs(CKPT_DIR, exist_ok=True)   # git cannot track the empty dir

    net = WizNet(cfg.obs_dim, cfg.max_bid, cfg.hidden_dim)
    net_opp = WizNet(cfg.obs_dim, cfg.max_bid, cfg.hidden_dim)
    net_opp.load_state_dict(torch.load(OPP_CKPT), strict=False)

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    for update in range(updates):

        # cfg.games_per_update: with cfg.p_heur of the opponents being heuristics, only
        # the RL agents contribute transitions, so more games are needed to keep the
        # batch at roughly the 13.800 transitions that 20 pure self-play games gave.
        batch = []
        for _ in range(cfg.games_per_update):
            others = [HeuristicAgent() if rng.random() < cfg.p_heur
                      else RLAgent(net, bid_mode=cfg.bid_mode)
                      for _ in range(2)]
            rl = RLAgent(net, bid_mode=cfg.bid_mode)
            agents = [rl] + others
        
            game = Game()
            for i, ag in enumerate(agents):
                game.add_player(Player(f"p{i}", i, ag))
            game.start_sample(cfg.round_weights_exp)
            for ag in agents:
                if isinstance(ag, RLAgent):     # heuristics carry no transitions
                    batch.extend(ag.drain_buffer())

        loss, stats = reinforce_loss(net, batch, cfg)
        opt.zero_grad()
        loss.backward()

        heads = [("play", net.card_head), ("value", net.value_head),
                 ("tricks", net.tricks_head), ("trunk", net.layer2)]
        if cfg.bid_mode == "policy":
            heads.append(("bid", net.bid_head))
        for name, mod in heads:
            if any(p.grad is None for p in mod.parameters()):
                continue                      # head is inactive in this config
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

        # --- Policy-Statistik, fuer jeden Kopf der unter cfg eine Policy ist --
        for h in ("bid", "play"):
            for key, tag in (("loss_return",  f"loss/{h}_return"),
                             ("loss_entropy", f"loss/{h}_entropy"),
                             ("entropy",      f"policy/{h}_entropy"),
                             ("entropy_real", f"policy/{h}_entropy_real"),
                             ("entropy_norm", f"policy/{h}_entropy_norm"),
                             ("logit_absmax", f"policy/{h}_logit_absmax"),
                             ("n_decisions",  f"policy/{h}_n_decisions")):
                if f"{h}_{key}" in stats:
                    writer.add_scalar(tag, stats[f"{h}_{key}"], update)

        # ====================================================================
        if update % 50 == 0:
            collect = (update % 250 == 0)

            if collect:
                points, hits, metrics, bids20 = evaluate(net, n_games=1000, collect=True, bid_mode=cfg.bid_mode)
                writer.add_histogram("bids/round20", bids20, update)
            else:
                points, hits, metrics = evaluate(net, bid_mode=cfg.bid_mode)

            points_opp, hits_opp, metrics_opp = evaluate(net, net_opp, bid_mode=cfg.bid_mode)

            # Heuristics are also training opponents when cfg.p_heur > 0, so this score
            # is no longer a clean generalisation measure -- score_vs_random is the
            # axis that is never trained against. Kept because it is deterministic
            # and therefore low-variance, hence fewer games.
            points_heu, hits_heu, metrics_heu = evaluate(net, n_games=100, opponent="heuristic", bid_mode=cfg.bid_mode)

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
    train(Config(name="default"))
