import torch.nn as nn
import torch.nn.functional as F

class WizNet(nn.Module):
    def __init__(self, obs_dim, max_bid, hidden_dim):
        super().__init__()

        self.layer1 = nn.Linear(obs_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bid_head = nn.Linear(hidden_dim, max_bid + 1)
        self.card_head = nn.Linear(hidden_dim, 60)

    def forward(self, enc_obs):
        h = F.relu(self.layer1(enc_obs))
        h = F.relu(self.layer2(h))

        return self.bid_head(h), self.card_head(h)
