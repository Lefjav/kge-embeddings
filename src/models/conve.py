import torch, torch.nn as nn, torch.nn.functional as F

class ConvE(nn.Module):
    def __init__(self, num_entities, num_relations, dim, conve_hidden_channels=32,
                 conve_dropout=0.2, device='cuda', **_):
        super().__init__()
        self.ent = nn.Embedding(num_entities, dim)
        self.rel = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.ent.weight); nn.init.xavier_uniform_(self.rel.weight)
        self.hc = conve_hidden_channels
        self.dim = dim
        self.device = device
        # reshape (2, sqrt(dim), sqrt(dim)) if dim is a square; fallback to (2, dim, 1)
        self.side = int(dim**0.5)
        self.use_square = (self.side*self.side == dim)
        in_h, in_w = (self.side, self.side) if self.use_square else (dim, 1)
        self.conv = nn.Conv2d(2, self.hc, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(conve_dropout)
        self.fc = nn.Linear(self.hc*in_h*in_w, dim)

    def _embed2d(self, h, r):
        he = self.ent(h); re = self.rel(r)
        if self.use_square:
            he = he.view(-1, 1, self.side, self.side)
            re = re.view(-1, 1, self.side, self.side)
        else:
            he = he.view(-1, 1, self.dim, 1)
            re = re.view(-1, 1, self.dim, 1)
        x = torch.cat([he, re], dim=1)
        x = self.dropout(F.relu(self.conv(x)))
        x = x.flatten(1)
        x = self.dropout(self.fc(x))  # shape [B, dim]
        return x

    def score_triples(self, h, r, t):
        x = self._embed2d(h, r)                   # projected query
        all_t = self.ent.weight                   # [N, dim]
        return torch.matmul(x, all_t.t())[torch.arange(h.size(0)), t]  # take the true t scores
