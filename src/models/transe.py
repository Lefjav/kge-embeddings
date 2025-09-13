import torch, torch.nn as nn

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, dim, margin, device='cuda', **_):
        super().__init__()
        self.ent = nn.Embedding(num_entities, dim)
        self.rel = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.ent.weight); nn.init.xavier_uniform_(self.rel.weight)
        self.p = 1  # L1 works well for TransE
        self.margin = margin
        self.device = device

    def score_triples(self, h, r, t):
        # higher is better -> negative distance
        h_e = self.ent(h); r_e = self.rel(r); t_e = self.ent(t)
        dist = torch.norm(h_e + r_e - t_e, p=self.p, dim=-1)
        return -dist
