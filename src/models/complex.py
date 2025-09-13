import torch, torch.nn as nn

class ComplEx(nn.Module):
    def __init__(self, num_entities, num_relations, dim, device='cuda', **_):
        super().__init__()
        self.dim = dim
        self.ent = nn.Embedding(num_entities, dim*2)  # real|imag
        self.rel = nn.Embedding(num_relations, dim*2)
        nn.init.xavier_uniform_(self.ent.weight); nn.init.xavier_uniform_(self.rel.weight)
        self.device = device

    def score_triples(self, h, r, t):
        hr, hi = torch.chunk(self.ent(h), 2, dim=-1)
        rr, ri = torch.chunk(self.rel(r), 2, dim=-1)
        tr, ti = torch.chunk(self.ent(t), 2, dim=-1)
        # real( <h, r, conj(t)> )
        score = (hr*rr*tr + hr*ri*ti + hi*rr*ti - hi*ri*tr).sum(-1)
        return score
