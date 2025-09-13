import torch, torch.nn as nn

class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, dim, margin, device='cuda', **_):
        super().__init__()
        # use half-dim for real/imag parts internally
        self.dim = dim
        self.ent = nn.Embedding(num_entities, dim*2)   # [real, imag]
        self.rel = nn.Embedding(num_relations, dim)    # phase
        nn.init.uniform_(self.ent.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.rel.weight, a=-3.1416, b=3.1416)
        self.device = device

    def score_triples(self, h, r, t):
        h_e = self.ent(h); t_e = self.ent(t)
        hr, hi = torch.chunk(h_e, 2, dim=-1)
        tr, ti = torch.chunk(t_e, 2, dim=-1)
        phase = self.rel(r)  # radians
        cos, sin = torch.cos(phase), torch.sin(phase)
        rr = hr*cos - hi*sin
        ri = hr*sin + hi*cos
        dist = torch.norm(torch.cat([rr-tr, ri-ti], dim=-1), p=1, dim=-1)
        return -dist
