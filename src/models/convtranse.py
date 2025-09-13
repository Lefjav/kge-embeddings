import torch, torch.nn as nn, torch.nn.functional as F

class ConvTransE(nn.Module):
    def __init__(self, num_entities, num_relations, dim,
                 convtranse_kernel_size=3, convtranse_channels=32,
                 device='cuda', **_):
        super().__init__()
        self.ent = nn.Embedding(num_entities, dim)
        self.rel = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.ent.weight); nn.init.xavier_uniform_(self.rel.weight)
        self.conv = nn.Conv1d(1, convtranse_channels, kernel_size=convtranse_kernel_size, padding='same')
        self.proj = nn.Linear(convtranse_channels*dim, dim)
        self.device = device

    def score_triples(self, h, r, t):
        h_e = self.ent(h); r_e = self.rel(r)
        x = torch.cat([h_e, r_e], dim=-1)[:, None, :]  # [B,1,2d]
        x = F.relu(self.conv(x))                       # [B,C,2d]
        x = x[..., :h_e.size(-1)]                      # fold back to d
        x = self.proj(x.flatten(1))                    # [B,d]
        all_t = self.ent.weight
        return torch.matmul(x, all_t.t())[torch.arange(h.size(0)), t]
