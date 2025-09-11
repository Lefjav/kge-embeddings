import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):

    def __init__(self, entity_count, relation_count, device, norm=1, dim=100, margin=1.0, use_soft_loss=True):
        super(TransE, self).__init__()
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.device = device
        self.norm = norm
        self.dim = dim
        self.use_soft_loss = use_soft_loss
        self.entities_emb = self._init_enitity_emb()
        self.relations_emb = self._init_relation_emb()
        if not use_soft_loss:
            self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')

    def _init_enitity_emb(self):
        entities_emb = nn.Embedding(num_embeddings=self.entity_count + 1,
                                    embedding_dim=self.dim,
                                    padding_idx=self.entity_count)
        uniform_range = 6 / np.sqrt(self.dim)
        entities_emb.weight.data.uniform_(-uniform_range, uniform_range)
        return entities_emb

    def _init_relation_emb(self):
        relations_emb = nn.Embedding(num_embeddings=self.relation_count + 1,
                                     embedding_dim=self.dim,
                                     padding_idx=self.relation_count)
        uniform_range = 6 / np.sqrt(self.dim)
        relations_emb.weight.data.uniform_(-uniform_range, uniform_range)
        # -1 to avoid nan for OOV vector
        relations_emb.weight.data[:-1, :].div_(relations_emb.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        return relations_emb

    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):
        """Return model losses based on the input.

        :param positive_triplets: triplets of positives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :param negative_triplets: triplets of negatives in (B*num_negs)x3 shape (B*num_negs - batch*negatives, 3 - head, relation and tail)
        :return: tuple of the model loss, positive triplets loss component, negative triples loss component
        """
        # -1 to avoid nan for OOV vector
        self.entities_emb.weight.data[:-1, :].div_(self.entities_emb.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))

        assert positive_triplets.size()[1] == 3
        positive_distances = self._distance(positive_triplets)

        assert negative_triplets.size()[1] == 3
        negative_distances = self._distance(negative_triplets)

        return self.loss(positive_distances, negative_distances), positive_distances, negative_distances

    def predict(self, triplets: torch.LongTensor):
        """Calculated dissimilarity score for given triplets.

        :param triplets: triplets in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: dissimilarity score for given triplets
        """
        return self._distance(triplets)

    def loss(self, positive_distances, negative_distances):
        """Handle multiple negatives per positive with soft or margin loss."""
        batch_size = positive_distances.size(0)
        num_negs = negative_distances.size(0) // batch_size
        
        # Reshape negative distances to [batch_size, num_negs]
        negative_distances = negative_distances.view(batch_size, num_negs)
        
        if self.use_soft_loss:
            # 2) Soft loss: convert distances to scores (smaller distance = higher score)
            pos_scores = -positive_distances.unsqueeze(1)  # [batch_size, 1]
            neg_scores = -negative_distances  # [batch_size, num_negs]
            
            # Softplus pairwise loss: log(1 + exp(neg_score - pos_score))
            # We want pos_score > neg_score, so minimize when neg_score > pos_score
            loss_per_pair = F.softplus(neg_scores - pos_scores)  # [batch_size, num_negs]
            return loss_per_pair.mean(dim=1)  # [batch_size]
        else:
            # Original margin ranking loss
            # Repeat positive distances to match negative shape
            positive_distances = positive_distances.unsqueeze(1).expand(-1, num_negs)
            
            # Create target tensor (we want positive < negative, so target = -1)
            target = torch.full((batch_size, num_negs), -1, dtype=torch.long, device=self.device)
            
            # Compute loss for each positive-negative pair
            loss_per_pair = self.criterion(positive_distances, negative_distances, target)
            
            # Return mean loss per positive (average over negatives)
            return loss_per_pair.mean(dim=1)

    def _distance(self, triplets):
        """Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id."""
        assert triplets.size()[1] == 3
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
        return (self.entities_emb(heads) + self.relations_emb(relations) - self.entities_emb(tails)).norm(p=self.norm,
                                                                                                          dim=1)

    def score_triple(self, h, r, t):
        """
        h, r, t: 1D tensors of same length (IDs)
        return: 1D tensor of scores (higher = better)
        """
        # For TransE: higher score = better (neg distance)
        emb_h = self.entities_emb(h)
        emb_r = self.relations_emb(r)
        emb_t = self.entities_emb(t)
        return -torch.norm(emb_h + emb_r - emb_t, p=self.norm, dim=1)

    def score_hr_t(self, h, r, all_tails):
        """
        h, r: 1D tensors (typically length 1), all_tails: 1D tensor of tail IDs
        return: 1D tensor of |all_tails| scores
        """
        # Expand h,r to match all_tails and call score_triple
        H = h.expand_as(all_tails)
        R = r.expand_as(all_tails)
        return self.score_triple(H, R, all_tails)



# Updated ComplEx class for model.py
class ComplEx(nn.Module):
    """ComplEx model implementation for knowledge graph embedding."""
    
    def __init__(self, entity_count, relation_count, device, dim=100, margin=1.0, use_soft_loss=True):
        super(ComplEx, self).__init__()
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.device = device
        self.dim = dim
        self.use_soft_loss = use_soft_loss
        
        # Complex embeddings: real and imaginary parts
        self.entities_emb_re = self._init_entity_emb()
        self.entities_emb_im = self._init_entity_emb()
        self.relations_emb_re = self._init_relation_emb()
        self.relations_emb_im = self._init_relation_emb()
        
        if not use_soft_loss:
            self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')
    
    def _init_entity_emb(self):
        entities_emb = nn.Embedding(num_embeddings=self.entity_count + 1,
                                    embedding_dim=self.dim,
                                    padding_idx=self.entity_count)
        # Xavier initialization for better convergence
        nn.init.xavier_uniform_(entities_emb.weight)
        entities_emb.weight.data[self.entity_count].fill_(0)  # padding
        return entities_emb
    
    def _init_relation_emb(self):
        relations_emb = nn.Embedding(num_embeddings=self.relation_count + 1,
                                     embedding_dim=self.dim,
                                     padding_idx=self.relation_count)
        # Xavier initialization
        nn.init.xavier_uniform_(relations_emb.weight)
        relations_emb.weight.data[self.relation_count].fill_(0)  # padding
        return relations_emb
    
    def forward(self, positive_triplets, negative_triplets):
        positive_scores = self._score(positive_triplets)
        negative_scores = self._score(negative_triplets)
        return self.loss(positive_scores, negative_scores), positive_scores, negative_scores
    
    def predict(self, triplets):
        return self._score(triplets)
    
    def _score(self, triplets):
        """Calculate ComplEx score for triplets."""
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
        
        # Get embeddings
        h_re = self.entities_emb_re(heads)
        h_im = self.entities_emb_im(heads)
        r_re = self.relations_emb_re(relations)
        r_im = self.relations_emb_im(relations)
        t_re = self.entities_emb_re(tails)
        t_im = self.entities_emb_im(tails)
        
       
        
        # Real part only (ComplEx uses only real part)
        real_part = (h_re * r_re * t_re + h_im * r_re * t_im + 
                    h_re * r_im * t_im - h_im * r_im * t_re).sum(dim=1)
        
        # Return negative score (lower is better for distance-based loss)
        return -real_part
    
    def loss(self, positive_scores, negative_scores):
        batch_size = positive_scores.size(0)
        num_negs = negative_scores.size(0) // batch_size
        negative_scores = negative_scores.view(batch_size, num_negs)
        
        if self.use_soft_loss:
            pos_scores = positive_scores.unsqueeze(1)
            neg_scores = negative_scores
            loss_per_pair = F.softplus(neg_scores - pos_scores)
            return loss_per_pair.mean(dim=1)
        else:
            positive_scores = positive_scores.unsqueeze(1).expand(-1, num_negs)
            target = torch.full((batch_size, num_negs), -1, dtype=torch.long, device=self.device)
            loss_per_pair = self.criterion(positive_scores, negative_scores, target)
            return loss_per_pair.mean(dim=1)

    def score_triple(self, h, r, t):
        """
        h, r, t: 1D tensors of same length (IDs)
        return: 1D tensor of scores (higher = better)
        """
        # For ComplEx: higher score = better (return positive score)
        heads = h
        relations = r
        tails = t
        
        # Get embeddings
        h_re = self.entities_emb_re(heads)
        h_im = self.entities_emb_im(heads)
        r_re = self.relations_emb_re(relations)
        r_im = self.relations_emb_im(relations)
        t_re = self.entities_emb_re(tails)
        t_im = self.entities_emb_im(tails)
        
        # Real part only (ComplEx uses only real part)
        real_part = (h_re * r_re * t_re + h_im * r_re * t_im + 
                    h_re * r_im * t_im - h_im * r_im * t_re).sum(dim=1)
        
        return real_part

    def score_hr_t(self, h, r, all_tails):
        """
        h, r: 1D tensors (typically length 1), all_tails: 1D tensor of tail IDs
        return: 1D tensor of |all_tails| scores
        """
        # Expand h,r to match all_tails and call score_triple
        H = h.expand_as(all_tails)
        R = r.expand_as(all_tails)
        return self.score_triple(H, R, all_tails)


class RotatE(nn.Module):
    """RotatE model implementation for knowledge graph embedding."""
    
    def __init__(self, entity_count, relation_count, device, dim=100, margin=1.0, use_soft_loss=True):
        super(RotatE, self).__init__()
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.device = device
        self.dim = dim
        self.use_soft_loss = use_soft_loss
        
        # Entity embeddings (complex)
        self.entities_emb_re = self._init_entity_emb()
        self.entities_emb_im = self._init_entity_emb()
        # Relation embeddings (rotation angles)
        self.relations_emb = self._init_relation_emb()
        
        if not use_soft_loss:
            self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')
    
    def _init_entity_emb(self):
        entities_emb = nn.Embedding(num_embeddings=self.entity_count + 1,
                                    embedding_dim=self.dim,
                                    padding_idx=self.entity_count)
        uniform_range = 6 / np.sqrt(self.dim)
        entities_emb.weight.data.uniform_(-uniform_range, uniform_range)
        return entities_emb
    
    def _init_relation_emb(self):
        relations_emb = nn.Embedding(num_embeddings=self.relation_count + 1,
                                     embedding_dim=self.dim,
                                     padding_idx=self.relation_count)
        # Initialize relation embeddings to small values for rotation angles
        relations_emb.weight.data.uniform_(-1, 1)
        return relations_emb
    
    def forward(self, positive_triplets, negative_triplets):
        positive_distances = self._distance(positive_triplets)
        negative_distances = self._distance(negative_triplets)
        return self.loss(positive_distances, negative_distances), positive_distances, negative_distances
    
    def predict(self, triplets):
        return self._distance(triplets)
    
    def _distance(self, triplets):
        """Calculate RotatE distance for triplets."""
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
        
        # Get embeddings
        h_re = self.entities_emb_re(heads)
        h_im = self.entities_emb_im(heads)
        r_angles = self.relations_emb(relations)
        t_re = self.entities_emb_re(tails)
        t_im = self.entities_emb_im(tails)
        
        # Convert rotation angles to complex numbers
        r_re = torch.cos(r_angles)
        r_im = torch.sin(r_angles)
        
        # RotatE: h * r - t (complex multiplication)
        # (h_re + i*h_im) * (r_re + i*r_im) - (t_re + i*t_im)
        result_re = h_re * r_re - h_im * r_im - t_re
        result_im = h_re * r_im + h_im * r_re - t_im
        
        # L2 norm of complex difference
        distance = torch.sqrt(result_re**2 + result_im**2 + 1e-8).sum(dim=1)
        return distance
    
    def loss(self, positive_distances, negative_distances):
        batch_size = positive_distances.size(0)
        num_negs = negative_distances.size(0) // batch_size
        negative_distances = negative_distances.view(batch_size, num_negs)
        
        if self.use_soft_loss:
            pos_scores = -positive_distances.unsqueeze(1)
            neg_scores = -negative_distances
            loss_per_pair = F.softplus(neg_scores - pos_scores)
            return loss_per_pair.mean(dim=1)
        else:
            positive_distances = positive_distances.unsqueeze(1).expand(-1, num_negs)
            target = torch.full((batch_size, num_negs), -1, dtype=torch.long, device=self.device)
            loss_per_pair = self.criterion(positive_distances, negative_distances, target)
            return loss_per_pair.mean(dim=1)

    def score_triple(self, h, r, t):
        """
        h, r, t: 1D tensors of same length (IDs)
        return: 1D tensor of scores (higher = better)
        """
        # For RotatE: higher score = better (neg distance)
        heads = h
        relations = r
        tails = t
        
        # Get embeddings
        h_re = self.entities_emb_re(heads)
        h_im = self.entities_emb_im(heads)
        r_angles = self.relations_emb(relations)
        t_re = self.entities_emb_re(tails)
        t_im = self.entities_emb_im(tails)
        
        # Convert rotation angles to complex numbers
        r_re = torch.cos(r_angles)
        r_im = torch.sin(r_angles)
        
        # RotatE: h * r - t (complex multiplication)
        # (h_re + i*h_im) * (r_re + i*r_im) - (t_re + i*t_im)
        result_re = h_re * r_re - h_im * r_im - t_re
        result_im = h_re * r_im + h_im * r_re - t_im
        
        # L2 norm of complex difference
        distance = torch.sqrt(result_re**2 + result_im**2 + 1e-8).sum(dim=1)
        return -distance

    def score_hr_t(self, h, r, all_tails):
        """
        h, r: 1D tensors (typically length 1), all_tails: 1D tensor of tail IDs
        return: 1D tensor of |all_tails| scores
        """
        # Expand h,r to match all_tails and call score_triple
        H = h.expand_as(all_tails)
        R = r.expand_as(all_tails)
        return self.score_triple(H, R, all_tails)