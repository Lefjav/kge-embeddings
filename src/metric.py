import torch
from collections import defaultdict
import numpy as np


def hits_at_k(pred: torch.Tensor, gold: torch.Tensor, k: int = 10, largest: bool = False) -> float:
    """Calculates hits@k score.
    
    :param pred: [B, N] tensor of prediction scores where B is batch size and N is number of classes
    :param gold: [B] tensor with ground truth class indices
    :param k: number of top K results to be considered as hits
    :param largest: if True, higher scores are better; if False, lower scores are better (for distances)
    :return: Hits@K score as fraction in [0,1]
    """
    topk = pred.topk(k=k, largest=largest).indices                         # [B, k]
    return topk.eq(gold.view(-1,1)).any(dim=1).float().mean().item()       # fraction in [0,1]


def mrr(pred: torch.Tensor, gold: torch.Tensor, largest: bool = False) -> float:
    """Calculates mean reciprocal rank (MRR).
    
    :param pred: [B, N] tensor of prediction scores where B is batch size and N is number of classes
    :param gold: [B] tensor with ground truth class indices
    :param largest: if True, higher scores are better; if False, lower scores are better (for distances)
    :return: Mean reciprocal rank score
    """
    # rank = 1 for best; with distances we want ascending order
    order = pred.argsort(dim=1, descending=largest)                         # [B, N]
    # find position of gold per row
    pos = order.eq(gold.view(-1,1)).nonzero(as_tuple=False)                 # [B, 2], rows sorted
    ranks = pos[:,1].float() + 1.0
    return (1.0 / ranks).mean().item()


def hits_at_k_via_ranks(scores, gold, k, smaller_is_better=True):
    """Tie-safe hits@k calculation using rank counting method."""
    if smaller_is_better:
        gold_scores = scores[torch.arange(scores.size(0), device=scores.device), gold]
        ranks = 1 + (scores < gold_scores.unsqueeze(1)).sum(dim=1)
    else:
        gold_scores = scores[torch.arange(scores.size(0), device=scores.device), gold]
        ranks = 1 + (scores > gold_scores.unsqueeze(1)).sum(dim=1)
    return (ranks <= k).float().mean().item()


def build_filter_sets(paths):
    """Build set of known triples for filtered evaluation."""
    triples = set()
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                h, r, t = line.rstrip("\n").split("\t")
                triples.add((h, r, t))
    return triples


def evaluate_filtered_tail(model, dataset, entity2id, relation2id, known_triples, device, debug_spoiler=True):
    """Evaluate tail prediction with filtered evaluation."""
    model.eval()
    E = len(entity2id)
    scores_all, gold_all, heads_all = [], [], []

    with torch.no_grad():
        for h, r, t in dataset:     # h,r,t are already indices (thanks to Dataset)
            # filter: remove all true tails for (h,r,?)
            h_str = list(entity2id.keys())[list(entity2id.values()).index(h)]
            r_str = list(relation2id.keys())[list(relation2id.values()).index(r)]
            true_tails = {entity2id[x[2]] for x in known_triples if x[0]==h_str and x[1]==r_str and x[2] in entity2id}

            # candidates = all entities except filtered true tails, but keep gold
            mask = torch.ones(E, dtype=torch.bool)
            idx_true = torch.tensor(list(true_tails), dtype=torch.long)
            mask[idx_true] = False
            mask[t] = True  # ensure gold remains
            
            # B) Schema-aware masking: forbid self tail for hierarchical relations
            # For hierarchical relations (child/parent), head != tail by definition
            relation_is_hierarchical = r_str in ['is_child_of', 'has_child', 'is_parent_of', 'has_parent', 
                                                'parent_of', 'child_of', 'subclass_of', 'superclass_of']
            if relation_is_hierarchical:
                mask[h] = False  # forbid self tail

            # score (h,r,all_candidates)
            candidates = torch.arange(E, dtype=torch.long, device=device)[mask.to(device)]
            batch_h = torch.full_like(candidates, h, device=device)
            batch_r = torch.full_like(candidates, r, device=device)
            trip = torch.stack([batch_h, batch_r, candidates], dim=1)
            d = model.predict(trip)   # [num_cands]  (distance; smaller is better)

            # make a dense vector of size E filled with +inf, then write candidate distances
            vec = torch.full((E,), float("inf"), device=device)
            vec[mask.to(device)] = d
            scores_all.append(vec.unsqueeze(0))
            gold_all.append(torch.tensor([t], device=device))
            heads_all.append(torch.tensor([h], device=device))

    scores = torch.cat(scores_all, dim=0)         # [B, E]
    gold   = torch.cat(gold_all, dim=0).view(-1)  # [B]
    
    # Debugging checks
    print("=== TAIL PREDICTION DEBUG ===")
    print("has_nan:", torch.isnan(scores).any().item(), "has_inf:", torch.isinf(scores).any().item())
    
    # Tie-safe rank calculation
    gold_scores = scores[torch.arange(scores.size(0), device=scores.device), gold]
    ranks = 1 + (scores < gold_scores.unsqueeze(1)).sum(dim=1)
    h1_alt = (ranks == 1).float().mean().item()
    print("H@1_alt =", h1_alt)
    
    # Histogram of ranks
    hist = torch.bincount(ranks.clamp(max=20).cpu(), minlength=21)
    print("rank<=10 frac =", (ranks<=10).float().mean().item(), "rank==2 frac =", (ranks==2).float().mean().item())
    
    # Sanity check: gold must be in candidate set
    assert torch.all(torch.isfinite(gold_scores)).item(), "Gold is masked out (or became inf/NaN)."
    
    # A) Spoiler detection
    if debug_spoiler:
        heads = torch.cat(heads_all, dim=0).view(-1)  # [B]
        top1 = scores.argmin(dim=1)  # because distance: smaller is better
        spoiler_is_head_frac = (top1 == heads).float().mean().item()
        vals, counts = torch.unique(top1, return_counts=True)
        top_spoiler = vals[counts.argmax()].item()
        print("spoiler_is_head_frac:", spoiler_is_head_frac, "top_spoiler_id:", top_spoiler, "count:", counts.max().item())
    
    return {
        "MRR": mrr(scores, gold, largest=False),
        "H@1": hits_at_k(scores, gold, k=1, largest=False),
        "H@3": hits_at_k(scores, gold, k=3, largest=False),
        "H@10": hits_at_k(scores, gold, k=10, largest=False),
        "num_queries": scores.size(0),
    }


def evaluate_filtered_head(model, dataset, entity2id, relation2id, known_triples, device, debug_spoiler=True):
    """Evaluate head prediction with filtered evaluation."""
    model.eval()
    E = len(entity2id)
    scores_all, gold_all, tails_all = [], [], []

    with torch.no_grad():
        for h, r, t in dataset:     # h,r,t are already indices (thanks to Dataset)
            # filter: remove all true heads for (?,r,t)
            h_str = list(entity2id.keys())[list(entity2id.values()).index(h)]
            r_str = list(relation2id.keys())[list(relation2id.values()).index(r)]
            t_str = list(entity2id.keys())[list(entity2id.values()).index(t)]
            true_heads = {entity2id[x[0]] for x in known_triples if x[0] in entity2id and x[1]==r_str and x[2]==t_str}

            # candidates = all entities except filtered true heads, but keep gold
            mask = torch.ones(E, dtype=torch.bool)
            idx_true = torch.tensor(list(true_heads), dtype=torch.long)
            mask[idx_true] = False
            mask[h] = True  # ensure gold remains
            
            # B) Schema-aware masking: forbid self head for hierarchical relations
            # For hierarchical relations (child/parent), head != tail by definition
            relation_is_hierarchical = r_str in ['is_child_of', 'has_child', 'is_parent_of', 'has_parent', 
                                                'parent_of', 'child_of', 'subclass_of', 'superclass_of']
            if relation_is_hierarchical:
                mask[t] = False  # forbid self head

            # score (all_candidates,r,t)
            candidates = torch.arange(E, dtype=torch.long, device=device)[mask.to(device)]
            batch_r = torch.full_like(candidates, r, device=device)
            batch_t = torch.full_like(candidates, t, device=device)
            trip = torch.stack([candidates, batch_r, batch_t], dim=1)
            d = model.predict(trip)   # [num_cands]  (distance; smaller is better)

            # make a dense vector of size E filled with +inf, then write candidate distances
            vec = torch.full((E,), float("inf"), device=device)
            vec[mask.to(device)] = d
            scores_all.append(vec.unsqueeze(0))
            gold_all.append(torch.tensor([h], device=device))
            tails_all.append(torch.tensor([t], device=device))

    scores = torch.cat(scores_all, dim=0)         # [B, E]
    gold   = torch.cat(gold_all, dim=0).view(-1)  # [B]
    
    # Debugging checks
    print("=== HEAD PREDICTION DEBUG ===")
    print("has_nan:", torch.isnan(scores).any().item(), "has_inf:", torch.isinf(scores).any().item())
    
    # Tie-safe rank calculation
    gold_scores = scores[torch.arange(scores.size(0), device=scores.device), gold]
    ranks = 1 + (scores < gold_scores.unsqueeze(1)).sum(dim=1)
    h1_alt = (ranks == 1).float().mean().item()
    print("H@1_alt =", h1_alt)
    
    # Histogram of ranks
    hist = torch.bincount(ranks.clamp(max=20).cpu(), minlength=21)
    print("rank<=10 frac =", (ranks<=10).float().mean().item(), "rank==2 frac =", (ranks==2).float().mean().item())
    
    # Sanity check: gold must be in candidate set
    assert torch.all(torch.isfinite(gold_scores)).item(), "Gold is masked out (or became inf/NaN)."
    
    # A) Spoiler detection for head prediction
    if debug_spoiler:
        tails = torch.cat(tails_all, dim=0).view(-1)  # [B]
        top1 = scores.argmin(dim=1)  # because distance: smaller is better
        spoiler_is_tail_frac = (top1 == tails).float().mean().item()
        vals, counts = torch.unique(top1, return_counts=True)
        top_spoiler = vals[counts.argmax()].item()
        print("spoiler_is_tail_frac:", spoiler_is_tail_frac, "top_spoiler_id:", top_spoiler, "count:", counts.max().item())
    
    return {
        "MRR": mrr(scores, gold, largest=False),
        "H@1": hits_at_k(scores, gold, k=1, largest=False),
        "H@3": hits_at_k(scores, gold, k=3, largest=False),
        "H@10": hits_at_k(scores, gold, k=10, largest=False),
        "num_queries": scores.size(0),
    }


# Legacy functions for backward compatibility
def hit_at_k(predictions: torch.Tensor, ground_truth_idx: torch.Tensor, device: torch.device, k: int = 10) -> int:
    """Legacy function - use hits_at_k instead."""
    # Convert to new format: [B] instead of [B, 1]
    gold = ground_truth_idx.squeeze() if ground_truth_idx.dim() > 1 else ground_truth_idx
    hits_score = hits_at_k(predictions, gold, k=k, largest=False)
    return int(hits_score * predictions.size(0))  # Convert back to count for legacy compatibility


def _build_constraints(train_triples):
    """Build type constraints from training triples."""
    from collections import defaultdict
    allowed_heads, allowed_tails = defaultdict(set), defaultdict(set)
    for (h, r, t) in train_triples:
        allowed_heads[r].add(h)
        allowed_tails[r].add(t)
    return allowed_heads, allowed_tails


def evaluate_ranking(model, test_triples, all_true, e2id, r2id,
                     filtered=True, per_relation=True, ks=(1,3,10), 
                     train_triples=None, type_constrained=False):
    """Comprehensive ranking evaluation with optional filtering and per-relation metrics."""
    device = next(model.parameters()).device
    E = len(e2id)

    # Precompute true tails for efficiency
    true_tails = defaultdict(set)
    for (h, r, t) in all_true:
        true_tails[(h, r)].add(t)

    # Build type constraints if requested
    allowed_heads, allowed_tails = None, None
    if type_constrained and train_triples is not None:
        allowed_heads, allowed_tails = _build_constraints(train_triples)

    def rank_tail(h, r, t):
        tails = torch.arange(E, device=device)
        scores = model.score_hr_t(torch.tensor([h], device=device),
                                  torch.tensor([r], device=device),
                                  tails)
        if filtered:
            with torch.no_grad():
                # mask true (h,r,t') except the actual t
                for tprime in true_tails.get((h, r), set()):
                    if tprime != t:
                        scores[tprime] = -1e9  # assuming higher is better
        
        # Apply type constraints
        if type_constrained and allowed_tails is not None:
            allowed = allowed_tails.get(r, None)
            if allowed is not None and len(allowed) < E:
                mask = torch.ones(E, dtype=torch.bool, device=device)
                mask[list(allowed)] = False
                scores[mask] = -1e9
        
        rank = (scores > scores[t]).sum().item() + 1
        return rank

    def agg(ranks):
        arr = np.array(ranks, dtype=float)
        out = {"MRR": np.mean(1.0/arr), "MR": np.mean(arr)}
        for k in ks: 
            out[f"Hits@{k}"] = np.mean(arr <= k)
        return out

    ranks_all, by_rel = [], defaultdict(list)
    for (h, r, t) in test_triples:
        rk = rank_tail(h, r, t)
        ranks_all.append(rk)
        by_rel[r].append(rk)

    res = {"overall": agg(ranks_all)}
    if per_relation:
        res["per_relation"] = {r: agg(rs) for r, rs in by_rel.items()}
    
    # Optional: aggregate by category
    if train_triples is not None:
        from data import categorize_relations
        # Create entity type mapping for categorization
        e2type = {}
        for (h, r, t) in train_triples:
            if h not in e2type:
                e2type[h] = "CWE" if str(h).startswith("CWE-") else "CAPEC" if str(h).startswith("CAPEC-") else "CWE"
            if t not in e2type:
                e2type[t] = "CWE" if str(t).startswith("CWE-") else "CAPEC" if str(t).startswith("CAPEC-") else "CWE"
        
        rel_category = categorize_relations(train_triples, e2type, {})
        res["per_category"] = aggregate_by_category(test_triples, by_rel, rel_category)
    
    return res


def aggregate_by_category(test_triples, ranks_by_rel, rel_category):
    """Aggregate evaluation results by relation category."""
    from collections import defaultdict
    agg = defaultdict(list)
    for (h, r, t) in test_triples:
        if r in ranks_by_rel and len(ranks_by_rel[r]):
            agg[rel_category.get(r, "other")].extend(ranks_by_rel[r])
    
    # Compute MRR/Hits@k per category
    def agg_ranks(ranks):
        if not ranks:
            return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "count": 0}
        arr = np.array(ranks, dtype=float)
        return {
            "MRR": np.mean(1.0/arr),
            "Hits@1": np.mean(arr <= 1),
            "Hits@3": np.mean(arr <= 3),
            "Hits@10": np.mean(arr <= 10),
            "count": len(ranks)
        }
    
    return {cat: agg_ranks(ranks) for cat, ranks in agg.items()}


def fit_calibrator(model, valid_triples, method="platt", n_neg_per_pos=1):
    """Fit a calibrator to convert model scores to probabilities."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    device = next(model.parameters()).device
    X, y = [], []
    E = max(max(h, t) for (h, _, t) in valid_triples) + 1
    all_true = set(valid_triples)

    for (h, r, t) in valid_triples:
        # Positive example
        X.append(model.score_triple(torch.tensor([h], device=device),
                                    torch.tensor([r], device=device),
                                    torch.tensor([t], device=device)).item())
        y.append(1)
        
        # Negative examples
        for _ in range(n_neg_per_pos):
            # uniform negative tail not in all_true
            while True:
                tneg = np.random.randint(0, E)
                if (h, r, tneg) not in all_true:
                    break
            X.append(model.score_triple(torch.tensor([h], device=device),
                                        torch.tensor([r], device=device),
                                        torch.tensor([tneg], device=device)).item())
            y.append(0)

    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    
    if method == "platt":
        lr = LogisticRegression(max_iter=1000).fit(X, y)
        class _Platt: 
            def transform(self, s): 
                return lr.predict_proba(np.array(s).reshape(-1, 1))[:, 1]
        return _Platt()
    elif method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip").fit(X.ravel(), y)
        class _Iso:
            def transform(self, s):
                return ir.predict(np.array(s))
        return _Iso()
    else:
        return None