import torch
from collections import defaultdict
import numpy as np


def ranks_from_scores(scores: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """scores: [B,N], gold: [B] — higher is better"""
    gold_s = scores[torch.arange(scores.size(0), device=scores.device), gold]
    return 1 + (scores > gold_s.unsqueeze(1)).sum(dim=1)


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
                     train_triples=None, type_constrained=False,
                     id2entity=None, rel_category=None):
    device = next(model.parameters()).device
    E = len(e2id)
    model.eval()
    all_true = set(all_true)

    # precompute true tails and heads (ID-space)
    true_tails = defaultdict(set)
    true_heads = defaultdict(set)
    for (h,r,t) in all_true:
        true_tails[(h,r)].add(t)
        true_heads[(r,t)].add(h)

    # type constraints from train if requested
    allowed_heads = allowed_tails = None
    if type_constrained and train_triples is not None:
        from collections import defaultdict as dd
        ah, at = dd(set), dd(set)
        for (h,r,t) in train_triples:
            ah[r].add(h); at[r].add(t)
        allowed_heads, allowed_tails = ah, at

    # collect ranks (tail side only; head side similar if you want both)
    ranks_all, by_rel = [], defaultdict(list)
    with torch.no_grad():
        for (h,r,t) in test_triples:
            cand = torch.arange(E, device=device)
            # scores for (h,r, cand)
            s = model.score_triples(
                torch.full_like(cand, h), torch.full_like(cand, r), cand
            )

            # filtered masking (keep gold)
            if filtered:
                for t2 in true_tails.get((h,r), ()):
                    if t2 != t:
                        s[t2] = float("-inf")

            # type constraints (keep gold)
            if type_constrained and allowed_tails is not None:
                allowed = allowed_tails.get(r, None)
                if allowed is not None and len(allowed) < E:
                    mask = torch.full((E,), False, device=device)
                    mask[list(allowed)] = True
                    mask[t] = True
                    s = torch.where(mask, s, torch.tensor(float("-inf"), device=device))

            rank = (s > s[t]).sum().item() + 1
            ranks_all.append(rank)
            by_rel[r].append(rank)

    def agg(ranks):
        arr = np.array(ranks, dtype=float)
        out = {"MRR": float(np.mean(1.0/arr)), "MR": float(np.mean(arr))}
        for k in ks: out[f"Hits@{k}"] = float(np.mean(arr <= k))
        return out

    res = {"overall": agg(ranks_all)}
    if per_relation:
        res["per_relation"] = {r: agg(rs) for r, rs in by_rel.items()}

    # --- Per-category aggregation: USE PROVIDED rel_category if available ---
    if rel_category is not None:
        res["per_category"] = aggregate_by_category(test_triples, by_rel, rel_category)
    else:
        # Fallback: build name-based categories (stable)
        id2rel = {i: s for s, i in r2id.items()}
        from data import categorize_relations_by_name
        rc = categorize_relations_by_name(id2rel)
        res["per_category"] = aggregate_by_category(test_triples, by_rel, rc)

    return res


# REPLACE aggregate_by_category with this exact version (no duplication)
from collections import defaultdict
def aggregate_by_category(test_triples, ranks_by_rel, rel_category):
    agg = defaultdict(list)
    for r, ranks in ranks_by_rel.items():          # once per relation
        agg[rel_category.get(r, "other")].extend(ranks)
    def agg_ranks(ranks):
        if not ranks:
            return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "count": 0}
        import numpy as np
        arr = np.array(ranks, dtype=float)
        return {
            "MRR": float((1.0/arr).mean()),
            "Hits@1": float((arr <= 1).mean()),
            "Hits@3": float((arr <= 3).mean()),
            "Hits@10": float((arr <= 10).mean()),
            "count": int(len(ranks)),
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
        X.append(model.score_triples(torch.tensor([h], device=device),
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
            X.append(model.score_triples(torch.tensor([h], device=device),
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