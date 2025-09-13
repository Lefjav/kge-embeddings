from absl import app
from absl import flags
import data
from models import build_model
import os
import storage
import torch
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils import tensorboard
from collections import defaultdict
import time
import json
import math

FLAGS = flags.FLAGS


def save_ckpt(state, path):
    """Save checkpoint with automatic directory creation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_ckpt(path, model, optimizer=None, map_location="cpu"):
    """Load checkpoint and return epoch and best_tc_mrr."""
    chk = torch.load(path, map_location=map_location)
    model.load_state_dict(chk["model"])
    if optimizer and "optim" in chk:
        optimizer.load_state_dict(chk["optim"])
    return chk.get("epoch", 0), chk.get("best_tc_mrr", None)


# Core model parameters
flags.DEFINE_enum("model", default="TransE", enum_values=["TransE", "RotatE", "ComplEx", "ConvE", "ConvTransE"], 
                  help="Model to use: TransE, RotatE, ComplEx, ConvE, or ConvTransE")
flags.DEFINE_integer("embedding_dim", default=200, help="Embedding dimension for entities and relations")
flags.DEFINE_float("margin", default=12.0, help="Margin value for TransE/RotatE margin-ranking loss")
flags.DEFINE_enum("loss", default="margin", enum_values=["margin", "bce"], help="Loss function: margin or binary cross-entropy")
flags.DEFINE_integer("neg_ratio", default=64, help="Number of negatives per positive")
flags.DEFINE_float("lr", default=1e-3, help="Learning rate value")
flags.DEFINE_integer("epochs", default=200, help="Number of training epochs")
flags.DEFINE_integer("batch_size", default=1024, help="Batch size for training")
flags.DEFINE_integer("eval_every", default=2, help="Validate every N epochs")
flags.DEFINE_bool("eval_filtered", default=True, help="Use filtered ranking metrics")
flags.DEFINE_integer("early_stop_patience", default=6, help="Stop if no TC-MRR improvement for this many validations")
flags.DEFINE_float("early_stop_delta", default=0.002, help="Minimal improvement in TC-MRR to reset patience")
flags.DEFINE_string("checkpoint_dir", default="./checkpoints", help="Root folder for checkpoints/<ModelName>")
flags.DEFINE_string("results_dir", default="./results", help="Where to write results_*.txt")
flags.DEFINE_bool("nouse_gpu", default=False, help="Disable GPU usage")
flags.DEFINE_string("resume", default="", help="Path to a checkpoint (.pt) to resume from")

# ConvE/ConvTransE specific parameters
flags.DEFINE_integer("conve_hidden_channels", default=32, help="Hidden channels for ConvE")
flags.DEFINE_float("conve_dropout", default=0.2, help="Dropout rate for ConvE")
flags.DEFINE_integer("convtranse_kernel_size", default=3, help="Kernel size for ConvTransE")
flags.DEFINE_integer("convtranse_channels", default=32, help="Number of channels for ConvTransE")

# Additional parameters
flags.DEFINE_integer("seed", default=1234, help="Seed value.")
flags.DEFINE_integer("validation_batch_size", default=64, help="Maximum batch size during model validation.")
flags.DEFINE_integer("norm", default=1, help="Norm used for calculating dissimilarity metric (usually 1 or 2).")
flags.DEFINE_string("dataset_path", default="./mitre-data/txt",
                    help="Path to TXT splits: train.txt/valid.txt/test.txt")
flags.DEFINE_bool("multi_hop_aware", default=True, help="Use multi-hop aware training techniques.")
flags.DEFINE_float("direct_relation_weight", default=2.0, help="Weight boost for direct CWE-CAPEC relationships.")
flags.DEFINE_bool("relation_aware_negatives", default=True, help="Use relation-aware negative sampling (now optimized).")
flags.DEFINE_string("tensorboard_log_dir", default="./runs", help="Path for tensorboard log directory.")
flags.DEFINE_bool("self_adversarial", default=True, help="Use self-adversarial weighting for negatives.")

# add eval & calibration toggles
flags.DEFINE_bool("eval_per_relation", default=True, help="Report per-relation metrics")
flags.DEFINE_bool("eval_type_constrained", default=True, help="Apply type-domain/range masks in eval")
flags.DEFINE_string("calibration", default="none",
                    help="Score->prob calibration: none|platt|isotonic")

# add the lightweight direct-over-2hop separation knobs
flags.DEFINE_float("twohop_margin", default=0.5,
                   help="δ margin for direct vs 2-hop ranking")
flags.DEFINE_float("twohop_weight", default=0.1,
                   help="λ weight for direct>2hop loss term")
flags.DEFINE_float("twohop_keep_ratio", default=0.2,
                   help="Keep ratio for 2-hop hard negatives in sampler (0..1)")
flags.DEFINE_enum("twohop_scope", "direct_only", ["all", "direct_only", "none"],
                  help="Apply two-hop margin/sampler filtering to: all|direct_only|none")
# If your relation names are strings, list the hierarchical ones here:
flags.DEFINE_list("hier_relations", ["is_child_of", "has_child"],
                  help="Relations to ignore in 2-hop expansion (transitive hierarchy).")

# optional: simple AMP switch
flags.DEFINE_bool("amp", default=True, help="Enable torch.cuda.amp autocast")



def build_known_sets_id(paths, entity2id, relation2id):
    """Build filtered sets in ID space for efficient evaluation."""
    tails_by_hr = defaultdict(set)  # (h_id,r_id) -> {t_id}
    heads_by_rt = defaultdict(set)  # (r_id,t_id) -> {h_id}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                h, r, t = line.rstrip("\n").split("\t")
                if h in entity2id and r in relation2id and t in entity2id:
                    hi, ri, ti = entity2id[h], relation2id[r], entity2id[t]
                    tails_by_hr[(hi, ri)].add(ti)
                    heads_by_rt[(ri, ti)].add(hi)
    return tails_by_hr, heads_by_rt


def bernoulli_p(triples):
    """Compute Bernoulli corruption probabilities for each relation."""
    by_r_h, by_r_t = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    for h, r, t in triples:
        by_r_h[r][h].add(t)
        by_r_t[r][t].add(h)
    p = {}
    for r in by_r_h:
        tph = sum(len(s) for s in by_r_h[r].values()) / max(len(by_r_h[r]), 1)
        hpt = sum(len(s) for s in by_r_t[r].values()) / max(len(by_r_t[r]), 1)
        p[r] = tph / max(tph + hpt, 1e-9)   # prob to corrupt tail
    return p


# Import the unified rank helper from metric.py
from metric import ranks_from_scores




# Unfiltered evaluation function
def eval_tail_unfiltered(model, dataset, E, device, hier_rel_ids, etype_tensor=None, type_constrained=False, allowed_tails=None):
    """Unfiltered tail evaluation - no masking of true triples."""
    model.eval()
    all_ranks = []
    for h, r, t in dataset:
        # No filtering - evaluate against all entities
        cand = torch.arange(E, device=device)
        bh = torch.full_like(cand, h, device=device)
        br = torch.full_like(cand, r, device=device)
        trip = torch.stack([bh, br, cand], dim=1)
        scores = model.score_triples(trip[:, 0], trip[:, 1], trip[:, 2])
        
        # Type-constrained evaluation: use train-observed tail ranges per relation
        if type_constrained and allowed_tails is not None:
            allowed = allowed_tails.get(r, None)
            if allowed is not None and len(allowed) < E:
                # Only allow entities that appeared as tails for this relation in training
                mask = torch.tensor([i in allowed for i in range(E)], device=device)
                scores = torch.where(mask, scores, torch.tensor(float("-inf"), device=device))
        
        # rank for this query (higher=better)
        rank = ranks_from_scores(scores.unsqueeze(0), torch.tensor([t], device=device))
        all_ranks.append(rank)
        
        # Sanity checks
        assert (rank >= 1).all() and (rank <= E).all(), f"Invalid rank: {rank}"

    ranks = torch.cat(all_ranks, dim=0)
    
    return {
        "MRR": float((1.0 / ranks.float()).mean().item()),
        "H@1": float((ranks <= 1).float().mean().item()),
        "H@3": float((ranks <= 3).float().mean().item()),
        "H@10": float((ranks <= 10).float().mean().item()),
        "num_queries": int(ranks.numel())
    }


def eval_head_unfiltered(model, dataset, E, device, hier_rel_ids, etype_tensor=None, type_constrained=False, allowed_heads=None):
    """Unfiltered head evaluation - no masking of true triples."""
    model.eval()
    all_ranks = []
    for h, r, t in dataset:
        # No filtering - evaluate against all entities
        cand = torch.arange(E, device=device)
        br = torch.full_like(cand, r, device=device)
        bt = torch.full_like(cand, t, device=device)
        trip = torch.stack([cand, br, bt], dim=1)
        scores = model.score_triples(trip[:, 0], trip[:, 1], trip[:, 2])
        
        # Type-constrained evaluation: use train-observed head ranges per relation
        if type_constrained and allowed_heads is not None:
            allowed = allowed_heads.get(r, None)
            if allowed is not None and len(allowed) < E:
                # Only allow entities that appeared as heads for this relation in training
                mask = torch.tensor([i in allowed for i in range(E)], device=device)
                scores = torch.where(mask, scores, torch.tensor(float("-inf"), device=device))
        
        # rank for this query (higher=better)
        rank = ranks_from_scores(scores.unsqueeze(0), torch.tensor([h], device=device))
        all_ranks.append(rank)
        
        # Sanity checks
        assert (rank >= 1).all() and (rank <= E).all(), f"Invalid rank: {rank}"

    ranks = torch.cat(all_ranks, dim=0)
    
    return {
        "MRR": float((1.0 / ranks.float()).mean().item()),
        "H@1": float((ranks <= 1).float().mean().item()),
        "H@3": float((ranks <= 3).float().mean().item()),
        "H@10": float((ranks <= 10).float().mean().item()),
        "num_queries": int(ranks.numel())
    }


def evaluate_all(model, dataset, E, device, tails_by_hr, heads_by_rt, hier_rel_ids, etype_tensor=None, allowed_tails=None, allowed_heads=None, eval_filtered=True):
    """
    Unified evaluation function that returns both filtered and type-constrained filtered results.
    
    Returns a dict:
    {
      "filtered": {"tail": {...}, "head": {...}, "average": {...}},
      "tc_filtered": {"tail": {...}, "head": {...}, "average": {...}}
    }
    """
    model.eval()
    
    # Run filtered evaluation (both regular and type-constrained)
    if eval_filtered:
        tail_results_filtered = eval_tail_filtered(model, dataset, E, device, tails_by_hr, hier_rel_ids, etype_tensor, type_constrained=False, allowed_tails=allowed_tails)
        head_results_filtered = eval_head_filtered(model, dataset, E, device, heads_by_rt, hier_rel_ids, etype_tensor, type_constrained=False, allowed_heads=allowed_heads)
        
        tail_results_tc = eval_tail_filtered(model, dataset, E, device, tails_by_hr, hier_rel_ids, etype_tensor, type_constrained=True, allowed_tails=allowed_tails)
        head_results_tc = eval_head_filtered(model, dataset, E, device, heads_by_rt, hier_rel_ids, etype_tensor, type_constrained=True, allowed_heads=allowed_heads)
    else:
        tail_results_filtered = eval_tail_unfiltered(model, dataset, E, device, hier_rel_ids, etype_tensor, type_constrained=False, allowed_tails=allowed_tails)
        head_results_filtered = eval_head_unfiltered(model, dataset, E, device, hier_rel_ids, etype_tensor, type_constrained=False, allowed_heads=allowed_heads)
        
        tail_results_tc = eval_tail_unfiltered(model, dataset, E, device, hier_rel_ids, etype_tensor, type_constrained=True, allowed_tails=allowed_tails)
        head_results_tc = eval_head_unfiltered(model, dataset, E, device, hier_rel_ids, etype_tensor, type_constrained=True, allowed_heads=allowed_heads)
    
    # Calculate averages for filtered results
    avg_hits_1_filtered = (tail_results_filtered["H@1"] + head_results_filtered["H@1"]) / 2
    avg_hits_3_filtered = (tail_results_filtered["H@3"] + head_results_filtered["H@3"]) / 2
    avg_hits_10_filtered = (tail_results_filtered["H@10"] + head_results_filtered["H@10"]) / 2
    avg_mrr_filtered = (tail_results_filtered["MRR"] + head_results_filtered["MRR"]) / 2
    
    # Calculate averages for type-constrained filtered results
    avg_hits_1_tc = (tail_results_tc["H@1"] + head_results_tc["H@1"]) / 2
    avg_hits_3_tc = (tail_results_tc["H@3"] + head_results_tc["H@3"]) / 2
    avg_hits_10_tc = (tail_results_tc["H@10"] + head_results_tc["H@10"]) / 2
    avg_mrr_tc = (tail_results_tc["MRR"] + head_results_tc["MRR"]) / 2
    
    return {
        "filtered": {
            "tail": tail_results_filtered,
            "head": head_results_filtered,
            "average": {
                "MRR": avg_mrr_filtered,
                "H@1": avg_hits_1_filtered,
                "H@3": avg_hits_3_filtered,
                "H@10": avg_hits_10_filtered
            }
        },
        "tc_filtered": {
            "tail": tail_results_tc,
            "head": head_results_tc,
            "average": {
                "MRR": avg_mrr_tc,
                "H@1": avg_hits_1_tc,
                "H@3": avg_hits_3_tc,
                "H@10": avg_hits_10_tc
            }
        }
    }


# Fix for eval_tail_filtered function
def eval_tail_filtered(model, dataset, E, device, tails_by_hr, hier_rel_ids, etype_tensor=None, type_constrained=False, allowed_tails=None):
    """Efficient filtered tail evaluation."""
    model.eval()
    all_ranks = []
    for h, r, t in dataset:              # ints from Dataset
        # candidate mask
        mask = torch.ones(E, dtype=torch.bool, device=device)           # start with all entities
        # filter out other true tails
        true_t = tails_by_hr.get((h, r), set())
        if true_t:
            mask[torch.tensor(list(true_t), device=device)] = False
        # forbid self for hierarchical relations
        if r in hier_rel_ids:
            mask[h] = False
        
        # Type-constrained evaluation: use train-observed tail ranges per relation
        if type_constrained and allowed_tails is not None:
            allowed = allowed_tails.get(r, None)
            if allowed is not None and len(allowed) < E:
                # Only allow entities that appeared as tails for this relation in training
                mask = mask & torch.tensor([i in allowed for i in range(E)], device=device)
        
        # CRITICAL FIX: Always ensure gold is included AFTER type constraint
        mask[t] = True

        # TC baseline debug (only when type_constrained)
        if type_constrained and allowed_tails is not None:
            allowed = allowed_tails.get(r, None)
            if allowed:
                tc_rand_h10 = min(10, len(allowed)) / len(allowed)
                # (log occasionally; e.g., once per 1000 queries)
                if len(all_ranks) % 1000 == 0:
                    print(f"[TC DEBUG] Query {len(all_ranks)}: r={r}, allowed_tails={len(allowed)}, tc_rand_h10={tc_rand_h10:.3f}")

        cand = torch.arange(E, device=device)[mask.to(device)]
        bh = torch.full_like(cand, h, device=device)
        br = torch.full_like(cand, r, device=device)
        trip = torch.stack([bh, br, cand], dim=1)
        scores = model.score_triples(trip[:, 0], trip[:, 1], trip[:, 2])  # scores; higher=better

        # dense vector with -inf elsewhere (since higher scores are better)
        vec = torch.full((E,), float("-inf"), device=device)
        vec[mask.to(device)] = scores
        # sanity: gold must be finite
        assert torch.isfinite(vec[t]).item(), "Gold was filtered out!"

        # rank for this query (higher=better)
        rank = ranks_from_scores(vec.unsqueeze(0), torch.tensor([t], device=device))
        all_ranks.append(rank)
        
        # Sanity checks
        assert (rank >= 1).all() and (rank <= E).all(), f"Invalid rank: {rank}"
        if type_constrained:
            assert mask[t].item(), "Gold not re-included under type constraints"

    ranks = torch.cat(all_ranks, dim=0)
    
    # Evaluation guardrails
    h10_result = float((ranks <= 10).float().mean().item())
    assert h10_result >= 0.0, "H@10 should be non-negative"
    
    # Random baseline sanity check
    avg_cands = sum(len(tails_by_hr.get((h, r), set())) for h, r, t in dataset) / len(dataset)
    if avg_cands > 0:
        rand_h10 = min(10, avg_cands) / avg_cands
        print(f"Tail eval: avg_candidates={avg_cands:.1f}, random_H@10~{rand_h10:.3f}, actual_H@10={h10_result:.3f}")
    
    return {
        "MRR": float((1.0 / ranks.float()).mean().item()),
        "H@1": float((ranks <= 1).float().mean().item()),
        "H@3": float((ranks <= 3).float().mean().item()),
        "H@10": h10_result,
        "num_queries": int(ranks.numel())
    }


@torch.no_grad()
def eval_head_filtered(model, dataset, E, device, heads_by_rt, hier_rel_ids, etype_tensor=None, type_constrained=False, allowed_heads=None):
    """Efficient filtered head evaluation."""
    model.eval()
    all_ranks = []
    for h, r, t in dataset:
        mask = torch.ones(E, dtype=torch.bool, device=device)
        true_h = heads_by_rt.get((r, t), set())
        if true_h:
            mask[torch.tensor(list(true_h), device=device)] = False
        if r in hier_rel_ids:
            mask[t] = False
        
        # Type-constrained evaluation: use train-observed head ranges per relation
        if type_constrained and allowed_heads is not None:
            allowed = allowed_heads.get(r, None)
            if allowed is not None and len(allowed) < E:
                # Only allow entities that appeared as heads for this relation in training
                mask = mask & torch.tensor([i in allowed for i in range(E)], device=device)
        
        # CRITICAL FIX: Always ensure gold is included AFTER type constraint
        mask[h] = True

        # TC baseline debug (only when type_constrained)
        if type_constrained and allowed_heads is not None:
            allowed = allowed_heads.get(r, None)
            if allowed:
                tc_rand_h10 = min(10, len(allowed)) / len(allowed)
                # (log occasionally; e.g., once per 1000 queries)
                if len(all_ranks) % 1000 == 0:
                    print(f"[TC DEBUG] Query {len(all_ranks)}: r={r}, allowed_heads={len(allowed)}, tc_rand_h10={tc_rand_h10:.3f}")

        cand = torch.arange(E, device=device)[mask.to(device)]
        br = torch.full_like(cand, r, device=device)
        bt = torch.full_like(cand, t, device=device)
        trip = torch.stack([cand, br, bt], dim=1)
        scores = model.score_triples(trip[:, 0], trip[:, 1], trip[:, 2])  # scores; higher=better

        vec = torch.full((E,), float("-inf"), device=device)
        vec[mask.to(device)] = scores
        assert torch.isfinite(vec[h]).item(), "Gold was filtered out!"

        rank = ranks_from_scores(vec.unsqueeze(0), torch.tensor([h], device=device))
        all_ranks.append(rank)
        
        # Sanity checks
        assert (rank >= 1).all() and (rank <= E).all(), f"Invalid rank: {rank}"
        if type_constrained:
            assert mask[h].item(), "Gold not re-included under type constraints"

    ranks = torch.cat(all_ranks, dim=0)
    
    # Evaluation guardrails
    h10_result = float((ranks <= 10).float().mean().item())
    assert h10_result >= 0.0, "H@10 should be non-negative"
    
    # Random baseline sanity check
    avg_cands = sum(len(heads_by_rt.get((r, t), set())) for h, r, t in dataset) / len(dataset)
    if avg_cands > 0:
        rand_h10 = min(10, avg_cands) / avg_cands
        print(f"Head eval: avg_candidates={avg_cands:.1f}, random_H@10~{rand_h10:.3f}, actual_H@10={h10_result:.3f}")
    
    return {
        "MRR": float((1.0 / ranks.float()).mean().item()),
        "H@1": float((ranks <= 1).float().mean().item()),
        "H@3": float((ranks <= 3).float().mean().item()),
        "H@10": h10_result,
        "num_queries": int(ranks.numel())
    }






def main(_):
    torch.random.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Core parameters
    batch_size = FLAGS.batch_size
    vector_length = FLAGS.embedding_dim
    margin = FLAGS.margin
    norm = FLAGS.norm
    epochs = FLAGS.epochs
    
    # GPU settings
    use_gpu = not FLAGS.nouse_gpu
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    if use_gpu and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU instead.")
        
    # --- MITRE TXT loading and precompute helpers ---
    from data import load_mitre_txt
    from storage import build_twohop_map

    # 1) Load MITRE TXT (space or tab separated)
    e2id, r2id, (train_triples, valid_triples, test_triples), all_true = \
        load_mitre_txt(FLAGS.dataset_path)

    # Print basic train stats
    print(f"[DATA] Train={len(train_triples)} Valid={len(valid_triples)} Test={len(test_triples)} "
          f"Entities={len(e2id)} Relations={len(r2id)}")

    # Import model factory and build model
    from models import build_model
    
    model = build_model(
        name=FLAGS.model,
        num_entities=len(e2id),
        num_relations=len(r2id),
        dim=FLAGS.embedding_dim,
        margin=FLAGS.margin,
        conve_hidden_channels=FLAGS.conve_hidden_channels,
        conve_dropout=FLAGS.conve_dropout,
        convtranse_kernel_size=FLAGS.convtranse_kernel_size,
        convtranse_channels=FLAGS.convtranse_channels,
        device=('cpu' if FLAGS.nouse_gpu else 'cuda')
    )

    # 1.5) Create inverse maps
    id2entity = [None] * len(e2id)
    for e, i in e2id.items(): 
        id2entity[i] = e
    id2rel = [None] * len(r2id)
    for r, i in r2id.items(): 
        id2rel[i] = r

    # Create dictionary version for categorization
    id2rel_dict = {i: r for r, i in r2id.items()}

    # Use NAME-BASED categorization for reporting (stable & matches your printouts)
    from data import categorize_relations_by_name
    rel_category = categorize_relations_by_name(id2rel_dict)

    # Replace ALL older prints with this single line:
    num_direct = sum(v=='direct' for v in rel_category.values())
    num_hier   = sum(v=='hier'   for v in rel_category.values())
    num_other  = sum(v=='other'  for v in rel_category.values())
    print(f"Relation categories by name: direct={num_direct}, hier={num_hier}, other={num_other}")

    # Create dataset paths for the old MitreDataset class
    path = FLAGS.dataset_path
    train_path = os.path.join(path, "train.txt")
    validation_path = os.path.join(path, "valid.txt")
    test_path = os.path.join(path, "test.txt")
    
    train_set = data.MitreDataset(train_path, e2id, r2id)
    train_generator = torch_data.DataLoader(
        train_set, 
        batch_size=batch_size,
        num_workers=4 if not FLAGS.nouse_gpu else 0,
        pin_memory=not FLAGS.nouse_gpu
    )
    validation_set = data.MitreDataset(validation_path, e2id, r2id)
    test_set = data.MitreDataset(test_path, e2id, r2id)
    
    # Compute Bernoulli negative sampling statistics
    print("Computing Bernoulli negative sampling statistics...")
    train_triples_str = []
    for h, r, t in train_set:
        h_val = h.item() if hasattr(h, 'item') else h
        r_val = r.item() if hasattr(r, 'item') else r
        t_val = t.item() if hasattr(t, 'item') else t
        h_str = id2entity[h_val]
        r_str = id2rel[r_val] 
        t_str = id2entity[t_val]
        train_triples_str.append((h_str, r_str, t_str))
    
    p_tail = bernoulli_p(train_triples_str)  # string rel -> float in [0,1]
    p_tail_id = torch.zeros(len(r2id), device=device)
    for rel, pid in r2id.items():
        base = rel[:-4] if rel.endswith("_rev") else rel
        p_tail_id[pid] = p_tail.get(base, 0.5)
    print(f"Computed Bernoulli corruption probabilities for {len(p_tail)} relations")
    
    # Build filtered sets in ID space
    print("Building filtered evaluation sets...")
    tails_by_hr, heads_by_rt = build_known_sets_id([train_path, validation_path, test_path], e2id, r2id)
    print(f"Built filtered sets: {len(tails_by_hr)} head-rel pairs, {len(heads_by_rt)} rel-tail pairs")
    
    # Entity type detection for type-constrained evaluation
    print("Building entity type mappings...")
    etype_tensor = torch.zeros(len(e2id), dtype=torch.long, device=device)  # 0=CWE, 1=CAPEC
    for entity, eid in e2id.items():
        if entity.startswith("CWE-"):
            etype_tensor[eid] = 0  # CWE type
        elif entity.startswith("CAPEC-"):
            etype_tensor[eid] = 1  # CAPEC type
        else:
            etype_tensor[eid] = 0  # default to CWE type
    
    cwe_count = (etype_tensor == 0).sum().item()
    capec_count = (etype_tensor == 1).sum().item()
    print(f"Entity types: {cwe_count} CWE entities, {capec_count} CAPEC entities")
    
    # Create entity type mapping for relation categorization
    e2type = {}
    for entity, eid in e2id.items():
        if entity.startswith("CWE-"):
            e2type[eid] = "CWE"
        elif entity.startswith("CAPEC-"):
            e2type[eid] = "CAPEC"
        else:
            e2type[eid] = "CWE"  # default to CWE type
    
    # Note: Using name-based categorization instead of data-driven categorization
    
    # Build allowed heads/tails for type-constrained evaluation
    from collections import defaultdict
    allowed_heads = defaultdict(set)
    allowed_tails = defaultdict(set)
    for (h, r, t) in train_triples:
        allowed_heads[r].add(h)
        allowed_tails[r].add(t)
    print(f"Built type constraints: {len(allowed_heads)} relations with head constraints, {len(allowed_tails)} relations with tail constraints")
    
    # Print allowed-tail sizes per relation (explains easy H@10)
    sizes = {rid: len(s) for rid, s in allowed_tails.items()}
    def _stats(msk): 
        arr = [sizes[rid] for rid, cat in rel_category.items() if msk(cat) and rid in sizes]
        if not arr: return "n/a"
        import numpy as np; arr = np.array(arr)
        return f"min={arr.min()} med={np.median(arr):.1f} mean={arr.mean():.1f} max={arr.max()}"
    print("[Allowed tail sizes] direct:", _stats(lambda c: c=='direct'),
          "| hier:", _stats(lambda c: c=='hier'),
          "| other:", _stats(lambda c: c=='other'))
    
    # Drive masks from the same map (no string checks)
    HIER_REL_IDS = {rid for rid, cat in rel_category.items() if cat == 'hier'}
    DIRECT_REL_IDS = torch.tensor([rid for rid, cat in rel_category.items() if cat == 'direct'], device=device)
    PATH_REL_IDS   = torch.tensor([], device=device)  # not used for reporting anymore
    
    # Build scoped 2-hop map with gating
    ignore_names = set(FLAGS.hier_relations)                 # existing
    ignore_names |= {name for name in id2rel_dict.values() if name.startswith('PATH_')}  # add
    
    from storage import build_twohop_map
    twohop_map = build_twohop_map(train_triples,
                                  rel_category=rel_category,
                                  ignore_rel_names=ignore_names,
                                  id2rel=id2rel_dict,
                                  scope=FLAGS.twohop_scope)
    
    # Verify scoped twohop_map is working
    num_keys = len(twohop_map)
    avg_twohop = (sum(len(v) for v in twohop_map.values()) / max(1, num_keys)) if num_keys else 0.0
    print(f"[2HOP] scoped keys={num_keys}, avg_twohop={avg_twohop:.2f}, scope={FLAGS.twohop_scope}")
    print(f"[2HOP] Config: multi_hop_aware={FLAGS.multi_hop_aware}, weight={FLAGS.twohop_weight}, margin={FLAGS.twohop_margin}")

    # Import and call the training function
    from train import train_and_eval
    
    train_and_eval(
        model, FLAGS,
        e2id, r2id,
        train_triples, valid_triples, test_triples,
        train_set, validation_set, test_set,
        tails_by_hr, heads_by_rt,
        rel_category, HIER_REL_IDS,
        allowed_heads, allowed_tails,
        etype_tensor,
        twohop_map,
        p_tail_id,
        save_ckpt=save_ckpt,
        load_ckpt=load_ckpt,
        evaluate_all=evaluate_all,
    )




if __name__ == '__main__':
    app.run(main)
