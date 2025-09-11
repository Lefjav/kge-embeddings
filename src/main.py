from absl import app
from absl import flags
import data
import model as model_definition
import os
import storage
import torch
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils import tensorboard
from collections import defaultdict

FLAGS = flags.FLAGS
flags.DEFINE_float("lr", default=0.01, help="Learning rate value.")
flags.DEFINE_integer("seed", default=1234, help="Seed value.")
flags.DEFINE_integer("batch_size", default=256, help="Maximum batch size (increased for faster training).")
flags.DEFINE_integer("validation_batch_size", default=64, help="Maximum batch size during model validation.")
flags.DEFINE_integer("vector_length", default=50, help="Length of entity/relation vector.")
flags.DEFINE_float("margin", default=1.0, help="Margin value in margin-based ranking loss.")
flags.DEFINE_integer("norm", default=1, help="Norm used for calculating dissimilarity metric (usually 1 or 2).")
flags.DEFINE_integer("epochs", default=500, help="Number of training epochs (reduced for faster training).")
flags.DEFINE_string("dataset_path", default="./mitre-data/txt",
                    help="Path to TXT splits: train.txt/valid.txt/test.txt")
flags.DEFINE_bool("multi_hop_aware", default=True, help="Use multi-hop aware training techniques.")
flags.DEFINE_float("direct_relation_weight", default=2.0, help="Weight boost for direct CWE-CAPEC relationships.")
flags.DEFINE_bool("relation_aware_negatives", default=True, help="Use relation-aware negative sampling (now optimized).")
flags.DEFINE_bool("use_gpu", default=True, help="Flag enabling gpu usage.")
flags.DEFINE_integer("validation_freq", default=10, help="Validate model every X epochs.")
flags.DEFINE_string("checkpoint_path", default="", help="Path to model checkpoint (by default train from scratch).")
flags.DEFINE_string("tensorboard_log_dir", default="./runs", help="Path for tensorboard log directory.")
flags.DEFINE_integer("num_negs", default=256, help="Number of negatives per positive.")
flags.DEFINE_bool("self_adversarial", default=True, help="Use self-adversarial weighting for negatives.")
flags.DEFINE_string("model", default="TransE", help="Model to use: TransE, ComplEx, or RotatE")

# add eval & calibration toggles
flags.DEFINE_bool("eval_filtered", default=True, help="Use filtered ranking metrics")
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


def ranks_from_scores(scores, gold, smaller_is_better=True):
    """Calculate ranks from scores using tie-safe counting method."""
    gold_scores = scores[torch.arange(scores.size(0), device=scores.device), gold]
    if smaller_is_better:
        ranks = 1 + (scores < gold_scores.unsqueeze(1)).sum(dim=1)
    else:
        ranks = 1 + (scores > gold_scores.unsqueeze(1)).sum(dim=1)
    return ranks


def hits_at_k_from_ranks(ranks, k): 
    """Calculate hits@k from ranks."""
    return (ranks <= k).float().mean().item()


def mrr_from_ranks(ranks): 
    """Calculate MRR from ranks."""
    return (1.0 / ranks.float()).mean().item()


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

        cand = torch.arange(E, device=device)[mask.to(device)]
        bh = torch.full_like(cand, h, device=device)
        br = torch.full_like(cand, r, device=device)
        trip = torch.stack([bh, br, cand], dim=1)
        d = model.predict(trip)                               # distances; smaller=better

        # dense vector with +inf elsewhere
        vec = torch.full((E,), float("inf"), device=device)
        vec[mask.to(device)] = d
        # sanity: gold must be finite
        assert torch.isfinite(vec[t]).item(), "Gold was filtered out!"

        # rank for this query
        rank = ranks_from_scores(vec.unsqueeze(0), torch.tensor([t], device=device), smaller_is_better=True)
        all_ranks.append(rank)

    ranks = torch.cat(all_ranks, dim=0)
    
    # Evaluation guardrails
    h10_result = hits_at_k_from_ranks(ranks, 10)
    assert h10_result >= 0.0, "H@10 should be non-negative"
    
    # Random baseline sanity check
    avg_cands = sum(len(tails_by_hr.get((h, r), set())) for h, r, t in dataset) / len(dataset)
    if avg_cands > 0:
        rand_h10 = min(10, avg_cands) / avg_cands
        print(f"Tail eval: avg_candidates={avg_cands:.1f}, random_H@10~{rand_h10:.3f}, actual_H@10={h10_result:.3f}")
    
    return {
        "MRR": mrr_from_ranks(ranks),
        "H@1": hits_at_k_from_ranks(ranks, 1),
        "H@3": hits_at_k_from_ranks(ranks, 3),
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

        cand = torch.arange(E, device=device)[mask.to(device)]
        br = torch.full_like(cand, r, device=device)
        bt = torch.full_like(cand, t, device=device)
        trip = torch.stack([cand, br, bt], dim=1)
        d = model.predict(trip)

        vec = torch.full((E,), float("inf"), device=device)
        vec[mask.to(device)] = d
        assert torch.isfinite(vec[h]).item(), "Gold was filtered out!"

        rank = ranks_from_scores(vec.unsqueeze(0), torch.tensor([h], device=device), smaller_is_better=True)
        all_ranks.append(rank)

    ranks = torch.cat(all_ranks, dim=0)
    
    # Evaluation guardrails
    h10_result = hits_at_k_from_ranks(ranks, 10)
    assert h10_result >= 0.0, "H@10 should be non-negative"
    
    # Random baseline sanity check
    avg_cands = sum(len(heads_by_rt.get((r, t), set())) for h, r, t in dataset) / len(dataset)
    if avg_cands > 0:
        rand_h10 = min(10, avg_cands) / avg_cands
        print(f"Head eval: avg_candidates={avg_cands:.1f}, random_H@10~{rand_h10:.3f}, actual_H@10={h10_result:.3f}")
    
    return {
        "MRR": mrr_from_ranks(ranks),
        "H@1": hits_at_k_from_ranks(ranks, 1),
        "H@3": hits_at_k_from_ranks(ranks, 3),
        "H@10": h10_result,
        "num_queries": int(ranks.numel())
    }




def sample_negatives_hr(h, r, t, all_entities, pos_set, twohop_map, k):
    """Sample k negative triples with 50/50 head-vs-tail corruption and 2-hop awareness.
    
    This balanced approach helps both head and tail prediction performance, addressing
    the common issue where head metrics lag due to tail-only corruption during training.
    """
    import random
    keep_twohop = FLAGS.twohop_keep_ratio  # e.g., 0.2
    direct_scope = (FLAGS.twohop_scope != "none")  # quick check
    negs = []
    while len(negs) < k:
        if random.random() < 0.5:        # Bernoulli already computed? then use its p[r] if available
            # tail corruption
            t_neg = random.choice(all_entities)
            if (h, r, t_neg) in pos_set: 
                continue
            if direct_scope and t_neg in twohop_map.get((h, r), ()):
                if random.random() > keep_twohop:
                    continue
            negs.append((h, r, t_neg))
        else:
            # head corruption
            h_neg = random.choice(all_entities)
            if (h_neg, r, t) in pos_set:
                continue
            negs.append((h_neg, r, t))
    return negs


def generate_relation_aware_negatives(pos, entity2id, relation2id, id2rel, device, num_negs, p_tail_id, twohop_map=None, all_true=None):
    """Generate negatives that challenge direct vs transitive reasoning - with 2-hop awareness."""
    B = pos.size(0)
    
    # If we have 2-hop map and all_true set, use the new balanced sampling
    if FLAGS.multi_hop_aware and twohop_map is not None and all_true is not None:
        all_entities = list(range(len(entity2id)))
        neg_list = []
        
        for i in range(B):
            h_id = pos[i, 0].item()
            r_id = pos[i, 1].item()
            t_id = pos[i, 2].item()
            
            # Sample negatives with 50/50 head-vs-tail corruption and 2-hop awareness
            negs = sample_negatives_hr(h_id, r_id, t_id, all_entities, all_true, twohop_map, num_negs)
            neg_list.extend(negs)
        
        # Convert to tensor format
        neg_tensor = torch.tensor(neg_list, device=device)
        return neg_tensor
    
    # Fallback to standard Bernoulli corruption
    rep = pos.repeat_interleave(num_negs, dim=0)  # [B*num_negs, 3]
    r_flat = rep[:, 1]
    p = p_tail_id[r_flat]
    coin = (torch.rand_like(p) > p).long()
    rand_e = torch.randint(len(entity2id), (B*num_negs,), device=device)
    rep[:, 0] = torch.where(coin==1, rand_e, rep[:, 0])
    rep[:, 2] = torch.where(coin==0, rand_e, rep[:, 2])
    
    return rep


# Removed generate_challenging_negatives - too slow, using standard corruption instead


def apply_multi_hop_aware_weighting(loss, pos, direct_rel_ids, path_rel_ids, direct_weight):
    """Apply higher weights to direct relationship losses - OPTIMIZED."""
    weights = torch.ones_like(loss)
    
    # Vectorized operations instead of Python loops
    r_ids = pos[:, 1]
    
    # Create boolean masks for different relation types
    direct_mask = torch.isin(r_ids, direct_rel_ids)
    path_mask = torch.isin(r_ids, path_rel_ids)
    
    # Apply weights vectorized
    weights[direct_mask] = direct_weight
    weights[path_mask] = 0.5
    
    return loss * weights


def main(_):
    torch.random.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = FLAGS.batch_size
    vector_length = FLAGS.vector_length
    margin = FLAGS.margin
    norm = FLAGS.norm
    epochs = FLAGS.epochs
    device = torch.device('cuda' if FLAGS.use_gpu and torch.cuda.is_available() else 'cpu')
    if FLAGS.use_gpu and not torch.cuda.is_available():
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

    # 1.5) Create inverse maps
    id2entity = [None] * len(e2id)
    for e, i in e2id.items(): 
        id2entity[i] = e
    id2rel = [None] * len(r2id)
    for r, i in r2id.items(): 
        id2rel[i] = r


    # Create dataset paths for the old MitreDataset class
    path = FLAGS.dataset_path
    train_path = os.path.join(path, "train.txt")
    validation_path = os.path.join(path, "valid.txt")
    test_path = os.path.join(path, "test.txt")
    
    train_set = data.MitreDataset(train_path, e2id, r2id)
    train_generator = torch_data.DataLoader(train_set, batch_size=batch_size)
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
    
    # Categorize relations
    from data import categorize_relations
    rel_category = categorize_relations(train_triples, e2type, r2id)
    
    # Count relations by category
    direct_count = sum(1 for cat in rel_category.values() if cat == "direct")
    hier_count = sum(1 for cat in rel_category.values() if cat == "hier")
    path_count = sum(1 for cat in rel_category.values() if cat == "path")
    print(f"Relation categories: {direct_count} direct, {hier_count} hierarchical, {path_count} path")
    
    # Build allowed heads/tails for type-constrained evaluation
    from collections import defaultdict
    allowed_heads = defaultdict(set)
    allowed_tails = defaultdict(set)
    for (h, r, t) in train_triples:
        allowed_heads[r].add(h)
        allowed_tails[r].add(t)
    print(f"Built type constraints: {len(allowed_heads)} relations with head constraints, {len(allowed_tails)} relations with tail constraints")
    
    # Schema-aware hierarchical relations and CWE-CAPEC specific relations
    HIER_NAMES = {"is_child_of", "has_child", "is_parent_of", "has_parent", "parent_of", "child_of", "subclass_of", "superclass_of", "related_to"}
    CWE_CAPEC_NAMES = {"has_attack_pattern", "exploits_weakness", "related_to"}
    ALL_REL_NAMES = HIER_NAMES | CWE_CAPEC_NAMES
    HIER_REL_IDS = {r2id[r] for r in r2id if r in ALL_REL_NAMES}
    
    # Precompute relation ID sets for fast multi-hop aware weighting using categorized relations
    DIRECT_REL_IDS = torch.tensor([r_id for r_id, cat in rel_category.items() if cat == "direct"], device=device)
    PATH_REL_IDS = torch.tensor([r_id for r_id, cat in rel_category.items() if cat == "path"], device=device)
    HIER_REL_IDS_CAT = {r_id for r_id, cat in rel_category.items() if cat == "hier"}
    
    print(f"Found {len(HIER_REL_IDS)} hierarchical relations (name-based): {[id2rel[r_id] for r_id in HIER_REL_IDS]}")
    print(f"Found {len(HIER_REL_IDS_CAT)} hierarchical relations (category-based): {[id2rel[r_id] for r_id in HIER_REL_IDS_CAT]}")
    print(f"Found {len(DIRECT_REL_IDS)} direct relations, {len(PATH_REL_IDS)} path relations")
    
    # Build scoped 2-hop map with gating
    ignore_names = set(FLAGS.hier_relations)
    id2rel_dict = {v: k for k, v in r2id.items()}
    
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

    if FLAGS.model == "TransE":
        model = model_definition.TransE(entity_count=len(e2id), relation_count=len(r2id), 
                                   dim=vector_length, margin=margin, device=device, norm=norm, use_soft_loss=True)
    elif FLAGS.model == "ComplEx":
        model = model_definition.ComplEx(entity_count=len(e2id), relation_count=len(r2id), 
                                     dim=vector_length, margin=margin, device=device, use_soft_loss=True)
    elif FLAGS.model == "RotatE":
        model = model_definition.RotatE(entity_count=len(e2id), relation_count=len(r2id), 
                                    dim=vector_length, margin=margin, device=device, use_soft_loss=True)
    else:
        raise ValueError(f"Unknown model: {FLAGS.model}")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=1e-4)

    # 3) Seed + AMP scaler
    import random, numpy as np
    torch.manual_seed(FLAGS.seed); random.seed(FLAGS.seed); np.random.seed(FLAGS.seed)
    scaler = torch.cuda.amp.GradScaler(enabled=(FLAGS.amp and FLAGS.use_gpu and torch.cuda.is_available()))
    device = next(model.parameters()).device

    summary_writer = tensorboard.SummaryWriter(log_dir=FLAGS.tensorboard_log_dir)
    start_epoch_id = 1
    step = 0
    best_score = 0.0

    if FLAGS.checkpoint_path:
        start_epoch_id, step, best_score = storage.load_checkpoint(FLAGS.checkpoint_path, model, optimizer)

    print(model)

    # Training loop
    for epoch_id in range(start_epoch_id, epochs + 1):
        print("Starting epoch: ", epoch_id)
        loss_impacting_samples_count = 0
        samples_count = 0
        model.train()
        num_negs = FLAGS.num_negs
        
        for h, r, t in train_generator:
            h, r, t = h.to(device), r.to(device), t.to(device)
            pos = torch.stack([h, r, t], dim=1)                        # [B, 3]
            B = h.size(0)  # Define B here so it's always available

            # Multi-hop aware negative sampling
            if FLAGS.relation_aware_negatives:
                neg = generate_relation_aware_negatives(pos, e2id, r2id, id2rel, device, num_negs, p_tail_id, twohop_map, all_true)
            else:
                # Standard Bernoulli corruption
                rep = pos.repeat_interleave(num_negs, dim=0)               # [B*num_negs, 3]
                r_flat = rep[:, 1]                                         # relation IDs for each negative
                p = p_tail_id[r_flat]                                      # corruption probabilities
                coin = (torch.rand_like(p) > p).long()                     # 0: corrupt tail with prob p, else head
                rand_e = torch.randint(len(e2id), (B*num_negs,), device=device)
                rep[:, 0] = torch.where(coin==1, rand_e, rep[:, 0])        # corrupt head
                rep[:, 2] = torch.where(coin==0, rand_e, rep[:, 2])        # corrupt tail
                neg = rep

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(FLAGS.amp and FLAGS.use_gpu and torch.cuda.is_available())):
                # your existing positive/negative loss (e.g., margin-based ranking)
                loss, pd, nd = model(pos, neg)                             # per-sample loss
                
                # Multi-hop aware loss weighting
                if FLAGS.multi_hop_aware:
                    loss = apply_multi_hop_aware_weighting(loss, pos, DIRECT_REL_IDS, PATH_REL_IDS, FLAGS.direct_relation_weight)
                
                loss = loss.mean()
                
                # OPTIONAL: add inline direct>2hop margin if enabled
                if FLAGS.multi_hop_aware and FLAGS.twohop_weight > 0.0 and FLAGS.twohop_scope != "none":
                    # 'batch_pos' is a Python list of (h,r,t) IDs from this mini-batch
                    batch_pos = [(h[i].item(), r[i].item(), t[i].item()) for i in range(B)]
                    margin_terms = []
                    for (h_id, r_id, t_id) in batch_pos:
                        # only apply to 'direct' relations
                        if rel_category.get(r_id) != "direct":
                            continue
                        t2_list = list(twohop_map.get((h_id, r_id), []))[:2]  # small sample per pos
                        if not t2_list: 
                            continue
                        H = torch.tensor([h_id]*len(t2_list), device=device)
                        R = torch.tensor([r_id]*len(t2_list), device=device)
                        T_pos = torch.tensor([t_id]*len(t2_list), device=device)
                        T2 = torch.tensor(t2_list, device=device)
                        s_pos = model.score_triple(H, R, T_pos)   # you expose this in model.py (see Patch 4)
                        s_neg = model.score_triple(H, R, T2)
                        margin_terms.append(torch.relu(s_neg - s_pos + FLAGS.twohop_margin))
                    if margin_terms:
                        twohop_loss = FLAGS.twohop_weight * torch.cat(margin_terms).mean()
                        loss = loss + twohop_loss
                        # Debug: log 2-hop loss occasionally
                        if step % 100 == 0:
                            print(f"[2HOP] Step {step}: margin_terms={len(margin_terms)}, twohop_loss={twohop_loss.item():.4f}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            
            # Update tracking variables
            loss_impacting_samples_count += 1 if loss.item() > 0 else 0
            samples_count += B
            
            # Log training metrics
            summary_writer.add_scalar('Loss/train', loss.data.cpu().numpy(), global_step=step)
            summary_writer.add_scalar('Distance/positive', pd.mean().data.cpu().numpy(), global_step=step)
            summary_writer.add_scalar('Distance/negative', nd.mean().data.cpu().numpy(), global_step=step)

        # Log epoch-level metrics
        if samples_count > 0:
            summary_writer.add_scalar('Metrics/loss_impacting_samples', loss_impacting_samples_count / samples_count * 100,
                                      global_step=epoch_id)

        # Initialize score to avoid UnboundLocalError
        score = 0.0
        
        if epoch_id % FLAGS.validation_freq == 0:
            model.eval()
            
            # Score separation diagnostic: check if model is learning to distinguish positives from negatives
            # Sample 200 positives and 200 uniform negatives to inspect score gap
            import numpy as np, random
            device = next(model.parameters()).device
            sample_pos = random.sample(valid_triples, min(200, len(valid_triples)))
            pos_scores = []
            neg_scores = []
            E = len(e2id)
            true_set = set(train_triples) | set(valid_triples) | set(test_triples)

            for (h,r,t) in sample_pos:
                pos_scores.append(model.score_triple(torch.tensor([h],device=device),
                                                     torch.tensor([r],device=device),
                                                     torch.tensor([t],device=device)).item())
                # uniform tail neg not in true_set
                while True:
                    tneg = random.randrange(E)
                    if (h,r,tneg) not in true_set: break
                neg_scores.append(model.score_triple(torch.tensor([h],device=device),
                                                     torch.tensor([r],device=device),
                                                     torch.tensor([tneg],device=device)).item())
            print(f"[VAL] pos_mean={np.mean(pos_scores):.3f} neg_mean={np.mean(neg_scores):.3f} "
                  f"sep={np.mean(pos_scores)-np.mean(neg_scores):.3f}")
            
            # Use efficient filtered evaluation (both regular and type-constrained)
            print("\n--- Regular Filtered Evaluation ---")
            tail_results = eval_tail_filtered(model, validation_set, len(e2id), device, tails_by_hr, HIER_REL_IDS, etype_tensor, type_constrained=False, allowed_tails=allowed_tails)
            head_results = eval_head_filtered(model, validation_set, len(e2id), device, heads_by_rt, HIER_REL_IDS, etype_tensor, type_constrained=False, allowed_heads=allowed_heads)
            
            print("\n--- Type-Constrained Filtered Evaluation ---")
            tail_results_tc = eval_tail_filtered(model, validation_set, len(e2id), device, tails_by_hr, HIER_REL_IDS, etype_tensor, type_constrained=FLAGS.eval_type_constrained, allowed_tails=allowed_tails)
            head_results_tc = eval_head_filtered(model, validation_set, len(e2id), device, heads_by_rt, HIER_REL_IDS, etype_tensor, type_constrained=FLAGS.eval_type_constrained, allowed_heads=allowed_heads)
            
            # Average the results (use type-constrained for main metrics)
            avg_hits_10 = (tail_results_tc["H@10"] + head_results_tc["H@10"]) / 2
            avg_mrr = (tail_results_tc["MRR"] + head_results_tc["MRR"]) / 2
            
            print(f"Epoch {epoch_id} - Regular - Tail: H@1={tail_results['H@1']:.3f}, H@10={tail_results['H@10']:.3f}, MRR={tail_results['MRR']:.3f}")
            print(f"Epoch {epoch_id} - Regular - Head: H@1={head_results['H@1']:.3f}, H@10={head_results['H@10']:.3f}, MRR={head_results['MRR']:.3f}")
            print(f"Epoch {epoch_id} - Type-Constrained - Tail: H@1={tail_results_tc['H@1']:.3f}, H@10={tail_results_tc['H@10']:.3f}, MRR={tail_results_tc['MRR']:.3f}")
            print(f"Epoch {epoch_id} - Type-Constrained - Head: H@1={head_results_tc['H@1']:.3f}, H@10={head_results_tc['H@10']:.3f}, MRR={head_results_tc['MRR']:.3f}")
            print(f"Epoch {epoch_id} - Type-Constrained Avg: H@1={(tail_results_tc['H@1'] + head_results_tc['H@1'])/2:.3f}, H@10={avg_hits_10:.3f}, MRR={avg_mrr:.3f}")
            
            score = avg_hits_10
        
        if score > best_score:
            best_score = score
            checkpoint_path = f"checkpoint_{FLAGS.model}.tar"
            storage.save_checkpoint(model, optimizer, epoch_id, step, best_score, checkpoint_path)

    # Testing the best checkpoint on test dataset
    checkpoint_path = f"checkpoint_{FLAGS.model}.tar"
    storage.load_checkpoint(checkpoint_path, model, optimizer)
    best_model = model.to(device)
    best_model.eval()
    
    # Use efficient filtered evaluation on test set (both regular and type-constrained)
    print("=== FINAL TEST RESULTS ===")
    print("\n--- Regular Filtered Test Results ---")
    tail_results = eval_tail_filtered(best_model, test_set, len(e2id), device, tails_by_hr, HIER_REL_IDS, etype_tensor, type_constrained=False, allowed_tails=allowed_tails)
    head_results = eval_head_filtered(best_model, test_set, len(e2id), device, heads_by_rt, HIER_REL_IDS, etype_tensor, type_constrained=False, allowed_heads=allowed_heads)
    
    print(f"Regular - Tail: H@1={tail_results['H@1']:.3f}, H@3={tail_results['H@3']:.3f}, H@10={tail_results['H@10']:.3f}, MRR={tail_results['MRR']:.3f}")
    print(f"Regular - Head: H@1={head_results['H@1']:.3f}, H@3={head_results['H@3']:.3f}, H@10={head_results['H@10']:.3f}, MRR={head_results['MRR']:.3f}")
    
    avg_hits_1 = (tail_results["H@1"] + head_results["H@1"]) / 2
    avg_hits_3 = (tail_results["H@3"] + head_results["H@3"]) / 2
    avg_hits_10 = (tail_results["H@10"] + head_results["H@10"]) / 2
    avg_mrr = (tail_results["MRR"] + head_results["MRR"]) / 2
    print(f"Regular Average: H@1={avg_hits_1:.3f}, H@3={avg_hits_3:.3f}, H@10={avg_hits_10:.3f}, MRR={avg_mrr:.3f}")
    
    print("\n--- Type-Constrained Filtered Test Results ---")
    tail_results_tc = eval_tail_filtered(best_model, test_set, len(e2id), device, tails_by_hr, HIER_REL_IDS, etype_tensor, type_constrained=FLAGS.eval_type_constrained, allowed_tails=allowed_tails)
    head_results_tc = eval_head_filtered(best_model, test_set, len(e2id), device, heads_by_rt, HIER_REL_IDS, etype_tensor, type_constrained=FLAGS.eval_type_constrained, allowed_heads=allowed_heads)
    
    print(f"Type-Constrained - Tail: H@1={tail_results_tc['H@1']:.3f}, H@3={tail_results_tc['H@3']:.3f}, H@10={tail_results_tc['H@10']:.3f}, MRR={tail_results_tc['MRR']:.3f}")
    print(f"Type-Constrained - Head: H@1={head_results_tc['H@1']:.3f}, H@3={head_results_tc['H@3']:.3f}, H@10={head_results_tc['H@10']:.3f}, MRR={head_results_tc['MRR']:.3f}")
    
    avg_hits_1_tc = (tail_results_tc["H@1"] + head_results_tc["H@1"]) / 2
    avg_hits_3_tc = (tail_results_tc["H@3"] + head_results_tc["H@3"]) / 2
    avg_hits_10_tc = (tail_results_tc["H@10"] + head_results_tc["H@10"]) / 2
    avg_mrr_tc = (tail_results_tc["MRR"] + head_results_tc["MRR"]) / 2
    print(f"Type-Constrained Average: H@1={avg_hits_1_tc:.3f}, H@3={avg_hits_3_tc:.3f}, H@10={avg_hits_10_tc:.3f}, MRR={avg_mrr_tc:.3f}")
    
    # === NEW RANKING EVALUATION WITH CALIBRATION ===
    print("\n=== COMPREHENSIVE RANKING EVALUATION ===")
    from metric import evaluate_ranking, fit_calibrator
    from inference import rank_topk, predict_binary
    
    metrics = evaluate_ranking(model, test_triples, all_true=all_true,
                               e2id=e2id, r2id=r2id,
                               filtered=FLAGS.eval_filtered,
                               per_relation=FLAGS.eval_per_relation,
                               train_triples=train_triples,
                               type_constrained=FLAGS.eval_type_constrained)
    
    # Print per-category results if available
    if "per_category" in metrics:
        print("\n--- Per-Category Results ---")
        for category, results in metrics["per_category"].items():
            print(f"{category}: MRR={results['MRR']:.3f}, H@1={results['Hits@1']:.3f}, "
                  f"H@3={results['Hits@3']:.3f}, H@10={results['Hits@10']:.3f}, count={results['count']}")
    
    if FLAGS.calibration != "none":
        print(f"Fitting {FLAGS.calibration} calibrator on validation set...")
        calibrator = fit_calibrator(model, valid_triples, method=FLAGS.calibration)
        print("Calibrator fitted successfully.")
    else:
        calibrator = None
        print("No calibration requested.")
    
    # Example quick checks (remove in prod):
    # print(rank_topk(model, "cwe:79", "RelatedWeakness", e2id, r2id, k=5,
    #                 filtered=FLAGS.eval_filtered, all_true=all_true))
    # print(predict_binary(model, ("cwe:79","RelatedWeakness","capec:63"), e2id, r2id, calibrator))
    
    # Store results in a text file
    store_results_to_file(
        regular_tail=tail_results,
        regular_head=head_results,
        tc_tail=tail_results_tc,
        tc_head=head_results_tc,
        dataset_path=FLAGS.dataset_path,
        model_params={
            'vector_length': FLAGS.vector_length,
            'margin': FLAGS.margin,
            'norm': FLAGS.norm,
            'lr': FLAGS.lr,
            'epochs': FLAGS.epochs,
            'batch_size': FLAGS.batch_size,
            'num_negs': FLAGS.num_negs,
            'self_adversarial': FLAGS.self_adversarial
        },
        model_name=FLAGS.model  # Add model name
    )


def store_results_to_file(regular_tail, regular_head, tc_tail, tc_head, dataset_path, model_params, model_name):
    """Store evaluation results to a text file with timestamp and model parameters."""
    import datetime
    
    # Create results filename with timestamp and model name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results_{model_name}_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{model_name} Model Evaluation Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Model: {model_name}\n")
        f.write("\n")
        
        # Model parameters
        f.write("Model Parameters:\n")
        f.write("-" * 40 + "\n")
        f.write(f"model: {model_name}\n")
        for param, value in model_params.items():
            f.write(f"{param}: {value}\n")
        f.write("\n")
        
        # Regular filtered evaluation results
        f.write("Regular Filtered Evaluation Results:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Tail Prediction:\n")
        f.write(f"  H@1:  {regular_tail['H@1']:.4f}\n")
        f.write(f"  H@3:  {regular_tail['H@3']:.4f}\n")
        f.write(f"  H@10: {regular_tail['H@10']:.4f}\n")
        f.write(f"  MRR:  {regular_tail['MRR']:.4f}\n")
        f.write(f"  Queries: {regular_tail['num_queries']}\n")
        f.write(f"\nHead Prediction:\n")
        f.write(f"  H@1:  {regular_head['H@1']:.4f}\n")
        f.write(f"  H@3:  {regular_head['H@3']:.4f}\n")
        f.write(f"  H@10: {regular_head['H@10']:.4f}\n")
        f.write(f"  MRR:  {regular_head['MRR']:.4f}\n")
        f.write(f"  Queries: {regular_head['num_queries']}\n")
        
        # Regular averages
        avg_hits_1 = (regular_tail["H@1"] + regular_head["H@1"]) / 2
        avg_hits_3 = (regular_tail["H@3"] + regular_head["H@3"]) / 2
        avg_hits_10 = (regular_tail["H@10"] + regular_head["H@10"]) / 2
        avg_mrr = (regular_tail["MRR"] + regular_head["MRR"]) / 2
        f.write(f"\nRegular Average:\n")
        f.write(f"  H@1:  {avg_hits_1:.4f}\n")
        f.write(f"  H@3:  {avg_hits_3:.4f}\n")
        f.write(f"  H@10: {avg_hits_10:.4f}\n")
        f.write(f"  MRR:  {avg_mrr:.4f}\n")
        f.write("\n")
        
        # Type-constrained evaluation results
        f.write("Type-Constrained Filtered Evaluation Results:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Tail Prediction:\n")
        f.write(f"  H@1:  {tc_tail['H@1']:.4f}\n")
        f.write(f"  H@3:  {tc_tail['H@3']:.4f}\n")
        f.write(f"  H@10: {tc_tail['H@10']:.4f}\n")
        f.write(f"  MRR:  {tc_tail['MRR']:.4f}\n")
        f.write(f"  Queries: {tc_tail['num_queries']}\n")
        f.write(f"\nHead Prediction:\n")
        f.write(f"  H@1:  {tc_head['H@1']:.4f}\n")
        f.write(f"  H@3:  {tc_head['H@3']:.4f}\n")
        f.write(f"  H@10: {tc_head['H@10']:.4f}\n")
        f.write(f"  MRR:  {tc_head['MRR']:.4f}\n")
        f.write(f"  Queries: {tc_head['num_queries']}\n")
        
        # Type-constrained averages
        avg_hits_1_tc = (tc_tail["H@1"] + tc_head["H@1"]) / 2
        avg_hits_3_tc = (tc_tail["H@3"] + tc_head["H@3"]) / 2
        avg_hits_10_tc = (tc_tail["H@10"] + tc_head["H@10"]) / 2
        avg_mrr_tc = (tc_tail["MRR"] + tc_head["MRR"]) / 2
        f.write(f"\nType-Constrained Average:\n")
        f.write(f"  H@1:  {avg_hits_1_tc:.4f}\n")
        f.write(f"  H@3:  {avg_hits_3_tc:.4f}\n")
        f.write(f"  H@10: {avg_hits_10_tc:.4f}\n")
        f.write(f"  MRR:  {avg_mrr_tc:.4f}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"Results saved to: {results_file}")


if __name__ == '__main__':
    app.run(main)
