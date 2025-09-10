#!/usr/bin/env python3
"""
Inference script for trained knowledge graph embedding models.
Loads a checkpoint and evaluates on the test set.
"""

from absl import app
from absl import flags
import data
import model as model_definition
import os
import storage
import torch
from torch.utils import data as torch_data
from collections import defaultdict
from typing import Dict, Any
import datetime

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", default="", help="Path to model checkpoint file")
flags.DEFINE_string("dataset_path", default="./mitre_data", help="Path to dataset")
flags.DEFINE_bool("analyze_direct_vs_transitive", default=True, help="Analyze direct vs transitive relationship performance")
flags.DEFINE_bool("show_relation_confidence", default=True, help="Show confidence breakdown by relation type")
flags.DEFINE_bool("use_gpu", default=True, help="Flag enabling gpu usage")
flags.DEFINE_bool("type_constrained", default=True, help="Use type-constrained evaluation")
flags.DEFINE_string("output_file", default="", help="Output file for results (default: auto-generated)")
flags.DEFINE_string("query_cwe", default="", help="CWE ID to query for related CAPECs")
flags.DEFINE_string("query_relation", default="related_to", help="Relation to use for querying (default: related_to)")
flags.DEFINE_integer("top_k", default=10, help="Number of top CAPECs to return")


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


@torch.no_grad()
def eval_tail_filtered(model, dataset, E, device, tails_by_hr, hier_rel_ids, etype_tensor=None, type_constrained=False):
    """Efficient filtered tail evaluation."""
    model.eval()
    all_ranks = []
    for h, r, t in dataset:
        # candidate mask
        mask = torch.ones(E, dtype=torch.bool, device=device)
        # filter out other true tails
        true_t = tails_by_hr.get((h, r), set())
        if true_t:
            mask[torch.tensor(list(true_t), device=device)] = False
        # forbid self for hierarchical relations
        if r in hier_rel_ids:
            mask[h] = False
        
        # Type-constrained evaluation: same type as head for hierarchical relations
        if type_constrained and etype_tensor is not None and r in hier_rel_ids:
            head_type = etype_tensor[h]
            mask = mask & (etype_tensor == head_type)
        
        # Always ensure gold is included AFTER type constraint
        mask[t] = True

        cand = torch.arange(E, device=device)[mask.to(device)]
        bh = torch.full_like(cand, h, device=device)
        br = torch.full_like(cand, r, device=device)
        trip = torch.stack([bh, br, cand], dim=1)
        d = model.predict(trip)

        # dense vector with +inf elsewhere
        vec = torch.full((E,), float("inf"), device=device)
        vec[mask.to(device)] = d
        # sanity: gold must be finite
        assert torch.isfinite(vec[t]).item(), "Gold was filtered out!"

        # rank for this query
        rank = ranks_from_scores(vec.unsqueeze(0), torch.tensor([t], device=device), smaller_is_better=True)
        all_ranks.append(rank)

    ranks = torch.cat(all_ranks, dim=0)
    
    return {
        "MRR": mrr_from_ranks(ranks),
        "H@1": hits_at_k_from_ranks(ranks, 1),
        "H@3": hits_at_k_from_ranks(ranks, 3),
        "H@10": hits_at_k_from_ranks(ranks, 10),
        "num_queries": int(ranks.numel())
    }


@torch.no_grad()
def eval_head_filtered(model, dataset, E, device, heads_by_rt, hier_rel_ids, etype_tensor=None, type_constrained=False):
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
        
        # Type-constrained evaluation: same type as tail for hierarchical relations
        if type_constrained and etype_tensor is not None and r in hier_rel_ids:
            tail_type = etype_tensor[t]
            mask = mask & (etype_tensor == tail_type)
        
        # Always ensure gold is included AFTER type constraint
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
    
    return {
        "MRR": mrr_from_ranks(ranks),
        "H@1": hits_at_k_from_ranks(ranks, 1),
        "H@3": hits_at_k_from_ranks(ranks, 3),
        "H@10": hits_at_k_from_ranks(ranks, 10),
        "num_queries": int(ranks.numel())
    }


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model and extract parameters from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model parameters from checkpoint
    model_state = checkpoint["model_state_dict"]
    
    # Initialize default values
    model_type = "TransE"  # default fallback
    entity_count = 0
    relation_count = 0
    dim = 50
    
    # Infer model type and parameters from the state dict
    if "entities_emb.weight" in model_state:
        entity_count = model_state["entities_emb.weight"].shape[0] - 1  # -1 for padding
        relation_count = model_state["relations_emb.weight"].shape[0] - 1  # -1 for padding
        dim = model_state["entities_emb.weight"].shape[1]
        
        # Determine model type based on relation embedding shape
        if len(model_state["relations_emb.weight"].shape) == 2:
            if model_state["relations_emb.weight"].shape[1] == dim:
                model_type = "TransE"
            elif model_state["relations_emb.weight"].shape[1] == dim * 2:
                model_type = "ComplEx"
            else:
                model_type = "RotatE"
        else:
            model_type = "TransE"  # default fallback
    else:
        print("Warning: Could not detect model type from checkpoint, using TransE as default")
    
    print(f"Detected model type: {model_type}")
    print(f"Entity count: {entity_count}, Relation count: {relation_count}, Dimension: {dim}")
    
    # Create model instance with correct dimensions
    if model_type == "TransE":
        model = model_definition.TransE(
            entity_count=entity_count, 
            relation_count=relation_count, 
            dim=dim, 
            margin=1.0, 
            device=device, 
            norm=1, 
            use_soft_loss=True
        )
    elif model_type == "ComplEx":
        model = model_definition.ComplEx(
            entity_count=entity_count, 
            relation_count=relation_count, 
            dim=dim, 
            margin=1.0, 
            device=device, 
            use_soft_loss=True
        )
    elif model_type == "RotatE":
        model = model_definition.RotatE(
            entity_count=entity_count, 
            relation_count=relation_count, 
            dim=dim, 
            margin=1.0, 
            device=device, 
            use_soft_loss=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Verify dimensions match
    print(f"Model created with: {entity_count} entities, {relation_count} relations, dim={dim}")
    print(f"Entity embedding shape: {model.entities_emb.weight.shape}")
    print(f"Relation embedding shape: {model.relations_emb.weight.shape}")
    
    # Load the state dict
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    
    return model, model_type, entity_count, relation_count, dim


@torch.no_grad()
def analyze_relation_type_performance(model, dataset, entity2id, relation2id, id2entity, id2rel, device):
    """Analyze performance breakdown by relation type (direct vs transitive)."""
    model.eval()
    
    relation_performance = {}
    
    for h, r, t in dataset:
        r_name = id2rel[r]
        
        if r_name not in relation_performance:
            relation_performance[r_name] = {'queries': [], 'ranks': []}
        
        # Simple ranking evaluation for this relation type
        h_name = id2entity[h]
        t_name = id2entity[t]
        
        # Create candidates of the same type as target
        if t_name.startswith('CAPEC-'):
            candidates = [i for i, name in enumerate(id2entity) if name.startswith('CAPEC-')]
        elif t_name.startswith('CWE-'):
            candidates = [i for i, name in enumerate(id2entity) if name.startswith('CWE-')]
        else:
            continue
        
        if len(candidates) <= 1:
            continue
            
        # Create query triplets
        h_tensor = torch.full((len(candidates),), h, device=device)
        r_tensor = torch.full((len(candidates),), r, device=device)
        t_tensor = torch.tensor(candidates, device=device)
        
        triplets = torch.stack([h_tensor, r_tensor, t_tensor], dim=1)
        scores = model.predict(triplets)
        
        # Find rank of gold answer
        gold_idx = candidates.index(t)
        gold_score = scores[gold_idx]
        better_scores = (scores < gold_score).sum().item()  # Lower score = better
        rank = better_scores + 1
        
        relation_performance[r_name]['queries'].append((h_name, r_name, t_name))
        relation_performance[r_name]['ranks'].append(rank)
    
    # Calculate metrics for each relation type
    results = {}
    for r_name, data in relation_performance.items():
        if len(data['ranks']) > 0:
            ranks = torch.tensor(data['ranks'], dtype=torch.float)
            results[r_name] = {
                'count': len(data['ranks']),
                'MRR': (1.0 / ranks).mean().item(),
                'H@1': (ranks <= 1).float().mean().item(),
                'H@3': (ranks <= 3).float().mean().item(),
                'H@10': (ranks <= 10).float().mean().item(),
                'avg_rank': ranks.mean().item()
            }
    
    return results


@torch.no_grad()
def query_cwe_to_capecs(model, cwe_id: str, relation: str, entity2id: Dict[str, int], relation2id: Dict[str, int], 
                        id2entity: list, etype_tensor: torch.Tensor, device: torch.device, top_k: int = 10):
    """Query for CAPECs related to a given CWE."""
    model.eval()
    
    # Check if CWE exists
    if cwe_id not in entity2id:
        print(f"Error: CWE {cwe_id} not found in entity mappings")
        print(f"Available CWEs: {[e for e in entity2id.keys() if e.startswith('CWE-')][:10]}...")
        return None
    
    # Check if relation exists
    if relation not in relation2id:
        print(f"Error: Relation {relation} not found in relation mappings")
        print(f"Available relations: {list(relation2id.keys())}")
        return None
    
    cwe_entity_id = entity2id[cwe_id]
    relation_id = relation2id[relation]
    
    print(f"Querying: {cwe_id} (ID: {cwe_entity_id}) -> {relation} (ID: {relation_id}) -> CAPECs")
    
    # Get all CAPEC entities (type 1)
    capec_mask = (etype_tensor == 1)
    capec_entity_ids = torch.arange(len(etype_tensor), device=device)[capec_mask]
    
    if len(capec_entity_ids) == 0:
        print("No CAPEC entities found in the dataset")
        return None
    
    print(f"Found {len(capec_entity_ids)} CAPEC entities")
    print(f"Sample CAPECs: {[id2entity[i.item()] for i in capec_entity_ids[:5]]}")
    
    # Create query triplets: (cwe_id, relation, capec_id)
    cwe_tensor = torch.full_like(capec_entity_ids, cwe_entity_id, device=device)
    relation_tensor = torch.full_like(capec_entity_ids, relation_id, device=device)
    query_triplets = torch.stack([cwe_tensor, relation_tensor, capec_entity_ids], dim=1)
    
    # Get prediction scores (lower is better for distance-based models)
    with torch.no_grad():
        scores = model.predict(query_triplets)
    
    # Convert to confidence scores (higher is better)
    # For distance-based models, we convert distance to similarity
    # Use a more robust confidence calculation
    mean_score = scores.mean()
    std_score = scores.std()
    
    if std_score > 0:
        # Normalize using z-score and convert to confidence
        z_scores = (scores - mean_score) / std_score
        # Convert to 0-1 range using sigmoid
        confidence_scores = torch.sigmoid(-z_scores)  # Negative because lower distance = higher confidence
    else:
        confidence_scores = torch.ones_like(scores) * 0.5  # All equal if no variation
    
    # Get top-k results
    top_indices = torch.topk(confidence_scores, min(top_k, len(confidence_scores))).indices
    top_capec_ids = capec_entity_ids[top_indices]
    top_confidences = confidence_scores[top_indices]
    
    # Convert to entity names and format results
    results = []
    for i, (capec_id, confidence) in enumerate(zip(top_capec_ids, top_confidences)):
        capec_name = id2entity[capec_id.item()]
        confidence_pct = confidence.item() * 100
        results.append({
            'rank': i + 1,
            'capec_id': capec_name,
            'confidence': confidence.item(),
            'confidence_pct': confidence_pct
        })
    
    return results


def print_query_results(query_results: list, cwe_id: str, relation: str):
    """Print query results in a formatted way."""
    if query_results is None:
        return
    
    print(f"\n{'='*60}")
    print(f"Query: CWE {cwe_id} -> {relation} -> CAPECs")
    print(f"{'='*60}")
    print(f"{'Rank':<6} {'CAPEC ID':<15} {'Confidence':<12} {'Confidence %':<15}")
    print(f"{'-'*60}")
    
    for result in query_results:
        print(f"{result['rank']:<6} {result['capec_id']:<15} {result['confidence']:<12.4f} {result['confidence_pct']:<15.2f}%")
    
    print(f"{'='*60}")


def save_query_results(query_results: list, cwe_id: str, relation: str, model_type: str, output_file: str):
    """Save query results to file."""
    if query_results is None:
        return
    
    import datetime
    
    if not output_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"query_{model_type}_{cwe_id}_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"CWE to CAPEC Query Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Query: CWE {cwe_id} -> {relation} -> CAPECs\n")
        f.write(f"Number of results: {len(query_results)}\n")
        f.write("\n")
        
        f.write(f"{'Rank':<6} {'CAPEC ID':<15} {'Confidence':<12} {'Confidence %':<15}\n")
        f.write("-" * 60 + "\n")
        
        for result in query_results:
            f.write(f"{result['rank']:<6} {result['capec_id']:<15} {result['confidence']:<12.4f} {result['confidence_pct']:<15.2f}%\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"Query results saved to: {output_file}")


def save_inference_results(results: Dict[str, Any], model_type: str, dataset_path: str, output_file: str):
    """Save inference results to file."""
    import datetime
    
    if not output_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"inference_{model_type}_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{model_type} Model Inference Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Type-Constrained: {results['type_constrained']}\n")
        f.write("\n")
        
        # Regular filtered evaluation results
        f.write("Regular Filtered Evaluation Results:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Tail Prediction:\n")
        f.write(f"  H@1:  {results['regular_tail']['H@1']:.4f}\n")
        f.write(f"  H@3:  {results['regular_tail']['H@3']:.4f}\n")
        f.write(f"  H@10: {results['regular_tail']['H@10']:.4f}\n")
        f.write(f"  MRR:  {results['regular_tail']['MRR']:.4f}\n")
        f.write(f"  Queries: {results['regular_tail']['num_queries']}\n")
        f.write(f"\nHead Prediction:\n")
        f.write(f"  H@1:  {results['regular_head']['H@1']:.4f}\n")
        f.write(f"  H@3:  {results['regular_head']['H@3']:.4f}\n")
        f.write(f"  H@10: {results['regular_head']['H@10']:.4f}\n")
        f.write(f"  MRR:  {results['regular_head']['MRR']:.4f}\n")
        f.write(f"  Queries: {results['regular_head']['num_queries']}\n")
        
        # Regular averages
        avg_hits_1 = (results['regular_tail']["H@1"] + results['regular_head']["H@1"]) / 2
        avg_hits_3 = (results['regular_tail']["H@3"] + results['regular_head']["H@3"]) / 2
        avg_hits_10 = (results['regular_tail']["H@10"] + results['regular_head']["H@10"]) / 2
        avg_mrr = (results['regular_tail']["MRR"] + results['regular_head']["MRR"]) / 2
        f.write(f"\nRegular Average:\n")
        f.write(f"  H@1:  {avg_hits_1:.4f}\n")
        f.write(f"  H@3:  {avg_hits_3:.4f}\n")
        f.write(f"  H@10: {avg_hits_10:.4f}\n")
        f.write(f"  MRR:  {avg_mrr:.4f}\n")
        f.write("\n")
        
        # Type-constrained evaluation results (if applicable)
        if results['type_constrained']:
            f.write("Type-Constrained Filtered Evaluation Results:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Tail Prediction:\n")
            f.write(f"  H@1:  {results['tc_tail']['H@1']:.4f}\n")
            f.write(f"  H@3:  {results['tc_tail']['H@3']:.4f}\n")
            f.write(f"  H@10: {results['tc_tail']['H@10']:.4f}\n")
            f.write(f"  MRR:  {results['tc_tail']['MRR']:.4f}\n")
            f.write(f"  Queries: {results['tc_tail']['num_queries']}\n")
            f.write(f"\nHead Prediction:\n")
            f.write(f"  H@1:  {results['tc_head']['H@1']:.4f}\n")
            f.write(f"  H@3:  {results['tc_head']['H@3']:.4f}\n")
            f.write(f"  H@10: {results['tc_head']['H@10']:.4f}\n")
            f.write(f"  MRR:  {results['tc_head']['MRR']:.4f}\n")
            f.write(f"  Queries: {results['tc_head']['num_queries']}\n")
            
            # Type-constrained averages
            avg_hits_1_tc = (results['tc_tail']["H@1"] + results['tc_head']["H@1"]) / 2
            avg_hits_3_tc = (results['tc_tail']["H@3"] + results['tc_head']["H@3"]) / 2
            avg_hits_10_tc = (results['tc_tail']["H@10"] + results['tc_head']["H@10"]) / 2
            avg_mrr_tc = (results['tc_tail']["MRR"] + results['tc_head']["MRR"]) / 2
            f.write(f"\nType-Constrained Average:\n")
            f.write(f"  H@1:  {avg_hits_1_tc:.4f}\n")
            f.write(f"  H@3:  {avg_hits_3_tc:.4f}\n")
            f.write(f"  H@10: {avg_hits_10_tc:.4f}\n")
            f.write(f"  MRR:  {avg_mrr_tc:.4f}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"Inference results saved to: {output_file}")


def main(_):
    if not FLAGS.checkpoint_path:
        print("Error: --checkpoint_path is required")
        return
    
    device = torch.device('cuda' if FLAGS.use_gpu and torch.cuda.is_available() else 'cpu')
    if FLAGS.use_gpu and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU instead.")
    
    # Load model from checkpoint
    print(f"Loading model from checkpoint: {FLAGS.checkpoint_path}")
    model, model_type, entity_count, relation_count, dim = load_model_from_checkpoint(FLAGS.checkpoint_path, device)
    
    # Load dataset
    path = FLAGS.dataset_path
    train_path = os.path.join(path, "train.txt")
    validation_path = os.path.join(path, "valid.txt")
    test_path = os.path.join(path, "test.txt")
    
    entity2id, relation2id = data.create_mappings_from_all(train_path, validation_path, test_path)
    
    # Build inverse maps
    id2entity = [None] * len(entity2id)
    for e, i in entity2id.items(): 
        id2entity[i] = e
    id2rel = [None] * len(relation2id)
    for r, i in relation2id.items(): 
        id2rel[i] = r
    
    # Load test set
    test_set = data.FB15KDataset(test_path, entity2id, relation2id)
    
    # Build filtered sets
    print("Building filtered evaluation sets...")
    tails_by_hr, heads_by_rt = build_known_sets_id([train_path, validation_path, test_path], entity2id, relation2id)
    
    # Schema-aware hierarchical relations and CWE-CAPEC specific relations
    HIER_NAMES = {"is_child_of", "has_child", "is_parent_of", "has_parent", "parent_of", "child_of", "subclass_of", "superclass_of", "related_to"}
    CWE_CAPEC_NAMES = {"has_attack_pattern", "exploits_weakness", "related_to"}
    ALL_REL_NAMES = HIER_NAMES | CWE_CAPEC_NAMES
    HIER_REL_IDS = {relation2id[r] for r in relation2id if r in ALL_REL_NAMES}
    print(f"Found {len(HIER_REL_IDS)} hierarchical relations: {[id2rel[r_id] for r_id in HIER_REL_IDS]}")
    
    # Entity type detection
    etype_tensor = torch.zeros(len(entity2id), dtype=torch.long, device=device)
    for entity, eid in entity2id.items():
        if entity.startswith("CWE-"):
            etype_tensor[eid] = 0  # CWE type
        elif entity.startswith("CAPEC-"):
            etype_tensor[eid] = 1  # CAPEC type
        else:
            etype_tensor[eid] = 0  # default to CWE type
    
    cwe_count = (etype_tensor == 0).sum().item()
    capec_count = (etype_tensor == 1).sum().item()
    print(f"Entity types: {cwe_count} CWE entities, {capec_count} CAPEC entities")
    
    # Run inference
    print("=== RUNNING INFERENCE ===")
    
    # Regular filtered evaluation
    print("\n--- Regular Filtered Evaluation ---")
    tail_results = eval_tail_filtered(model, test_set, len(entity2id), device, tails_by_hr, HIER_REL_IDS, etype_tensor, type_constrained=False)
    head_results = eval_head_filtered(model, test_set, len(entity2id), device, heads_by_rt, HIER_REL_IDS, etype_tensor, type_constrained=False)
    
    print(f"Regular - Tail: H@1={tail_results['H@1']:.3f}, H@3={tail_results['H@3']:.3f}, H@10={tail_results['H@10']:.3f}, MRR={tail_results['MRR']:.3f}")
    print(f"Regular - Head: H@1={head_results['H@1']:.3f}, H@3={head_results['H@3']:.3f}, H@10={head_results['H@10']:.3f}, MRR={head_results['MRR']:.3f}")
    
    avg_hits_1 = (tail_results["H@1"] + head_results["H@1"]) / 2
    avg_hits_3 = (tail_results["H@3"] + head_results["H@3"]) / 2
    avg_hits_10 = (tail_results["H@10"] + head_results["H@10"]) / 2
    avg_mrr = (tail_results["MRR"] + head_results["MRR"]) / 2
    print(f"Regular Average: H@1={avg_hits_1:.3f}, H@3={avg_hits_3:.3f}, H@10={avg_hits_10:.3f}, MRR={avg_mrr:.3f}")
    
    results = {
        'regular_tail': tail_results,
        'regular_head': head_results,
        'type_constrained': FLAGS.type_constrained,
        'dataset_path': FLAGS.dataset_path
    }
    
    # Type-constrained evaluation (if requested)
    if FLAGS.type_constrained:
        print("\n--- Type-Constrained Filtered Evaluation ---")
        tail_results_tc = eval_tail_filtered(model, test_set, len(entity2id), device, tails_by_hr, HIER_REL_IDS, etype_tensor, type_constrained=True)
        head_results_tc = eval_head_filtered(model, test_set, len(entity2id), device, heads_by_rt, HIER_REL_IDS, etype_tensor, type_constrained=True)
        
        print(f"Type-Constrained - Tail: H@1={tail_results_tc['H@1']:.3f}, H@3={tail_results_tc['H@3']:.3f}, H@10={tail_results_tc['H@10']:.3f}, MRR={tail_results_tc['MRR']:.3f}")
        print(f"Type-Constrained - Head: H@1={head_results_tc['H@1']:.3f}, H@3={head_results_tc['H@3']:.3f}, H@10={head_results_tc['H@10']:.3f}, MRR={head_results_tc['MRR']:.3f}")
        
        avg_hits_1_tc = (tail_results_tc["H@1"] + head_results_tc["H@1"]) / 2
        avg_hits_3_tc = (tail_results_tc["H@3"] + head_results_tc["H@3"]) / 2
        avg_hits_10_tc = (tail_results_tc["H@10"] + head_results_tc["H@10"]) / 2
        avg_mrr_tc = (tail_results_tc["MRR"] + head_results_tc["MRR"]) / 2
        print(f"Type-Constrained Average: H@1={avg_hits_1_tc:.3f}, H@3={avg_hits_3_tc:.3f}, H@10={avg_hits_10_tc:.3f}, MRR={avg_mrr_tc:.3f}")
        
        results['tc_tail'] = tail_results_tc
        results['tc_head'] = head_results_tc
    
    # Analyze relation type performance
    if FLAGS.analyze_direct_vs_transitive:
        print("\n=== RELATION TYPE ANALYSIS ===")
        relation_results = analyze_relation_type_performance(model, test_set, entity2id, relation2id, id2entity, id2rel, device)
        
        print("\nPerformance by Relation Type:")
        print(f"{'Relation':<25} {'Count':<8} {'H@1':<8} {'H@3':<8} {'H@10':<8} {'MRR':<8} {'Avg Rank':<10}")
        print("-" * 85)
        
        # Sort by whether relation is direct or not
        direct_relations = [(r, data) for r, data in relation_results.items() if r.startswith('DIRECT_')]
        transitive_relations = [(r, data) for r, data in relation_results.items() if r.startswith('PATH_')]
        other_relations = [(r, data) for r, data in relation_results.items() if not r.startswith('DIRECT_') and not r.startswith('PATH_')]
        
        for category, relations in [("DIRECT RELATIONS", direct_relations), 
                                   ("TRANSITIVE RELATIONS", transitive_relations),
                                   ("OTHER RELATIONS", other_relations)]:
            if relations:
                print(f"\n{category}:")
                for r_name, data in relations:
                    print(f"{r_name:<25} {data['count']:<8} {data['H@1']:<8.3f} {data['H@3']:<8.3f} "
                          f"{data['H@10']:<8.3f} {data['MRR']:<8.3f} {data['avg_rank']:<10.1f}")
        
        results['relation_analysis'] = relation_results

    # Save results
    save_inference_results(results, model_type, FLAGS.dataset_path, FLAGS.output_file)
    
    # Query functionality
    if FLAGS.query_cwe:
        print(f"\n=== QUERYING CWE {FLAGS.query_cwe} ===")
        query_results = query_cwe_to_capecs(
            model, 
            FLAGS.query_cwe, 
            FLAGS.query_relation, 
            entity2id, 
            relation2id, 
            id2entity, 
            etype_tensor, 
            device, 
            FLAGS.top_k
        )
        
        if query_results:
            print_query_results(query_results, FLAGS.query_cwe, FLAGS.query_relation)
            
            # Save query results
            query_output_file = f"query_{model_type}_{FLAGS.query_cwe}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            save_query_results(query_results, FLAGS.query_cwe, FLAGS.query_relation, model_type, query_output_file)


if __name__ == '__main__':
    app.run(main)
