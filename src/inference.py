#!/usr/bin/env python3
"""
Inference script for trained knowledge graph embedding models.
Loads a checkpoint and evaluates on the test set.
"""

from absl import app
from absl import flags
import data as data_module
# Models are now imported from the models package
from models import build_model
import os
import storage
import torch
from torch.utils import data as torch_data
from collections import defaultdict
from typing import Dict, Any
import datetime

FLAGS = flags.FLAGS

# Required flags
flags.DEFINE_string("checkpoint_path", default="", help="Path to model checkpoint (.pt file)")
flags.DEFINE_string("dataset_path", default="./mitre-data/txt", help="Path to dataset directory")
flags.DEFINE_bool("use_gpu", default=True, help="Use GPU if available")

# Optional flags
flags.DEFINE_bool("analyze_direct_vs_transitive", default=True, help="Analyze direct vs transitive relationship performance")
flags.DEFINE_bool("show_relation_confidence", default=True, help="Show confidence breakdown by relation type")
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




@torch.no_grad()
def eval_tail_filtered(model, dataset, E, device, tails_by_hr,
                       type_constrained=False, allowed_tails=None, hier_rel_ids=None):
    """Efficient filtered tail evaluation - higher=better."""
    model.eval()
    ranks = []
    for h, r, t in dataset:
        mask = torch.ones(E, dtype=torch.bool, device=device)
        true = tails_by_hr.get((h,r), set())
        if true:
            mask[torch.tensor(list(true), device=device)] = False
        if hier_rel_ids and r in hier_rel_ids:
            mask[h] = False
        if type_constrained and allowed_tails is not None:
            allowed = allowed_tails.get(r, None)
            if allowed is not None and len(allowed) < E:
                m2 = torch.zeros(E, dtype=torch.bool, device=device)
                m2[torch.tensor(list(allowed), device=device)] = True
                mask &= m2
        mask[t] = True

        cand = torch.arange(E, device=device)[mask]
        s = model.score_triples(
            torch.full_like(cand, h), torch.full_like(cand, r), cand
        )
        vec = torch.full((E,), float("-inf"), device=device); vec[mask] = s
        # higher=better rank
        gold = torch.tensor([t], device=device)
        gold_s = vec[gold]
        rank = 1 + (vec > gold_s).sum()
        ranks.append(rank)
        
        # Sanity checks
        assert torch.isfinite(vec[t]).item(), "Gold got masked!"
        assert (rank >= 1).item() and (rank <= E).item(), f"Invalid rank: {rank}"
        if type_constrained:
            assert mask[t].item(), "Gold not re-included under type constraints"
    
    ranks = torch.stack(ranks)
    return {
        "MRR": float((1.0 / ranks.float()).mean().item()),
        "H@1": float((ranks<=1).float().mean().item()),
        "H@3": float((ranks<=3).float().mean().item()),
        "H@10": float((ranks<=10).float().mean().item()),
        "num_queries": int(ranks.numel()),
    }


@torch.no_grad()
def eval_head_filtered(model, dataset, E, device, heads_by_rt,
                       type_constrained=False, allowed_heads=None, hier_rel_ids=None):
    """Efficient filtered head evaluation - higher=better."""
    model.eval()
    ranks = []
    for h, r, t in dataset:
        mask = torch.ones(E, dtype=torch.bool, device=device)
        true = heads_by_rt.get((r,t), set())
        if true:
            mask[torch.tensor(list(true), device=device)] = False
        if hier_rel_ids and r in hier_rel_ids:
            mask[t] = False
        if type_constrained and allowed_heads is not None:
            allowed = allowed_heads.get(r, None)
            if allowed is not None and len(allowed) < E:
                m2 = torch.zeros(E, dtype=torch.bool, device=device)
                m2[torch.tensor(list(allowed), device=device)] = True
                mask &= m2
        mask[h] = True

        cand = torch.arange(E, device=device)[mask]
        s = model.score_triples(
            cand, torch.full_like(cand, r), torch.full_like(cand, t)
        )
        vec = torch.full((E,), float("-inf"), device=device); vec[mask] = s
        # higher=better rank
        gold = torch.tensor([h], device=device)
        gold_s = vec[gold]
        rank = 1 + (vec > gold_s).sum()
        ranks.append(rank)
        
        # Sanity checks
        assert torch.isfinite(vec[h]).item(), "Gold got masked!"
        assert (rank >= 1).item() and (rank <= E).item(), f"Invalid rank: {rank}"
        if type_constrained:
            assert mask[h].item(), "Gold not re-included under type constraints"
    
    ranks = torch.stack(ranks)
    return {
        "MRR": float((1.0 / ranks.float()).mean().item()),
        "H@1": float((ranks<=1).float().mean().item()),
        "H@3": float((ranks<=3).float().mean().item()),
        "H@10": float((ranks<=10).float().mean().item()),
        "num_queries": int(ranks.numel()),
    }


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model and extract parameters from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model parameters from checkpoint
    model_state = checkpoint["model"]
    
    # Initialize default values
    model_type = "TransE"  # default fallback
    entity_count = 0
    relation_count = 0
    dim = 50
    
    # Infer model type and parameters from the state dict
    print(f"Available keys in checkpoint: {list(model_state.keys())}")
    
    if "ent.weight" in model_state:
        entity_count = model_state["ent.weight"].shape[0]  # Keep original count (includes padding)
        relation_count = model_state["rel.weight"].shape[0]  # Keep original count (includes padding)
        ent_dim = model_state["ent.weight"].shape[1]
        rel_dim = model_state["rel.weight"].shape[1]
        
        # Determine model type based on embedding dimensions
        if ent_dim == rel_dim:
            # Both have same dimension - could be TransE or ComplEx
            if ent_dim % 2 == 0:
                # Even dimension - likely ComplEx (real|imag)
                model_type = "ComplEx"
                dim = ent_dim // 2
                print(f"Detected ComplEx: entities={entity_count}, relations={relation_count}, dim={dim}")
            else:
                # Odd dimension - likely TransE
                model_type = "TransE"
                dim = ent_dim
                print(f"Detected TransE: entities={entity_count}, relations={relation_count}, dim={dim}")
        else:
            # Different dimensions - likely RotatE (ent: dim*2, rel: dim)
            model_type = "RotatE"
            dim = rel_dim
            print(f"Detected RotatE: entities={entity_count}, relations={relation_count}, dim={dim}")
    elif "entities_emb.weight" in model_state:
        entity_count = model_state["entities_emb.weight"].shape[0]  # Keep original count
        relation_count = model_state["relations_emb.weight"].shape[0]  # Keep original count
        dim = model_state["entities_emb.weight"].shape[1]
        model_type = "TransE"  # Default for single embedding
        print(f"Detected TransE (alt): entities={entity_count}, relations={relation_count}, dim={dim}")
    elif "entities_emb_re.weight" in model_state:
        entity_count = model_state["entities_emb_re.weight"].shape[0]  # Keep original count
        relation_count = model_state["relations_emb_re.weight"].shape[0]  # Keep original count
        dim = model_state["entities_emb_re.weight"].shape[1]
        
        # Check if it's ComplEx (has both real and imaginary relation embeddings)
        if "relations_emb_im.weight" in model_state:
            model_type = "ComplEx"
            print(f"Detected ComplEx: entities={entity_count}, relations={relation_count}, dim={dim}")
        else:
            model_type = "RotatE"
            print(f"Detected RotatE: entities={entity_count}, relations={relation_count}, dim={dim}")
    else:
        print("Warning: Could not detect model type from checkpoint, using TransE as default")
        # Try to infer from available keys
        if "entities_emb_re.weight" in model_state:
            entity_count = model_state["entities_emb_re.weight"].shape[0]
            relation_count = model_state["relations_emb.weight"].shape[0]
            dim = model_state["entities_emb_re.weight"].shape[1]
            model_type = "RotatE"
            print(f"Fallback RotatE: entities={entity_count}, relations={relation_count}, dim={dim}")
        else:
            entity_count = 0
            relation_count = 0
            dim = 50
            print(f"Fallback defaults: entities={entity_count}, relations={relation_count}, dim={dim}")
    
    print(f"Detected model type: {model_type}")
    print(f"Entity count: {entity_count}, Relation count: {relation_count}, Dimension: {dim}")
    
    # Create model instance with correct dimensions using build_model
    model = build_model(
        name=model_type,
        num_entities=entity_count,
        num_relations=relation_count,
        dim=dim,
        margin=1.0,
        device=device
    )
    
    # Verify dimensions match
    print(f"Model created with: {entity_count} entities, {relation_count} relations, dim={dim}")
    if hasattr(model, 'ent'):
        print(f"Entity embedding shape: {model.ent.weight.shape}")
        print(f"Relation embedding shape: {model.rel.weight.shape}")
    elif hasattr(model, 'entities_emb'):
        print(f"Entity embedding shape: {model.entities_emb.weight.shape}")
        print(f"Relation embedding shape: {model.relations_emb.weight.shape}")
    elif hasattr(model, 'entities_emb_re'):
        print(f"Entity embedding (real) shape: {model.entities_emb_re.weight.shape}")
        print(f"Entity embedding (imag) shape: {model.entities_emb_im.weight.shape}")
        if hasattr(model, 'relations_emb_re'):
            print(f"Relation embedding (real) shape: {model.relations_emb_re.weight.shape}")
            print(f"Relation embedding (imag) shape: {model.relations_emb_im.weight.shape}")
        else:
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
        scores = model.score_triples(triplets[:, 0], triplets[:, 1], triplets[:, 2])
        
        # Find rank of gold answer
        gold_idx = candidates.index(t)
        gold_score = scores[gold_idx]
        better_scores = (scores > gold_score).sum().item()  # Higher score = better
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
    
    # Get prediction scores (higher is better for score-based models)
    with torch.no_grad():
        scores = model.score_triples(query_triplets[:, 0], query_triplets[:, 1], query_triplets[:, 2])
    
    # Convert to confidence scores (higher is better)
    # For distance-based models, we convert distance to similarity
    # Use a more robust confidence calculation
    mean_score = scores.mean()
    std_score = scores.std()
    
    if std_score > 0:
        # Normalize using z-score and convert to confidence
        z_scores = (scores - mean_score) / std_score
        # Convert to 0-1 range using sigmoid (higher scores = higher confidence)
        confidence_scores = torch.sigmoid(z_scores)
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
    
    entity2id, relation2id = data_module.create_mappings_from_all(train_path, validation_path, test_path)
    
    # Build inverse maps (fast lookup)
    id2entity = {i: s for s, i in entity2id.items()}
    id2rel = {i: s for s, i in relation2id.items()}
    
    # Load test set
    test_set = data_module.MitreDataset(test_path, entity2id, relation2id)
    
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
    
    # Build type constraints for evaluation
    from collections import defaultdict
    allowed_heads = defaultdict(set)
    allowed_tails = defaultdict(set)
    train_triples = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            h, r, t = line.rstrip("\n").split("\t")
            if h in entity2id and r in relation2id and t in entity2id:
                h_id, r_id, t_id = entity2id[h], relation2id[r], entity2id[t]
                train_triples.append((h_id, r_id, t_id))
                allowed_heads[r_id].add(h_id)
                allowed_tails[r_id].add(t_id)

    # Regular filtered evaluation
    print("\n--- Regular Filtered Evaluation ---")
    tail_results = eval_tail_filtered(model, test_set, len(entity2id), device, tails_by_hr, 
                                     type_constrained=False, allowed_tails=allowed_tails, hier_rel_ids=HIER_REL_IDS)
    head_results = eval_head_filtered(model, test_set, len(entity2id), device, heads_by_rt, 
                                      type_constrained=False, allowed_heads=allowed_heads, hier_rel_ids=HIER_REL_IDS)
    
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
        tail_results_tc = eval_tail_filtered(model, test_set, len(entity2id), device, tails_by_hr, 
                                           type_constrained=True, allowed_tails=allowed_tails, hier_rel_ids=HIER_REL_IDS)
        head_results_tc = eval_head_filtered(model, test_set, len(entity2id), device, heads_by_rt, 
                                            type_constrained=True, allowed_heads=allowed_heads, hier_rel_ids=HIER_REL_IDS)
        
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


def rank_topk(model, h_str, r_str, e2id, r2id, k=10, filtered=True, all_true=None):
    """Rank top-k entities for a given head-relation pair."""
    device = next(model.parameters()).device
    H = torch.tensor([e2id[h_str]], device=device)
    R = torch.tensor([r2id[r_str]], device=device)
    E = len(e2id)
    tails = torch.arange(E, device=device)
    scores = model.score_triples(H, R, tails)
    
    if filtered and all_true is not None:
        h, r = H.item(), R.item()
        # mask all true except the one we will read out
        for tprime in range(E):
            if (h, r, tprime) in all_true:
                scores[tprime] = -1e9
    
    topk = torch.topk(scores, k)
    inv_e = {v: k for k, v in e2id.items()}
    return [(inv_e[idx.item()], scores[idx].item()) for idx in topk.indices]


def predict_binary(model, triple_str, e2id, r2id, calibrator=None):
    """Predict binary classification for a triple with optional calibration."""
    h, r, t = triple_str
    device = next(model.parameters()).device
    s = model.score_triples(torch.tensor([e2id[h]], device=device),
                           torch.tensor([r2id[r]], device=device),
                           torch.tensor([e2id[t]], device=device)).item()
    return {"score": s, "prob": (calibrator.transform([s])[0] if calibrator else None)}


if __name__ == '__main__':
    app.run(main)
