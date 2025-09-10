# Unified CWE-CAPEC Knowledge Graph Embedding

This project implements knowledge graph embeddings for MITRE CWE-CAPEC relationships with advanced techniques to distinguish between direct and transitive reasoning.

## 🎯 Key Features

### **Multi-Hop Aware Training**
- **Relation Marking**: Direct CWE-CAPEC relationships are marked with `DIRECT_` prefix
- **Path Relations**: Explicit 2-hop paths are created as `PATH_relation1_relation2`
- **Weighted Loss**: Direct relationships get higher loss weights (2x by default)
- **Relation-Aware Negatives**: Challenging negatives for direct relationships

### **Comprehensive Analysis**
- **Relation Type Breakdown**: Performance analysis by direct vs transitive relations
- **Ground Truth Comparison**: Shows which predictions are actually correct
- **Type-Constrained Evaluation**: Separate evaluation for CWE↔CAPEC relationships

## 🚀 Quick Start

### 1. Create Dataset
```bash
python create_mitre_dataset.py
```
This creates a unified dataset with:
- Direct CWE-CAPEC relationships (marked with `DIRECT_` prefix)
- Hierarchical relationships (for entities involved in direct relationships)
- Explicit 2-hop path relationships (marked with `PATH_` prefix)

### 2. Train Model
```bash
# Basic training with multi-hop awareness
python main.py --model TransE --epochs 1000

# Advanced training with custom weights
python main.py --model TransE --multi_hop_aware --direct_relation_weight 3.0 --relation_aware_negatives

# Disable multi-hop features for comparison
python main.py --model TransE --multi_hop_aware=False --relation_aware_negatives=False
```

### 3. Run Inference
```bash
# Full evaluation with relation type analysis
python inference.py --checkpoint_path checkpoint_TransE.tar

# Query specific CWE for CAPECs
python inference.py --checkpoint_path checkpoint_TransE.tar --query_cwe CWE-79 --query_relation DIRECT_has_attack_pattern

# Disable advanced analysis
python inference.py --checkpoint_path checkpoint_TransE.tar --analyze_direct_vs_transitive=False
```

## 🔧 Configuration Options

### Dataset Creation
- `include_hierarchical`: Include hierarchical CWE/CAPEC relationships (default: True)
- `mark_direct_relations`: Mark direct relationships with DIRECT_ prefix (default: True)

### Training
- `multi_hop_aware`: Enable multi-hop aware training techniques (default: True)
- `direct_relation_weight`: Weight boost for direct relationships (default: 2.0)
- `relation_aware_negatives`: Use relation-aware negative sampling (default: True)

### Inference
- `analyze_direct_vs_transitive`: Analyze performance by relation type (default: True)
- `show_relation_confidence`: Show confidence breakdown (default: True)

## 📊 Expected Results

With multi-hop aware training, you should see:

1. **Higher precision for direct relationships**: DIRECT_has_attack_pattern should perform better than transitive paths
2. **Better ranking of correct CAPECs**: Direct attack patterns like CWE-79 → CAPEC-85 should rank higher
3. **Clear separation**: Direct relations should have better H@1/H@3 scores than PATH_ relations

## 🎯 Techniques Used

### 1. **Relation-Aware Negative Sampling**
- 50% standard corruption
- 50% challenging negatives (same entity type)
- Forces model to distinguish between similar entities

### 2. **Multi-Hop Aware Loss Weighting**
- Direct relationships: 2x weight (configurable)
- Path relationships: 0.5x weight
- Encourages focus on direct attack patterns

### 3. **Explicit Path Modeling**
- Creates explicit PATH_rel1_rel2 relations for 2-hop paths
- Allows model to learn transitivity while distinguishing from direct relations

### 4. **Comprehensive Evaluation**
- Separate metrics for each relation type
- Ground truth comparison for queries
- Type-constrained evaluation for CWE↔CAPEC focus

## 📈 Monitoring Training

Use TensorBoard to monitor training:
```bash
tensorboard --logdir ./runs
```

Key metrics to watch:
- Loss trends for different relation types
- Validation performance on direct vs transitive relations
- Distance distributions for positive vs negative samples

## 🔍 Debugging

If direct relationships aren't ranking well:
1. Increase `direct_relation_weight` (try 3.0 or 5.0)
2. Enable `relation_aware_negatives` if disabled
3. Check dataset balance between direct and hierarchical relations
4. Verify DIRECT_ prefixed relations exist in your dataset

## 📝 Files

- `create_mitre_dataset.py`: Unified dataset creation
- `main.py`: Unified training with multi-hop awareness
- `inference.py`: Unified inference with relation analysis
- `model.py`: TransE, ComplEx, RotatE implementations
- `data.py`: Dataset loading utilities
- `storage.py`: Checkpoint management
