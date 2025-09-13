from collections import Counter
from torch.utils import data
from typing import Dict, Tuple
import os

Mapping = Dict[str, int]


def create_mappings_from_all(train_path: str, valid_path: str, test_path: str) -> Tuple[Mapping, Mapping]:
    """Creates mappings from all splits (transductive learning)."""
    entity_counter = Counter()
    relation_counter = Counter()
    
    for path in (train_path, valid_path, test_path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                h, r, t = line.rstrip("\n").split("\t")
                entity_counter.update([h, t])
                relation_counter.update([r])
    
    entity2id = {e: i for i, (e, _) in enumerate(entity_counter.most_common())}
    relation2id = {r: i for i, (r, _) in enumerate(relation_counter.most_common())}
    
    print(f"Created mappings: {len(entity2id)} entities, {len(relation2id)} relations")
    return entity2id, relation2id


def create_mappings(dataset_path: str) -> Tuple[Mapping, Mapping]:
    """Creates separate mappings to indices for entities and relations (legacy function)."""
    # counters to have entities/relations sorted from most frequent
    entity_counter = Counter()
    relation_counter = Counter()
    with open(dataset_path, "r") as f:
        for line in f:
            # -1 to remove newline sign
            head, relation, tail = line[:-1].split("\t")
            entity_counter.update([head, tail])
            relation_counter.update([relation])
    entity2id = {}
    relation2id = {}
    for idx, (mid, _) in enumerate(entity_counter.most_common()):
        entity2id[mid] = idx
    for idx, (relation, _) in enumerate(relation_counter.most_common()):
        relation2id[relation] = idx
    return entity2id, relation2id


class MitreDataset(data.Dataset):
    """Dataset implementation for handling MITRE attack pattern data."""

    def __init__(self, data_path: str, entity2id: Mapping, relation2id: Mapping):
        self.entity2id = entity2id
        self.relation2id = relation2id
        with open(data_path, "r") as f:
            # data in tuples (head, relation, tail)
            self.data = [line[:-1].split("\t") for line in f]

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.data)

    def __getitem__(self, index):
        """Returns (head id, relation id, tail id)."""
        head, relation, tail = self.data[index]
        head_id = self._to_idx(head, self.entity2id)
        relation_id = self._to_idx(relation, self.relation2id)
        tail_id = self._to_idx(tail, self.entity2id)
        return head_id, relation_id, tail_id

    @staticmethod
    def _to_idx(key: str, mapping: Mapping) -> int:
        assert key in mapping, f"OOV key in split: {key}"
        return mapping[key]


def _read_triples_txt(path):
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split("\t") if "\t" in s else s.split()
            if len(parts) != 3:
                continue
            h, r, t = parts
            triples.append((h, r, t))
    return triples

def _index_triples(triples):
    e2id, r2id = {}, {}
    def _get(d, k):
        if k not in d: d[k] = len(d)
        return d[k]
    id_triples = []
    for h, r, t in triples:
        id_triples.append((_get(e2id,h), _get(r2id,r), _get(e2id,t)))
    return e2id, r2id, id_triples

def load_mitre_txt(root):
    train = _read_triples_txt(os.path.join(root, "train.txt"))
    valid = _read_triples_txt(os.path.join(root, "valid.txt"))
    test  = _read_triples_txt(os.path.join(root, "test.txt"))
    all_str = train + valid + test
    e2id, r2id, _ = _index_triples(all_str)
    def _map(tr): return [(e2id[h], r2id[r], e2id[t]) for (h,r,t) in tr]
    train_id, valid_id, test_id = _map(train), _map(valid), _map(test)
    all_true = set(_map(all_str))
    return e2id, r2id, (train_id, valid_id, test_id), all_true


def categorize_relations(train_triples, e2type, r2id, type_CWE="CWE", type_CAPEC="CAPEC"):
    """
    Returns: rel_category[id] in {"direct","path","hier"}
    direct = only crosses CWE<->CAPEC (no intra-type)
    hier   = only intra-type and appears as parent/child (name heuristics optional)
    path   = everything else
    
    Filters out PATH_* pseudo-relations which are artifacts from two-hop composition.
    """
    from collections import defaultdict
    
    # Create reverse mapping from relation ID to relation name
    id2rel = {v: k for k, v in r2id.items()}
    
    pairs = defaultdict(set)   # r -> set of (type(h), type(t))
    for (h,r,t) in train_triples:
        # Filter out PATH_* pseudo-relations
        rel_name = id2rel.get(r, "")
        if rel_name.startswith("PATH_"):
            continue
        pairs[r].add((e2type[h], e2type[t]))

    rel_category = {}
    for r, ts in pairs.items():
        if ts.issubset({(type_CWE,type_CWE), (type_CAPEC,type_CAPEC)}):
            rel_category[r] = "hier"
        elif ts.issubset({(type_CWE,type_CAPEC), (type_CAPEC,type_CWE)}) and \
             not ts.intersection({(type_CWE,type_CWE),(type_CAPEC,type_CAPEC)}):
            rel_category[r] = "direct"
        else:
            rel_category[r] = "path"
    return rel_category


def categorize_relations_by_name(id2rel, direct_names=None, hier_names=None):
    """
    Return: rel_category[rid] in {"direct","hier","other"} using relation NAMES only.
    - Stable and matches your domain schema.
    - PATH_* or any unknown names -> "other".
    """
    if direct_names is None:
        direct_names = {
            "has_attack_pattern", "exploits_weakness", "related_to",
            "DIRECT_has_attack_pattern", "DIRECT_exploits_weakness", "DIRECT_related_to",
        }
    if hier_names is None:
        hier_names = {
            "is_child_of", "has_child", "is_parent_of", "has_parent",
            "parent_of", "child_of", "subclass_of", "superclass_of",
        }
    rel_category = {}
    for rid, name in id2rel.items():
        if name in hier_names:
            rel_category[rid] = "hier"
        elif name in direct_names:
            rel_category[rid] = "direct"
        else:
            rel_category[rid] = "other"
    return rel_category