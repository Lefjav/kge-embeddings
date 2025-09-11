import torch
from torch import nn
from torch.optim import optimizer
from typing import Tuple
from collections import defaultdict

_MODEL_STATE_DICT = "model_state_dict"
_OPTIMIZER_STATE_DICT = "optimizer_state_dict"
_EPOCH = "epoch"
_STEP = "step"
_BEST_SCORE = "best_score"


def load_checkpoint(checkpoint_path: str, model: nn.Module, optim: optimizer.Optimizer) -> Tuple[int, int, float]:
    """Loads training checkpoint.

    :param checkpoint_path: path to checkpoint
    :param model: model to update state
    :param optim: optimizer to  update state
    :return tuple of starting epoch id, starting step id, best checkpoint score
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint[_MODEL_STATE_DICT])
    optim.load_state_dict(checkpoint[_OPTIMIZER_STATE_DICT])
    start_epoch_id = checkpoint[_EPOCH] + 1
    step = checkpoint[_STEP] + 1
    best_score = checkpoint[_BEST_SCORE]
    return start_epoch_id, step, best_score


def save_checkpoint(model: nn.Module, optim: optimizer.Optimizer, epoch_id: int, step: int, best_score: float, checkpoint_path: str = "checkpoint.tar"):
    torch.save({
        _MODEL_STATE_DICT: model.state_dict(),
        _OPTIMIZER_STATE_DICT: optim.state_dict(),
        _EPOCH: epoch_id,
        _STEP: step,
        _BEST_SCORE: best_score
    }, checkpoint_path)


def build_twohop_map(train_triples, rel_category=None, ignore_rel_names=None, id2rel=None,
                     scope="direct_only"):
    """
    rel_category: dict[r_id] -> "direct"|"path"|"hier" (optional but recommended)
    ignore_rel_names: set of strings to ignore (e.g., {"is_child_of","has_child"})
    id2rel: dict[r_id]->str to match ignore_rel_names
    scope: "all" | "direct_only" | "none"
    """
    if scope == "none":
        return defaultdict(set)

    out_by_hr = defaultdict(set)   # (h,r)->{t} direct
    in_to = defaultdict(list)      # x -> list[(h,r_in)]
    out_of = defaultdict(list)     # x -> list[(r_out,t2)]

    for h,r,t in train_triples:
        out_by_hr[(h,r)].add(t)
        in_to[t].append((h,r))
        out_of[h].append((r,t))

    ignore_r_ids = set()
    if ignore_rel_names and id2rel:
        for r_id, name in id2rel.items():
            if name in ignore_rel_names:
                ignore_r_ids.add(r_id)

    twohop = defaultdict(set)
    for x, incoming in in_to.items():
        outgoing = out_of.get(x, [])
        if not outgoing: 
            continue
        for (h, r_in) in incoming:
            if r_in in ignore_r_ids:
                continue
            if rel_category is not None:
                if scope == "direct_only" and rel_category.get(r_in) != "direct":
                    continue
            for (r_out, t2) in outgoing:
                if r_out in ignore_r_ids:
                    continue
                # exclude direct
                if t2 not in out_by_hr[(h, r_in)]:
                    twohop[(h, r_in)].add(t2)
    return twohop
