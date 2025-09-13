import os, time, json, math, random
from typing import Dict, Tuple, Callable
import numpy as np
import torch
from torch.utils import tensorboard

# =========================
# Negative sampling helpers
# =========================

def sample_negatives_hr(h, r, t, all_entities, pos_set, twohop_map, k, keep_ratio, scope_enabled):
    """
    Balanced head/tail corruption with optional 2-hop filtering.
    Returns a list of k (h,r,t) negatives in ID space.
    """
    negs = []
    while len(negs) < k:
        if random.random() < 0.5:
            # tail corruption
            t_neg = random.choice(all_entities)
            if (h, r, t_neg) in pos_set:
                continue
            if scope_enabled and t_neg in twohop_map.get((h, r), ()):
                if random.random() > keep_ratio:
                    continue
            negs.append((h, r, t_neg))
        else:
            # head corruption
            h_neg = random.choice(all_entities)
            if (h_neg, r, t) in pos_set:
                continue
            negs.append((h_neg, r, t))
    return negs


def generate_relation_aware_negatives(
    pos: torch.Tensor,
    e2id: Dict[str,int],
    r2id: Dict[str,int],
    id2rel,  # unused but kept for API compatibility
    device: torch.device,
    num_negs: int,
    p_tail_id: torch.Tensor,
    twohop_map=None,
    all_true=None,
    multi_hop_aware: bool = True,
    twohop_keep_ratio: float = 0.2,
    twohop_scope: str = "direct_only",
):
    """
    Returns a [B*num_negs, 3] tensor of negatives (IDs).
    If multi-hop aware & maps provided, uses balanced head/tail sampling with 2-hop suppression.
    Otherwise falls back to Bernoulli corruption using p_tail_id.
    """
    B = pos.size(0)

    if multi_hop_aware and twohop_map is not None and all_true is not None:
        all_entities = list(range(len(e2id)))
        neg_list = []
        scope_enabled = (twohop_scope != "none")
        for i in range(B):
            h_id = pos[i, 0].item()
            r_id = pos[i, 1].item()
            t_id = pos[i, 2].item()
            negs = sample_negatives_hr(
                h_id, r_id, t_id,
                all_entities, all_true, twohop_map,
                num_negs, keep_ratio=twohop_keep_ratio, scope_enabled=scope_enabled
            )
            neg_list.extend(negs)
        return torch.tensor(neg_list, device=device, dtype=torch.long)

    # Bernoulli corruption (vectorized)
    rep = pos.repeat_interleave(num_negs, dim=0)                  # [B*num_negs, 3]
    r_flat = rep[:, 1]
    p = p_tail_id[r_flat]                                         # prob to corrupt tail
    coin = (torch.rand_like(p) > p).long()                        # 1=head, 0=tail
    rand_e = torch.randint(len(e2id), (B*num_negs,), device=device)
    rep[:, 0] = torch.where(coin == 1, rand_e, rep[:, 0])         # corrupt head
    rep[:, 2] = torch.where(coin == 0, rand_e, rep[:, 2])         # corrupt tail
    return rep


# =========================
# Loss weighting
# =========================

def apply_multi_hop_aware_weighting(loss, pos, direct_rel_ids, path_rel_ids, direct_weight: float):
    """
    Up-weight direct relations and down-weight path relations.
    Works for either scalar loss (already reduced) or per-sample loss vector.
    """
    if loss.dim() == 0:
        r_ids = pos[:, 1]
        direct_count = torch.isin(r_ids, direct_rel_ids).sum().float()
        path_count   = torch.isin(r_ids, path_rel_ids).sum().float()
        total_count  = len(r_ids)
        if total_count > 0:
            direct_ratio = direct_count / total_count
            path_ratio   = path_count   / total_count
            other_ratio  = 1.0 - direct_ratio - path_ratio
            weight_factor = direct_ratio * direct_weight + path_ratio * 0.5 + other_ratio * 1.0
            return loss * weight_factor
        return loss
    else:
        weights = torch.ones_like(loss)
        r_ids = pos[:, 1]
        direct_mask = torch.isin(r_ids, direct_rel_ids)
        path_mask   = torch.isin(r_ids, path_rel_ids)
        weights[direct_mask] = direct_weight
        weights[path_mask]   = 0.5
        return loss * weights


# =========================
# Results writer (migrated)
# =========================

def store_results_to_file(regular_tail, regular_head, tc_tail, tc_head, dataset_path, model_params, model_name, results_dir):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"results_{model_name}_{timestamp}.txt")
    os.makedirs(results_dir, exist_ok=True)
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{model_name} Model Evaluation Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Model: {model_name}\n\n")
        f.write("Model Parameters:\n" + "-" * 40 + "\n")
        f.write(f"model: {model_name}\n")
        for param, value in model_params.items():
            f.write(f"{param}: {value}\n")
        f.write("\nRegular Filtered Evaluation Results:\n" + "-" * 40 + "\n")
        f.write(f"Tail Prediction:\n")
        f.write(f"  H@1:  {regular_tail['H@1']:.4f}\n")
        f.write(f"  H@3:  {regular_tail['H@3']:.4f}\n")
        f.write(f"  H@10: {regular_tail['H@10']:.4f}\n")
        f.write(f"  MRR:  {regular_tail['MRR']:.4f}\n")
        f.write(f"  Queries: {regular_tail['num_queries']}\n\n")
        f.write(f"Head Prediction:\n")
        f.write(f"  H@1:  {regular_head['H@1']:.4f}\n")
        f.write(f"  H@3:  {regular_head['H@3']:.4f}\n")
        f.write(f"  H@10: {regular_head['H@10']:.4f}\n")
        f.write(f"  MRR:  {regular_head['MRR']:.4f}\n")
        f.write(f"  Queries: {regular_head['num_queries']}\n")
        avg_hits_1 = (regular_tail["H@1"] + regular_head["H@1"]) / 2
        avg_hits_3 = (regular_tail["H@3"] + regular_head["H@3"]) / 2
        avg_hits_10 = (regular_tail["H@10"] + regular_head["H@10"]) / 2
        avg_mrr = (regular_tail["MRR"] + regular_head["MRR"]) / 2
        f.write(f"\nRegular Average:\n")
        f.write(f"  H@1:  {avg_hits_1:.4f}\n")
        f.write(f"  H@3:  {avg_hits_3:.4f}\n")
        f.write(f"  H@10: {avg_hits_10:.4f}\n")
        f.write(f"  MRR:  {avg_mrr:.4f}\n\n")
        f.write("Type-Constrained Filtered Evaluation Results:\n" + "-" * 40 + "\n")
        f.write(f"Tail Prediction:\n")
        f.write(f"  H@1:  {tc_tail['H@1']:.4f}\n")
        f.write(f"  H@3:  {tc_tail['H@3']:.4f}\n")
        f.write(f"  H@10: {tc_tail['H@10']:.4f}\n")
        f.write(f"  MRR:  {tc_tail['MRR']:.4f}\n")
        f.write(f"  Queries: {tc_tail['num_queries']}\n\n")
        f.write(f"Head Prediction:\n")
        f.write(f"  H@1:  {tc_head['H@1']:.4f}\n")
        f.write(f"  H@3:  {tc_head['H@3']:.4f}\n")
        f.write(f"  H@10: {tc_head['H@10']:.4f}\n")
        f.write(f"  MRR:  {tc_head['MRR']:.4f}\n")
        f.write(f"  Queries: {tc_head['num_queries']}\n")
        avg_hits_1_tc = (tc_tail["H@1"] + tc_head["H@1"]) / 2
        avg_hits_3_tc = (tc_tail["H@3"] + tc_head["H@3"]) / 2
        avg_hits_10_tc = (tc_tail["H@10"] + tc_head["H@10"]) / 2
        avg_mrr_tc = (tc_tail["MRR"] + tc_head["MRR"]) / 2
        f.write(f"\nType-Constrained Average:\n")
        f.write(f"  H@1:  {avg_hits_1_tc:.4f}\n")
        f.write(f"  H@3:  {avg_hits_3_tc:.4f}\n")
        f.write(f"  H@10: {avg_hits_10_tc:.4f}\n")
        f.write(f"  MRR:  {avg_mrr_tc:.4f}\n\n")
        f.write("=" * 80 + "\n")
    print(f"Results saved to: {results_file}")


# =========================
# Training & evaluation
# =========================

# --- TYPE-AWARE NEGATIVE SAMPLING (drop-in) ---
import random

def sample_type_constrained_negatives(pos, num_negs, allowed_heads, allowed_tails, p_tail_id, E, all_true, device):
    """
    For each positive (h,r,t), draw negatives that respect relation's head/tail type ranges.
    Uses Bernoulli coin p_tail_id[r] to decide head vs tail corruption.
    Guarantees B*num_negs negatives; skips triples in all_true.
    """
    B = pos.size(0)
    out = []
    for i in range(B):
        h, r, t = map(int, pos[i].tolist())
        need = num_negs
        while need > 0:
            corrupt_tail = (random.random() < float(p_tail_id[r]))
            if corrupt_tail:
                pool = allowed_tails.get(r, None)
                if pool:
                    t_neg = random.choice(tuple(pool))
                else:
                    t_neg = random.randrange(E)
                if t_neg == t or (h, r, t_neg) in all_true:
                    continue
                out.append((h, r, t_neg))
            else:
                pool = allowed_heads.get(r, None)
                if pool:
                    h_neg = random.choice(tuple(pool))
                else:
                    h_neg = random.randrange(E)
                if h_neg == h or (h_neg, r, t) in all_true:
                    continue
                out.append((h_neg, r, t))
            need -= 1
    return torch.tensor(out, device=device, dtype=torch.long)

def train_and_eval(
    model,
    FLAGS,
    e2id, r2id,
    train_triples, valid_triples, test_triples,
    train_set, validation_set, test_set,
    tails_by_hr, heads_by_rt,
    rel_category, HIER_REL_IDS,
    allowed_heads, allowed_tails,
    etype_tensor,
    twohop_map,
    p_tail_id,
    *,
    save_ckpt: Callable,
    load_ckpt: Callable,
    evaluate_all: Callable,
):
    """
    Runs training with periodic validation and final testing.
    Expects model scores to follow HIGHER = BETTER semantics.
    """
    # Device & optimizer
    use_gpu = not FLAGS.nouse_gpu
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=1e-4)

    # AMP & RNG
    import numpy as _np
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.manual_seed(FLAGS.seed); random.seed(FLAGS.seed); _np.random.seed(FLAGS.seed)
    scaler = torch.cuda.amp.GradScaler(enabled=not FLAGS.nouse_gpu)

    writer = tensorboard.SummaryWriter(log_dir=FLAGS.tensorboard_log_dir)
    device = next(model.parameters()).device

    # Relation ID tensors for weighting
    DIRECT_REL_IDS = torch.tensor([r_id for r_id, cat in rel_category.items() if cat == "direct"], device=device)
    PATH_REL_IDS   = torch.tensor([r_id for r_id, cat in rel_category.items() if cat == "path"], device=device)

    # Training bookkeeping
    best_tc_mrr = -math.inf
    best_epoch = -1
    no_improve = 0
    start_epoch_id = 1
    step = 0

    # Resume
    if getattr(FLAGS, "resume", ""):
        start_epoch_id, prev_best = load_ckpt(
            FLAGS.resume, model, optimizer,
            map_location="cpu" if FLAGS.nouse_gpu else "cuda"
        )
        if prev_best is not None:
            best_tc_mrr = prev_best
            best_epoch = start_epoch_id
        print(f"[RESUME] start_epoch={start_epoch_id}, prev_best_tc_mrr={best_tc_mrr:.3f}")

    print(model)

    # Dataloader (use the one passed in)
    from torch.utils import data as torch_data
    train_generator = torch_data.DataLoader(
        train_set,
        batch_size=FLAGS.batch_size,
        num_workers=4 if not FLAGS.nouse_gpu else 0,
        pin_memory=not FLAGS.nouse_gpu
    )

    # =================
    # TRAINING LOOP
    # =================
    for epoch_id in range(start_epoch_id, FLAGS.epochs + 1):
        print("Starting epoch: ", epoch_id)
        loss_impacting_samples_count = 0
        samples_count = 0
        model.train()
        num_negs = FLAGS.neg_ratio

        for h, r, t in train_generator:
            h, r, t = h.to(device), r.to(device), t.to(device)
            pos = torch.stack([h, r, t], dim=1)  # [B,3]
            B = h.size(0)

            # NEGATIVE SAMPLING
            all_true = set(train_triples) | set(valid_triples) | set(test_triples)
            if FLAGS.relation_aware_negatives:
                neg = sample_type_constrained_negatives(
                    pos, num_negs,
                    allowed_heads, allowed_tails,
                    p_tail_id, E=len(e2id), all_true=all_true, device=device
                )
            else:
                # (keep your current Bernoulli fallback here unchanged)
                rep = pos.repeat_interleave(num_negs, dim=0)
                r_flat = rep[:, 1]
                p = p_tail_id[r_flat]
                coin = (torch.rand_like(p) > p).long()
                rand_e = torch.randint(len(e2id), (pos.size(0)*num_negs,), device=device)
                rep[:, 0] = torch.where(coin==1, rand_e, rep[:, 0])
                rep[:, 2] = torch.where(coin==0, rand_e, rep[:, 2])
                neg = rep

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=not FLAGS.nouse_gpu):
                # Scores (higher = better)
                s_pos = model.score_triples(pos[:, 0], pos[:, 1], pos[:, 2])   # [B]
                s_neg = model.score_triples(neg[:, 0], neg[:, 1], neg[:, 2])   # [B*num_negs]
                s_neg = s_neg.view(B, num_negs)                                 # [B, num_negs]

                # Loss
                if FLAGS.loss == 'margin':
                    s_pos_expanded = s_pos.unsqueeze(1).expand(-1, num_negs)    # [B, num_negs]
                    y = torch.ones_like(s_pos_expanded)
                    loss = torch.nn.MarginRankingLoss(margin=FLAGS.margin)(s_pos_expanded, s_neg, y)
                else:
                    # Actually use self-adversarial weighting with BCE
                    import torch.nn.functional as F
                    
                    # After you compute: s_pos: [B], s_neg: [B, K]
                    pos_loss = F.binary_cross_entropy_with_logits(s_pos, torch.ones_like(s_pos), reduction='none')
                    neg_loss = F.binary_cross_entropy_with_logits(s_neg, torch.zeros_like(s_neg), reduction='none')  # [B,K]

                    if FLAGS.self_adversarial:
                        # temperature can be a flag if you like; 1.0 works well on KGE
                        with torch.no_grad():
                            w = torch.softmax(s_neg, dim=1)  # [B,K]
                        neg_loss = (w * neg_loss).sum(dim=1)  # [B]
                    else:
                        neg_loss = neg_loss.mean(dim=1)       # [B]

                    loss = (pos_loss + neg_loss)              # [B]

                # Weighting
                if FLAGS.multi_hop_aware:
                    loss = apply_multi_hop_aware_weighting(loss, pos, DIRECT_REL_IDS, PATH_REL_IDS, FLAGS.direct_relation_weight)
                
                # keep your multi-hop weighting hook here (it handles vector or scalar)
                loss = loss.mean()

                # Optional 2-hop margin for direct relations
                if FLAGS.multi_hop_aware and FLAGS.twohop_weight > 0.0 and FLAGS.twohop_scope != "none":
                    batch_pos = [(h[i].item(), r[i].item(), t[i].item()) for i in range(B)]
                    margin_terms = []
                    for (h_id, r_id, t_id) in batch_pos:
                        if rel_category.get(r_id) != "direct":
                            continue
                        t2_list = list(twohop_map.get((h_id, r_id), []))[:2]
                        if not t2_list:
                            continue
                        H = torch.tensor([h_id]*len(t2_list), device=device)
                        R = torch.tensor([r_id]*len(t2_list), device=device)
                        T_pos = torch.tensor([t_id]*len(t2_list), device=device)
                        T2 = torch.tensor(t2_list, device=device)
                        s_pos2 = model.score_triples(H, R, T_pos)
                        s_neg2 = model.score_triples(H, R, T2)
                        margin_terms.append(torch.relu(s_neg2 - s_pos2 + FLAGS.twohop_margin))
                    if margin_terms:
                        twohop_loss = FLAGS.twohop_weight * torch.cat(margin_terms).mean()
                        loss = loss + twohop_loss
                        if step % 100 == 0:
                            print(f"[2HOP] Step {step}: margin_terms={len(margin_terms)}, twohop_loss={twohop_loss.item():.4f}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1

            loss_impacting_samples_count += 1 if loss.item() > 0 else 0
            samples_count += B

            # Logs
            writer.add_scalar('Loss/train', loss.detach().cpu().item(), global_step=step)
            writer.add_scalar('Score/positive', s_pos.mean().detach().cpu().item(), global_step=step)
            writer.add_scalar('Score/negative', s_neg.mean().detach().cpu().item(), global_step=step)

        # Epoch-level metric
        if samples_count > 0:
            writer.add_scalar('Metrics/loss_impacting_samples',
                              loss_impacting_samples_count / samples_count * 100.0,
                              global_step=epoch_id)

        # ================
        # VALIDATION
        # ================
        if epoch_id % FLAGS.eval_every == 0:
            model.eval()
            # quick score separation diagnostic
            import numpy as np
            sample_pos = random.sample(valid_triples, min(200, len(valid_triples)))
            pos_scores, neg_scores = [], []
            E = len(e2id)
            true_set = set(train_triples) | set(valid_triples) | set(test_triples)
            for (hh, rr, tt) in sample_pos:
                pos_scores.append(model.score_triples(torch.tensor([hh], device=device),
                                                      torch.tensor([rr], device=device),
                                                      torch.tensor([tt], device=device)).item())
                # uniform tail negative
                while True:
                    tneg = random.randrange(E)
                    if (hh, rr, tneg) not in true_set:
                        break
                neg_scores.append(model.score_triples(torch.tensor([hh], device=device),
                                                      torch.tensor([rr], device=device),
                                                      torch.tensor([tneg], device=device)).item())
            print(f"[VAL] pos_mean={np.mean(pos_scores):.3f} neg_mean={np.mean(neg_scores):.3f} "
                  f"sep={np.mean(pos_scores)-np.mean(neg_scores):.3f}")

            # unified evaluation (uses higher=better)
            eval_results = evaluate_all(
                model, validation_set, len(e2id), device,
                tails_by_hr, heads_by_rt, HIER_REL_IDS,
                etype_tensor, allowed_tails, allowed_heads, FLAGS.eval_filtered
            )

            m_f  = eval_results["filtered"]["average"]
            m_tc = eval_results["tc_filtered"]["average"]
            print(f"[VAL] ep={epoch_id}  Filt: MRR={m_f['MRR']:.3f} H@10={m_f['H@10']:.3f} | "
                  f"TC-Filt: MRR={m_tc['MRR']:.3f} H@10={m_tc['H@10']:.3f}")

            # early stop on TC-MRR
            tc_mrr = m_tc["MRR"]
            improved = tc_mrr > best_tc_mrr + FLAGS.early_stop_delta

            # save 'last'
            save_ckpt({
                "epoch": epoch_id,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "best_tc_mrr": best_tc_mrr,
            }, os.path.join(FLAGS.checkpoint_dir, FLAGS.model, "last.pt"))

            if improved:
                best_tc_mrr = tc_mrr
                best_epoch = epoch_id
                no_improve = 0
                save_ckpt({
                    "epoch": epoch_id,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "best_tc_mrr": best_tc_mrr,
                }, os.path.join(FLAGS.checkpoint_dir, FLAGS.model, "best.pt"))
                os.makedirs(FLAGS.results_dir, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                with open(os.path.join(FLAGS.results_dir, f"results_{FLAGS.model}_{stamp}.txt"), "w") as f:
                    f.write(json.dumps(eval_results, indent=2))
            else:
                no_improve += 1

            if no_improve >= FLAGS.early_stop_patience:
                print(f"[EARLY STOP] No TC-MRR improvement ≥ {FLAGS.early_stop_delta} "
                      f"for {FLAGS.early_stop_patience} validations. "
                      f"Best at epoch {best_epoch} (TC-MRR={best_tc_mrr:.3f}).")
                break

    # ================
    # FINAL TEST
    # ================
    print(f"[DONE] Best epoch {best_epoch} | best TC-MRR={best_tc_mrr:.3f}")
    best_ckpt = os.path.join(FLAGS.checkpoint_dir, FLAGS.model, "best.pt")
    last_ckpt = os.path.join(FLAGS.checkpoint_dir, FLAGS.model, "last.pt")
    load_ckpt(best_ckpt if os.path.exists(best_ckpt) else last_ckpt,
              model, optimizer, map_location="cpu" if FLAGS.nouse_gpu else "cuda")
    model = model.to(device).eval()

    print("=== FINAL TEST RESULTS ===")
    test_eval_results = evaluate_all(
        model, test_set, len(e2id), device,
        tails_by_hr, heads_by_rt, HIER_REL_IDS,
        etype_tensor, allowed_tails, allowed_heads, FLAGS.eval_filtered
    )

    # Pretty prints (optional)
    tail_results     = test_eval_results["filtered"]["tail"]
    head_results     = test_eval_results["filtered"]["head"]
    tail_results_tc  = test_eval_results["tc_filtered"]["tail"]
    head_results_tc  = test_eval_results["tc_filtered"]["head"]

    print(f"Regular - Tail: H@1={tail_results['H@1']:.3f}, H@3={tail_results['H@3']:.3f}, H@10={tail_results['H@10']:.3f}, MRR={tail_results['MRR']:.3f}")
    print(f"Regular - Head: H@1={head_results['H@1']:.3f}, H@3={head_results['H@3']:.3f}, H@10={head_results['H@10']:.3f}, MRR={head_results['MRR']:.3f}")
    print(f"Regular Average: H@1={test_eval_results['filtered']['average']['H@1']:.3f}, H@3={test_eval_results['filtered']['average']['H@3']:.3f}, H@10={test_eval_results['filtered']['average']['H@10']:.3f}, MRR={test_eval_results['filtered']['average']['MRR']:.3f}")

    print(f"Type-Constrained - Tail: H@1={tail_results_tc['H@1']:.3f}, H@3={tail_results_tc['H@3']:.3f}, H@10={tail_results_tc['H@10']:.3f}, MRR={tail_results_tc['MRR']:.3f}")
    print(f"Type-Constrained - Head: H@1={head_results_tc['H@1']:.3f}, H@3={head_results_tc['H@3']:.3f}, H@10={head_results_tc['H@10']:.3f}, MRR={head_results_tc['MRR']:.3f}")
    print(f"Type-Constrained Average: H@1={test_eval_results['tc_filtered']['average']['H@1']:.3f}, H@3={test_eval_results['tc_filtered']['average']['H@3']:.3f}, H@10={test_eval_results['tc_filtered']['average']['H@10']:.3f}, MRR={test_eval_results['tc_filtered']['average']['MRR']:.3f}")

    # Optional: comprehensive ranking + calibration
    from metric import evaluate_ranking, fit_calibrator
    # Build id2entity for proper entity type detection
    id2entity = {i: s for s, i in e2id.items()}
    
    metrics = evaluate_ranking(
        model, test_triples, all_true=set(train_triples)|set(valid_triples)|set(test_triples),
        e2id=e2id, r2id=r2id,
        filtered=FLAGS.eval_filtered,
        per_relation=FLAGS.eval_per_relation,
        train_triples=train_triples,
        type_constrained=FLAGS.eval_type_constrained,
        id2entity=id2entity,
        rel_category=rel_category
    )
    if "per_category" in metrics:
        print("\n--- Per-Category Results ---")
        for category, results in metrics["per_category"].items():
            print(f"{category}: MRR={results['MRR']:.3f}, H@1={results['Hits@1']:.3f}, H@3={results['Hits@3']:.3f}, H@10={results['Hits@10']:.3f}, count={results['count']}")

    if FLAGS.calibration != "none":
        print(f"Fitting {FLAGS.calibration} calibrator on validation set...")
        calibrator = fit_calibrator(model, valid_triples, method=FLAGS.calibration)
        print("Calibrator fitted successfully.")
    else:
        calibrator = None
        print("No calibration requested.")

    # Persist results (same format as before)
    store_results_to_file(
        regular_tail=tail_results,
        regular_head=head_results,
        tc_tail=tail_results_tc,
        tc_head=head_results_tc,
        dataset_path=FLAGS.dataset_path,
        model_params={
            'embedding_dim': FLAGS.embedding_dim,
            'margin': FLAGS.margin,
            'norm': FLAGS.norm,
            'lr': FLAGS.lr,
            'epochs': FLAGS.epochs,
            'batch_size': FLAGS.batch_size,
            'neg_ratio': FLAGS.neg_ratio,
            'self_adversarial': FLAGS.self_adversarial
        },
        model_name=FLAGS.model,
        results_dir=FLAGS.results_dir
    )