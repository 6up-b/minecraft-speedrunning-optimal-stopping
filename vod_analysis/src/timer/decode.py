from __future__ import annotations
import itertools

import torch
import torch.nn.functional as F


def digits_to_timer_str(digs7: list[int]) -> str:
    return f"{digs7[0]}{digs7[1]}:{digs7[2]}{digs7[3]}.{digs7[4]}{digs7[5]}{digs7[6]}"


def digits_to_seconds(digs7: list[int]) -> float:
    mm = 10 * digs7[0] + digs7[1]
    ss = 10 * digs7[2] + digs7[3]
    ms = 100 * digs7[4] + 10 * digs7[5] + digs7[6]
    return 60 * mm + ss + ms / 1000.0


def _apply_constraints(
    logits_7x10: torch.Tensor,
    constrain_d2_0to5: bool = True,
    constrain_d0_0to5: bool = False,
) -> torch.Tensor:
    """
    Returns constrained logits with invalid digits set to -inf.
    """
    logits = logits_7x10.clone()
    neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)

    if constrain_d2_0to5:
        logits[2, 6:] = neg_inf

    if constrain_d0_0to5:
        logits[0, 6:] = neg_inf

    return logits


def greedy_constrained_decode_timer(
    logits_7x10: torch.Tensor,
    constrain_d2_0to5: bool = True,
    constrain_d0_0to5: bool = False,
):
    """
    Fast path: constrained greedy decode.
    Returns:
      timer_str, mean_conf, digits, confs, margins
    """
    if logits_7x10.ndim != 2 or logits_7x10.shape != (7, 10):
        raise ValueError(f"Expected logits shape [7,10], got {tuple(logits_7x10.shape)}")

    logits = _apply_constraints(
        logits_7x10,
        constrain_d2_0to5=constrain_d2_0to5,
        constrain_d0_0to5=constrain_d0_0to5,
    )

    probs = F.softmax(logits, dim=1)
    top2_vals, top2_idx = torch.topk(probs, k=2, dim=1)

    digits = top2_idx[:, 0].cpu().tolist()
    confs = top2_vals[:, 0].cpu().tolist()
    margins = (top2_vals[:, 0] - top2_vals[:, 1]).cpu().tolist()

    timer_str = digits_to_timer_str(digits)
    mean_conf = float(sum(confs) / len(confs))

    return timer_str, mean_conf, digits, confs, margins


def topk_constrained_decode_timer(
    logits_7x10: torch.Tensor,
    k: int = 3,
    constrain_d2_0to5: bool = True,
    constrain_d0_0to5: bool = False,
    ambiguous_margin: float = 0.15,
):
    """
    Stateless inference decoder.

    Strategy:
      1) greedy constrained decode first
      2) identify ambiguous positions by top1-top2 margin
      3) run top-k search ONLY on ambiguous positions

    Returns:
      timer_str, score, digits, info_dict
    """
    if logits_7x10.ndim != 2 or logits_7x10.shape != (7, 10):
        raise ValueError(f"Expected logits shape [7,10], got {tuple(logits_7x10.shape)}")

    logits = _apply_constraints(
        logits_7x10,
        constrain_d2_0to5=constrain_d2_0to5,
        constrain_d0_0to5=constrain_d0_0to5,
    )
    logp = F.log_softmax(logits, dim=1)

    greedy_str, greedy_conf, greedy_digits, greedy_digit_confs, margins = greedy_constrained_decode_timer(
        logits_7x10,
        constrain_d2_0to5=constrain_d2_0to5,
        constrain_d0_0to5=constrain_d0_0to5,
    )

    ambiguous_positions = [i for i, m in enumerate(margins) if m < ambiguous_margin]
    if len(ambiguous_positions) == 0:
        greedy_tensor = torch.tensor(greedy_digits, device=logp.device)
        greedy_score = float(logp[torch.arange(7, device=logp.device), greedy_tensor].sum().item())
        return greedy_str, greedy_score, greedy_digits, {
            "mode": "greedy",
            "mean_conf": greedy_conf,
            "digit_confs": greedy_digit_confs,
            "margins": margins,
            "ambiguous_positions": [],
        }

    candidates = []
    top_vals, top_idx = torch.topk(logp, k=min(k, 10), dim=1)
    top_vals = top_vals.cpu()
    top_idx = top_idx.cpu()

    for i in range(7):
        if i in ambiguous_positions:
            cands_i = [(int(top_idx[i, j].item()), float(top_vals[i, j].item())) for j in range(top_idx.shape[1])]
        else:
            cands_i = [(greedy_digits[i], float(logp[i, greedy_digits[i]].item()))]
        candidates.append(cands_i)

    best = None  # (base_logprob, digs)
    for combo in itertools.product(*candidates):
        digs = [d for (d, lp) in combo]
        base_lp = sum(lp for (d, lp) in combo)
        if best is None or base_lp > best[0]:
            best = (base_lp, digs)

    best_score, best_digits = best
    best_str = digits_to_timer_str(best_digits)

    return best_str, float(best_score), best_digits, {
        "mode": "topk",
        "mean_conf": greedy_conf,
        "digit_confs": greedy_digit_confs,
        "margins": margins,
        "ambiguous_positions": ambiguous_positions,
        "greedy_str": greedy_str,
        "greedy_digits": greedy_digits,
    }