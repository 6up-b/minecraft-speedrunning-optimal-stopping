from __future__ import annotations
import itertools
import torch
import torch.nn.functional as F


def digits_to_timer_str(digs7: list[int]) -> str:
    # digs7 = [d0..d6]
    return f"{digs7[0]}{digs7[1]}:{digs7[2]}{digs7[3]}.{digs7[4]}{digs7[5]}{digs7[6]}"


def topk_constrained_decode_timer(
    logits_7x10: torch.Tensor,
    k: int = 3,
    constrain_d2_0to5: bool = True,
    constrain_d0_0to5: bool = False,
) -> tuple[str, float, list[int]]:
    """
    logits_7x10: torch.Tensor [7,10] (unnormalized logits) for digit slots d0..d6
    Returns: (best_timer_string, best_logprob_sum, best_digits_list)

    Constraints:
      - d2 (tens of seconds) must be 0..5 (Minecraft timer format)
      - optional d0 (tens of minutes) 0..5 (usually safe; disable if you might exceed 59 minutes)
    """
    if logits_7x10.ndim != 2 or logits_7x10.shape != (7, 10):
        raise ValueError(f"Expected logits shape [7,10], got {tuple(logits_7x10.shape)}")

    logp = F.log_softmax(logits_7x10, dim=1)  # [7,10]

    # top-k per position
    top_vals, top_idx = torch.topk(logp, k=min(k, 10), dim=1)  # each is [7,k]
    top_vals = top_vals.cpu()
    top_idx = top_idx.cpu()

    # build candidate lists per digit slot
    candidates = []
    for i in range(7):
        cands_i = []
        for j in range(top_idx.shape[1]):
            d = int(top_idx[i, j].item())
            lp = float(top_vals[i, j].item())
            cands_i.append((d, lp))
        candidates.append(cands_i)

    # Apply constraints by filtering candidates for constrained positions
    if constrain_d2_0to5:
        candidates[2] = [(d, lp) for (d, lp) in candidates[2] if 0 <= d <= 5]
        if not candidates[2]:
            # fallback to 0..5 from full distribution if top-k misses it
            all_lp = logp[2].cpu().numpy().tolist()
            candidates[2] = [(d, all_lp[d]) for d in range(6)]

    if constrain_d0_0to5:
        candidates[0] = [(d, lp) for (d, lp) in candidates[0] if 0 <= d <= 5]
        if not candidates[0]:
            all_lp = logp[0].cpu().numpy().tolist()
            candidates[0] = [(d, all_lp[d]) for d in range(6)]

    # brute-force over k^7 (k=3 => 2187) fast enough
    best = None  # (sum_lp, digits_list)
    for combo in itertools.product(*candidates):
        digs = [d for (d, lp) in combo]
        s = sum(lp for (d, lp) in combo)
        if best is None or s > best[0]:
            best = (s, digs)

    best_lp, best_digits = best
    return digits_to_timer_str(best_digits), float(best_lp), best_digits

'''

with torch.no_grad():
    logits = model(x)  # x: [7,3,32,32]  logits: [7,10]

timer_str, score, digits = topk_constrained_decode_timer(
    logits, k=3, constrain_d2_0to5=True, constrain_d0_0to5=False
)

'''