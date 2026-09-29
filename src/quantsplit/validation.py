"""Predeclared development checks for an actual loaded F/Q pair."""
import numpy as np
import torch
from .generation import Settings, generate_four
from .hf_backend import HFBackend, full_prefix_logits
from .sampling import PolicyPair, probabilities

TOLERANCES = {"bf16_cache_max_abs_logit": 0.25, "bf16_cache_tv": 0.001,
              "independent_baseline_max_abs_probability": 2e-6}


def validate_model_pair(f, q, prompt):
    cache = {}
    for name, model in [("F", f), ("Q", q)]:
        session, prefix = HFBackend(model).new_session(), list(prompt)
        logit_errors, tv_errors = [], []
        for step in range(5):
            if step:
                prefix.append(prompt[-1])
            cached, full = session.next_logits(prefix), full_prefix_logits(model, prefix)
            logit_errors.append(float(np.max(np.abs(cached-full))))
            tv_errors.append(float(np.abs(probabilities(cached).astype(np.float64)-probabilities(full)).sum()/2))
        cache[name] = {"max_abs_logit": max(logit_errors), "max_tv": max(tv_errors)}
    same = generate_four(HFBackend(f),HFBackend(f),prompt,"implementation-check",42,
                         Settings(max_new_tokens=8))
    identity = len({tuple(r.generated_ids) for r in same.values()}) == 1
    untruncated = generate_four(HFBackend(f),HFBackend(q),prompt,"implementation-check",42,
                                 Settings(max_new_tokens=8,top_p=1))
    p_one = (untruncated['FF'].generated_ids == untruncated['FQ'].generated_ids and
             untruncated['QQ'].generated_ids == untruncated['QF'].generated_ids)
    logits = {name: full_prefix_logits(model,prompt) for name,model in [('F',f),('Q',q)]}
    pair = PolicyPair.from_logits(logits['F'],logits['Q'])
    baseline_errors = {}
    for name in ['F','Q']:
        p = torch.softmax(torch.from_numpy(logits[name]).float()/0.6,dim=-1)
        order = torch.argsort(p,descending=True,stable=True)
        cdf = torch.cumsum(p[order],dim=0)
        crossing = int(torch.searchsorted(cdf,torch.tensor(.95),right=False))+1
        support = torch.zeros_like(p,dtype=torch.bool)
        support[order[:crossing]] = True
        baseline = torch.where(support,p,0)
        baseline /= baseline.sum()
        baseline_errors[name*2] = float(np.max(np.abs(baseline.numpy()-pair.distribution(name*2))))
    passed = (identity and p_one and
              all(v['max_abs_logit']<=TOLERANCES['bf16_cache_max_abs_logit'] and
                  v['max_tv']<=TOLERANCES['bf16_cache_tv'] for v in cache.values()) and
              max(baseline_errors.values())<=TOLERANCES['independent_baseline_max_abs_probability'])
    return {"passed":bool(passed),"same_weight_trace_identity":identity,"top_p_one_trace_identity":p_one,
            "cache":cache,"baseline_distribution_errors":baseline_errors,"tolerances":TOLERANCES,
            "scope":"short-prefix development check; full 32768-token budget still needs target-GPU validation"}


def validate_long_context_pair(f, q, prompt, target_length, chunk_size=512, progress=None):
    """Synthetic repeated-prefix stress; never a task-accuracy/throughput observation.

    Keep both caches resident, extend in chunks, then execute two one-token decode
    steps. Compare the final cached logits to an independent full-prefix forward.
    The caller records CUDA peaks, including both models and both resident caches.
    """
    if not prompt or target_length < len(prompt) + 2 or chunk_size < 1:
        raise ValueError("invalid context probe dimensions")
    if target_length > min(f.config.max_position_embeddings, q.config.max_position_embeddings):
        raise ValueError("synthetic context exceeds model limit")
    tokens = (list(prompt) * ((target_length + len(prompt) - 1) // len(prompt)))[:target_length]
    sessions = {"F": HFBackend(f).new_session(), "Q": HFBackend(q).new_session()}
    if sessions["F"].cache is sessions["Q"].cache:
        raise RuntimeError("context probe requires independent caches")
    end, logits = 0, {}
    while end < target_length:
        # Reserve the final two positions for one-token decode with the full caches.
        end = min(end + chunk_size, target_length - 2) if end < target_length - 2 else end + 1
        for name, session in sessions.items():
            logits[name] = session.next_logits(tokens[:end])
            if not np.isfinite(logits[name]).all():
                raise ArithmeticError(f"nonfinite {name} logits at context length {end}")
        if progress is not None:
            progress(end, target_length)
    errors = {}
    for name, model in [("F", f), ("Q", q)]:
        full = full_prefix_logits(model, tokens)
        if not np.isfinite(full).all():
            raise ArithmeticError(f"nonfinite {name} full-prefix logits")
        errors[name] = {"max_abs_logit": float(np.max(np.abs(logits[name] - full))),
                        "max_tv": float(np.abs(probabilities(logits[name]).astype(np.float64)
                                                - probabilities(full)).sum() / 2),
                        "cache_length": int(sessions[name].cache.get_seq_length())}
    passed = all(x["max_abs_logit"] <= TOLERANCES["bf16_cache_max_abs_logit"] and
                 x["max_tv"] <= TOLERANCES["bf16_cache_tv"] and
                 x["cache_length"] == target_length for x in errors.values())
    return {"passed": bool(passed), "synthetic": True, "scientific_H1_result": False,
            "target_cache_length": target_length, "chunk_size": chunk_size,
            "one_token_decode_steps": 2, "both_caches_resident": True,
            "cache": errors, "tolerances": TOLERANCES,
            "scope": "repeated development prompt; context/cache/memory only, not free generation or task quality"}
