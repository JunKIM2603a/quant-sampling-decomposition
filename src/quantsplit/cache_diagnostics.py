"""Short cache diagnostics; no tolerance selection or scientific evaluation."""
from contextlib import contextmanager, nullcontext
import numpy as np
import torch
from transformers import DynamicCache
from .hf_backend import HFBackend, full_prefix_logits
from .sampling import probabilities
from .validation import TOLERANCES


MODES = (
    "bf16_auto",
    "bf16_math",
    "bf16_math_full_reduction",
    "fp32_math",
)


@contextmanager
def diagnostic_mode(name):
    """Restore process settings even on failure; FP32 casting is caller-owned."""
    if name not in MODES:
        raise ValueError(f"unknown diagnostic mode: {name}")
    from torch.nn.attention import SDPBackend, sdpa_kernel
    matmul = torch.backends.cuda.matmul
    previous = (matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                matmul.allow_bf16_reduced_precision_reduction,
                torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
    try:
        matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if name in ("bf16_math_full_reduction", "fp32_math"):
            matmul.allow_bf16_reduced_precision_reduction = False
        if name != "bf16_auto":
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        context = nullcontext() if name == "bf16_auto" else sdpa_kernel(SDPBackend.MATH)
        with context:
            yield {
                "sdpa_policy": "automatic" if name == "bf16_auto" else "math_only",
                "matmul_tf32": matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "bf16_reduced_precision_reduction": matmul.allow_bf16_reduced_precision_reduction,
                "math_sdp_reduced_precision_reduction":
                    torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
            }
    finally:
        matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous[:2]
        matmul.allow_bf16_reduced_precision_reduction = previous[2]
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous[3])


def difference(left, right):
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ArithmeticError("nonfinite diagnostic logits")
    return {
        "max_abs_logit": float(np.max(np.abs(left - right))),
        "tv": float(np.abs(probabilities(left).astype(np.float64)
                           - probabilities(right)).sum() / 2),
    }


def _forward_with_cache(model, tokens, cache, start, total):
    """Direct HF reference, independent of HFSession, with explicit positions."""
    device = model.get_input_embeddings().weight.device
    positions = torch.arange(start, total, device=device)
    with torch.inference_mode():
        out = model(
            input_ids=torch.tensor([tokens], dtype=torch.long, device=device),
            attention_mask=torch.ones((1, total), dtype=torch.long, device=device),
            position_ids=positions.unsqueeze(0), cache_position=positions,
            past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1,
        )
    return out.logits[0, -1].float().cpu().numpy(), out.past_key_values


def _lengths(cache, model):
    return [int(cache.get_seq_length(i)) for i in range(model.config.num_hidden_layers)]


def diagnose_model(model, prompt):
    """Match validate_model_pair's prefill plus four repeated-last-token steps.

    The direct reference shares Transformers internals: agreement cannot prove
    the absence of a shared backend bug. No F-vs-Q or task outputs are measured.
    """
    if not prompt:
        raise ValueError("empty diagnostic prompt")
    session = HFBackend(model).new_session()
    direct_cache = DynamicCache(config=model.config)
    prefix, steps, previous_length = list(prompt), [], 0
    for step in range(5):
        if step:
            prefix.append(prompt[-1])
        cached = session.next_logits(prefix)
        full = full_prefix_logits(model, prefix)
        direct, direct_cache = _forward_with_cache(
            model, prefix[previous_length:], direct_cache, previous_length, len(prefix))
        fresh, fresh_cache = _forward_with_cache(
            model, prefix, DynamicCache(config=model.config), 0, len(prefix))
        repeated = full_prefix_logits(model, prefix)
        lengths = {"repository": _lengths(session.cache, model),
                   "direct": _lengths(direct_cache, model),
                   "fresh": _lengths(fresh_cache, model)}
        if any(n != len(prefix) for values in lengths.values() for n in values):
            raise ValueError(f"unexpected cache layer lengths: {lengths}")
        steps.append({
            "step": step, "prefix_tokens": len(prefix), "cache_lengths": lengths,
            "cached_vs_full": difference(cached, full),
            "repository_vs_direct": difference(cached, direct),
            "fresh_cache_vs_full": difference(fresh, full),
            "full_repeat": difference(repeated, full),
        })
        del fresh_cache
        previous_length = len(prefix)
    comparisons = ("cached_vs_full", "repository_vs_direct", "fresh_cache_vs_full", "full_repeat")
    maxima = {key: {"max_abs_logit": max(s[key]["max_abs_logit"] for s in steps),
                    "max_tv": max(s[key]["tv"] for s in steps)} for key in comparisons}
    cache = maxima["cached_vs_full"]
    return {"steps": steps, "maxima": maxima,
            "meets_existing_cache_tolerance":
                cache["max_abs_logit"] <= TOLERANCES["bf16_cache_max_abs_logit"] and
                cache["max_tv"] <= TOLERANCES["bf16_cache_tv"],
            "all_logits_finite": True, "all_cache_lengths_correct": True}
