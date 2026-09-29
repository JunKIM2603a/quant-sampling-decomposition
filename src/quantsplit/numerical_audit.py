"""Fixed-prefix implementation and policy sensitivity audit; never a GO gate."""
import numpy as np
from transformers import DynamicCache
from .cache_diagnostics import _forward_with_cache, _lengths, difference
from .hf_backend import HFBackend, full_prefix_logits
from .sampling import ARMS, PolicyPair, inverse_cdf, keyed_uniform, nucleus_mask, probabilities, restrict


def make_trace(model, prompt, question_id, plan, eos_ids):
    """A short F/FF path supplies identical inputs to all subsequent comparisons."""
    session, prefix, generated = HFBackend(model).new_session(), list(prompt), []
    stop = "cap"
    for step in range(plan["max_new_tokens"]):
        p = probabilities(session.next_logits(prefix), plan["temperature"])
        token = inverse_cdf(restrict(p, nucleus_mask(p, plan["top_p"])),
                            keyed_uniform(question_id, plan["seed"], step))
        generated.append(token)
        prefix.append(token)
        if token in eos_ids:
            stop = "eos"
            break
    # All observations precede sampling, including the last generated token.
    probes = sorted({s for s in plan["probe_steps"] if s < len(generated)} | {len(generated) - 1})
    return {"question_id": question_id, "prompt_ids": list(prompt), "generated_ids": generated,
            "stop_reason": stop, "probe_steps": probes}


def collect_trace(model, trace):
    """Check an explicit same-schedule cache at every step, full prefix at probes."""
    session = HFBackend(model).new_session()
    reference = DynamicCache(config=model.config)
    prefix, processed, points = list(trace["prompt_ids"]), 0, {}
    max_direct_logit, max_direct_tv = 0.0, 0.0
    selected = set(trace["probe_steps"])
    for step, next_token in enumerate(trace["generated_ids"]):
        cached = session.next_logits(prefix)
        direct, reference = _forward_with_cache(model, prefix[processed:], reference, processed, len(prefix))
        error = difference(cached, direct)
        max_direct_logit = max(max_direct_logit, error["max_abs_logit"])
        max_direct_tv = max(max_direct_tv, error["tv"])
        for cache in (session.cache, reference):
            if any(n != len(prefix) for n in _lengths(cache, model)):
                raise ValueError("incorrect cache layer length in numerical audit")
        if step in selected:
            full = full_prefix_logits(model, prefix)
            full_error = difference(cached, full)
            points[step] = {"cached": cached, "full": full, "error": full_error,
                            "prefix_tokens": len(prefix)}
        processed = len(prefix)
        prefix.append(next_token)
    return points, {"steps_checked": len(trace["generated_ids"]),
                    "all_logits_finite": True, "all_layer_lengths_correct": True,
                    "direct_max_abs_logit": max_direct_logit, "direct_max_tv": max_direct_tv,
                    "same_schedule_exact": max_direct_logit == 0.0 and max_direct_tv == 0.0}


def _tv(p, q):
    return float(np.abs(p.astype(np.float64) - q).sum() / 2)


def support_change(left_p, left_mask, right_p, right_mask, special_ids):
    changed, union = left_mask ^ right_mask, left_mask | right_mask
    return {"left_size": int(left_mask.sum()), "right_size": int(right_mask.sum()),
            "changed_tokens": int(changed.sum()),
            "jaccard": float((left_mask & right_mask).sum() / union.sum()),
            "left_mass_on_changed_tokens": float(left_p[changed].sum(dtype=np.float64)),
            "right_mass_on_changed_tokens": float(right_p[changed].sum(dtype=np.float64)),
            "special_membership": {str(t): [bool(left_mask[t]), bool(right_mask[t])] for t in special_ids}}


def policy_metrics(paths, temperature=.6, top_p=.95, special_ids=()):
    """Compare policies on one identical prefix. No trajectory effect estimate."""
    pairs = {mode: PolicyPair.from_logits(values["F"], values["Q"], temperature, top_p)
             for mode, values in paths.items()}
    bf16, full, fp32 = (pairs[k] for k in ("bf16_cached", "bf16_full", "fp32_cached"))
    def compare(left, right):
        return {
            "raw_tv": {"F": _tv(left.pf, right.pf), "Q": _tv(left.pq, right.pq)},
            "policy_tv": {arm: _tv(left.distribution(arm), right.distribution(arm)) for arm in ARMS},
            "support": {"F": support_change(left.pf, left.mf, right.pf, right.mf, special_ids),
                        "Q": support_change(left.pq, left.mq, right.pq, right.mq, special_ids)},
        }
    def gap(pair):
        return {"raw_F_Q_tv": _tv(pair.pf, pair.pq),
                "policy_FF_QQ_tv": _tv(pair.distribution("FF"), pair.distribution("QQ")),
                "policy_QQ_QF_tv": _tv(pair.distribution("QQ"), pair.distribution("QF")),
                "support_jaccard": float((pair.mf & pair.mq).sum() / (pair.mf | pair.mq).sum())}
    return {"bf16_cached_vs_full": compare(bf16, full),
            "bf16_cached_vs_fp32_cached": compare(bf16, fp32),
            "F_Q_same_prefix_difference": {"bf16": gap(bf16), "fp32": gap(fp32)}}


def summarize_points(points):
    """Descriptive statistics only; repeated positions are not independent samples."""
    if not points:
        raise ValueError("empty numerical audit")
    def stats(values):
        values = np.asarray(values, dtype=np.float64)
        return {"min": float(values.min()), "mean": float(values.mean()),
                "max": float(values.max()), "p95": float(np.quantile(values, .95))}
    result = {"probe_count": len(points), "review_required": True,
              "automatic_stage02_pass": False, "independent_sample_claim": False}
    for comparison in ("bf16_cached_vs_full", "bf16_cached_vs_fp32_cached"):
        result[comparison] = {
            "policy_tv": {a: stats([p["policies"][comparison]["policy_tv"][a] for p in points]) for a in ARMS},
            "support_jaccard": {m: stats([p["policies"][comparison]["support"][m]["jaccard"]
                                          for p in points]) for m in ("F", "Q")}}
    # Absolute side-by-side comparisons avoid dividing by a possibly tiny F-Q gap.
    result["precision_raw_tv_at_least_bf16_F_Q_tv_count"] = {
        m: sum(p["policies"]["bf16_cached_vs_fp32_cached"]["raw_tv"][m] >=
               p["policies"]["F_Q_same_prefix_difference"]["bf16"]["raw_F_Q_tv"] for p in points)
        for m in ("F", "Q")}
    return result
