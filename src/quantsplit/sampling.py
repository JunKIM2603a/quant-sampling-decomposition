"""FP32 policies; no second nucleus filtering of a hybrid distribution."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np

ARMS = ("FF", "QQ", "QF", "FQ")
RNG_VERSION = "quantsplit-sha256-uniform-v1"


def probabilities(logits, temperature=0.6):
    x = np.asarray(logits, dtype=np.float32)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("logits must be a finite nonempty vector")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    # Subtract before dividing to avoid overflow for finite large logits.
    with np.errstate(over="ignore", invalid="raise"):
        x = (x - x.max()) / np.float32(temperature)
    weights = np.exp(x, dtype=np.float32)
    return weights / weights.sum(dtype=np.float32)


def _check_probabilities(p):
    p = np.asarray(p, dtype=np.float32)
    if (p.ndim != 1 or not p.size or not np.isfinite(p).all()
            or (p < 0).any() or not np.isclose(p.sum(), 1, atol=2e-6, rtol=0)):
        raise ValueError("expected a normalized finite probability vector")
    return p


def nucleus_mask(p, top_p=0.95):
    p = _check_probabilities(p)
    if not np.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must lie in (0, 1]")
    if top_p == 1:
        return np.ones(p.size, dtype=bool)
    # Original order is ascending token ID; stable sort preserves ties.
    order = np.argsort(-p, kind="stable")
    cumulative = np.cumsum(p[order], dtype=np.float32)
    count = min(int(np.searchsorted(cumulative, np.float32(top_p), side="left")) + 1, p.size)
    mask = np.zeros(p.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def restrict(p, mask):
    p = _check_probabilities(p)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != p.shape or not mask.any():
        raise ValueError("invalid or empty support")
    weights = np.where(mask, p, np.float32(0))
    mass = weights.sum(dtype=np.float32)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("zero/nonfinite retained probability mass")
    return weights / mass


@dataclass(frozen=True)
class PolicyPair:
    pf: np.ndarray
    pq: np.ndarray
    mf: np.ndarray
    mq: np.ndarray

    @classmethod
    def from_logits(cls, f, q, temperature=0.6, top_p=0.95):
        pf, pq = probabilities(f, temperature), probabilities(q, temperature)
        if pf.shape != pq.shape:
            raise ValueError("F and Q vocabulary shapes differ")
        return cls(pf, pq, nucleus_mask(pf, top_p), nucleus_mask(pq, top_p))

    def distribution(self, arm):
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        return restrict(self.pf if arm[0] == "F" else self.pq,
                        self.mf if arm[1] == "F" else self.mq)

    def diagnostics(self, special_ids=()):
        intersection = self.mf & self.mq
        union = self.mf | self.mq
        a = float(self.pq[self.mq].sum(dtype=np.float32))
        b = float(self.pq[self.mf].sum(dtype=np.float32))
        c = float(self.pq[intersection].sum(dtype=np.float32))
        # Use FP64 for reported summaries only, after FP32 policy construction.
        qq, qf = self.distribution("QQ"), self.distribution("QF")
        result = {"n_f": int(self.mf.sum()), "n_q": int(self.mq.sum()),
                  "jaccard": float(intersection.sum() / union.sum()),
                  "q_mass_mq": a, "q_mass_mf": b,
                  "f_mass_mq": float(self.pf[self.mq].sum(dtype=np.float32)),
                  "f_mass_mf": float(self.pf[self.mf].sum(dtype=np.float32)),
                  "tv_qq_qf": float(np.abs(qq.astype(np.float64) - qf).sum() / 2),
                  "tv_identity": 1 - c / max(a, b)}
        for token in special_ids:
            if not 0 <= token < len(self.pf):
                raise ValueError("special token outside vocabulary")
            result[f"special_{token}_in_f"] = int(self.mf[token])
            result[f"special_{token}_in_q"] = int(self.mq[token])
        return result


def keyed_uniform(question_id: str, seed: int, step: int):
    if not isinstance(question_id, str) or not question_id:
        raise ValueError("question_id must be a nonempty string")
    if not isinstance(seed, int) or not isinstance(step, int) or step < 0:
        raise ValueError("seed and nonnegative step must be integers")
    payload = json.dumps([RNG_VERSION, question_id, seed, step],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    bits = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") >> 11
    return bits / 2**53  # Exactly representable binary64 value in [0, 1).


def inverse_cdf(p, uniform):
    p = _check_probabilities(p)
    if not np.isfinite(uniform) or not 0 <= uniform < 1:
        raise ValueError("uniform must lie in [0, 1)")
    cdf = np.cumsum(p, dtype=np.float32)
    cdf /= cdf[-1]  # Repair total accumulation drift, without adding tail mass.
    # Cast only for comparing a 53-bit uniform; arithmetic above remains FP32.
    token = int(np.searchsorted(cdf.astype(np.float64), uniform, side="right"))
    if token >= p.size or p[token] <= 0:
        raise ArithmeticError("invalid inverse-CDF sample")
    return token
