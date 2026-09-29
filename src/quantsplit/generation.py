"""Independent rollouts with two separate, synchronized sessions per arm."""
from dataclasses import dataclass
from typing import Protocol, Sequence
import numpy as np
from .sampling import ARMS, PolicyPair, inverse_cdf, keyed_uniform


class Session(Protocol):
    def next_logits(self, prefix: Sequence[int]) -> np.ndarray: ...


class Backend(Protocol):
    def new_session(self) -> Session: ...


@dataclass(frozen=True)
class Settings:
    max_new_tokens: int = 32768
    temperature: float = 0.6
    top_p: float = 0.95
    eos_ids: tuple[int, ...] = ()
    diagnostic_ids: tuple[int, ...] = ()

    def __post_init__(self):
        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens < 1:
            raise ValueError("positive integer max_new_tokens required")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("invalid temperature")
        if not 0 < self.top_p <= 1:
            raise ValueError("invalid top_p")
        if any(not isinstance(t, int) or t < 0 for t in self.eos_ids + self.diagnostic_ids):
            raise ValueError("invalid special token ID")


@dataclass
class Rollout:
    question_id: str
    seed: int
    arm: str
    generated_ids: list[int]
    stop_reason: str
    diagnostics_mean: dict[str, float]

    @property
    def consumed_tokens(self):
        return len(self.generated_ids)


def generate_arm(f: Backend, q: Backend, prompt, question_id, seed, arm, settings):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    prefix = list(prompt)
    if not prefix or any(not isinstance(t, int) or t < 0 for t in prefix):
        raise ValueError("nonempty tokenized prompt required")
    fs, qs = f.new_session(), q.new_session()
    if fs is qs:
        raise ValueError("F and Q sessions must be independent")
    generated, totals = [], {}
    reason = "cap"
    special = tuple(sorted(set(settings.eos_ids + settings.diagnostic_ids)))
    for step in range(settings.max_new_tokens):
        # Immutable snapshot prevents a backend from modifying the other's input.
        current = tuple(prefix)
        pair = PolicyPair.from_logits(fs.next_logits(current), qs.next_logits(current),
                                      settings.temperature, settings.top_p)
        token = inverse_cdf(pair.distribution(arm), keyed_uniform(question_id, seed, step))
        for key, value in pair.diagnostics(special).items():
            totals[key] = totals.get(key, 0.0) + value
        prefix.append(token)
        generated.append(token)
        if token in settings.eos_ids:
            reason = "eos"
            break
    return Rollout(question_id, seed, arm, generated, reason,
                   {key: value / len(generated) for key, value in totals.items()})


def generate_four(f, q, prompt, question_id, seed, settings, arm_order=ARMS):
    if sorted(arm_order) != sorted(ARMS):
        raise ValueError("each primary arm must occur exactly once")
    return {arm: generate_arm(f, q, prompt, question_id, seed, arm, settings)
            for arm in arm_order}
