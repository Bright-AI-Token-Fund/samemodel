"""Token-level comparison of two recorded runs.

Two greedy runs of the same prompt share a prefix of identical generated tokens up
to the first position where they emit different tokens (the *fork*). On every
position of that prefix, and on the fork position itself, both runs scored exactly
the same context, so their top-k log-probabilities can be compared directly.
Positions after the fork are ignored: the contexts differ there, so differences no
longer say anything about the two endpoints.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

Record = Dict[str, Any]


def _use_ids(records: Sequence[Record]) -> bool:
    """Token ids are only trusted when every token of every record carries one."""
    seen = False
    for r in records:
        for t in r.get("tokens", []):
            if t.get("id") is None:
                return False
            seen = True
            for alt in t.get("top", []):
                if alt[0] is None:
                    return False
    return seen


def _top(tok: Dict[str, Any], ids: bool) -> Dict[Hashable, float]:
    idx = 0 if ids else 1
    return {alt[idx]: alt[2] for alt in tok.get("top", []) if alt[2] is not None}


def _key(tok: Dict[str, Any], ids: bool) -> Hashable:
    return tok.get("id") if ids else tok.get("token")


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def logprob_resolution(records: Sequence[Record]) -> Optional[float]:
    """Return the grid step if every reported logprob sits on a coarse binary grid.

    Some engines compute logits in half precision and report log-probabilities that
    are multiples of 1/16 ... 1/1024. On such a grid small drifts round to zero, so
    the report has to say so. Returns None when the values look full precision.
    """
    values = [t["logprob"] for r in records for t in r.get("tokens", [])
              if t.get("logprob") is not None]
    if len(values) < 20:
        return None
    for k in range(0, 11):
        scale = float(2 ** k)
        if all(abs(v * scale - round(v * scale)) < 1e-6 for v in values):
            return 1.0 / scale
    return None


@dataclass
class Fork:
    prompt_id: str
    position: int
    token_a: Optional[str]
    token_b: Optional[str]
    # how much A preferred its own token over B's choice, in nats (None if unknown)
    margin_a: Optional[float]
    # how far the log-odds of the two emitted tokens moved between A and B
    shift: Optional[float]


@dataclass
class Comparison:
    prompts: int = 0
    identical_outputs: int = 0
    positions: int = 0          # identical-context positions compared
    agreeing: int = 0           # of those, positions with the same argmax
    max_logprob_diff: float = 0.0
    with_logprobs: bool = True
    kl: List[float] = field(default_factory=list)
    drift: List[float] = field(default_factory=list)
    forks: List[Fork] = field(default_factory=list)
    resolution: Optional[float] = None
    # share of prompt tokens the endpoint reported as served from its cache (selfcheck only)
    cache_hit_ratio: Optional[float] = None
    # the same for the first request of each prompt, which should be close to zero
    scratch_hit_ratio: Optional[float] = None

    @property
    def agreement(self) -> Optional[float]:
        return self.agreeing / self.positions if self.positions else None

    @property
    def bitwise_like(self) -> bool:
        """Same tokens everywhere and every compared logprob equal."""
        return (self.prompts > 0 and self.identical_outputs == self.prompts
                and self.max_logprob_diff <= 1e-9)

    def summary(self) -> Dict[str, Any]:
        shifts = [f.shift for f in self.forks if f.shift is not None]
        return {
            "prompts": self.prompts,
            "identical_outputs": self.identical_outputs,
            "positions": self.positions,
            "argmax_agreement": self.agreement,
            "max_logprob_diff": self.max_logprob_diff,
            "kl_p50": percentile(self.kl, 0.5),
            "kl_p99": percentile(self.kl, 0.99),
            "drift_p50": percentile(self.drift, 0.5),
            "drift_p99": percentile(self.drift, 0.99),
            "forks": len(self.forks),
            "fork_shift_p50": percentile(shifts, 0.5),
            "fork_position_p50": percentile([f.position for f in self.forks], 0.5),
            "logprob_resolution": self.resolution,
            "with_logprobs": self.with_logprobs,
            "cache_hit_ratio": self.cache_hit_ratio,
            "scratch_hit_ratio": self.scratch_hit_ratio,
        }


def _kl_shared(la: Dict[Hashable, float], lb: Dict[Hashable, float]) -> Optional[float]:
    """KL(A || B) over the tokens present in both top-k lists, renormalised."""
    shared = [k for k in la if k in lb]
    if len(shared) < 2:
        return None
    za = math.log(sum(math.exp(la[k]) for k in shared))
    zb = math.log(sum(math.exp(lb[k]) for k in shared))
    return max(0.0, sum(math.exp(la[k] - za) * ((la[k] - za) - (lb[k] - zb)) for k in shared))


def compare(run_a: Sequence[Record], run_b: Sequence[Record]) -> Comparison:
    """Compare two runs; records are paired by their ``id`` field."""
    by_id_b = {r["id"]: r for r in run_b}
    pairs: List[Tuple[Record, Record]] = [(r, by_id_b[r["id"]]) for r in run_a if r["id"] in by_id_b]
    ids = _use_ids([r for pair in pairs for r in pair])
    out = Comparison()
    out.resolution = logprob_resolution([r for pair in pairs for r in pair])
    any_logprobs = False

    for ra, rb in pairs:
        out.prompts += 1
        ta, tb = ra.get("tokens", []), rb.get("tokens", [])
        if not ta and not tb:
            # the endpoint returned no per-token data: fall back to comparing text
            if ra.get("text") == rb.get("text"):
                out.identical_outputs += 1
            continue
        n = min(len(ta), len(tb))
        t = 0
        while t < n and _key(ta[t], ids) == _key(tb[t], ids):
            t += 1
        forked = t < n
        if not forked and len(ta) == len(tb):
            out.identical_outputs += 1

        for i in range(0, t + 1 if forked else t):
            la, lb = _top(ta[i], ids), _top(tb[i], ids)
            out.positions += 1
            out.agreeing += _key(ta[i], ids) == _key(tb[i], ids)
            if not la or not lb:
                continue
            any_logprobs = True
            for k, v in la.items():
                if k in lb:
                    out.max_logprob_diff = max(out.max_logprob_diff, abs(v - lb[k]))
            kl = _kl_shared(la, lb)
            if kl is not None:
                out.kl.append(kl)
            ranked = sorted(la.items(), key=lambda kv: -kv[1])
            if len(ranked) >= 2:
                (a1, v1), (a2, v2) = ranked[0], ranked[1]
                if a1 in lb and a2 in lb:
                    out.drift.append(abs((v1 - v2) - (lb[a1] - lb[a2])))

        if forked:
            la, lb = _top(ta[t], ids), _top(tb[t], ids)
            ka, kb = _key(ta[t], ids), _key(tb[t], ids)
            margin = shift = None
            if ka in la and kb in la:
                margin = la[ka] - la[kb]
                if ka in lb and kb in lb:
                    shift = margin + (lb[kb] - lb[ka])
            out.forks.append(Fork(str(ra["id"]), t, ta[t].get("token"), tb[t].get("token"),
                                  margin, shift))
    out.with_logprobs = any_logprobs
    return out
