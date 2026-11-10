"""Turn a Comparison into a verdict and a readable report."""
from __future__ import annotations

import math
import textwrap
from typing import Any, Dict, List, Optional

from .compare import Comparison, percentile

# Verdict thresholds, in nats of top-2 log-odds drift on identical-context positions.
# Drift measures the perturbation itself; how often it flips a token also depends on
# how peaked the model's distributions are, so agreement is only used as a fallback.
# The numbers come from one set of measurements on a small model (see benchmarks/):
# treat the verdict as a reading aid, not a pass/fail standard. The raw numbers are
# always printed next to it.
# rounding-level: fp16 under another batch shape measures ~0.0015 (median) / ~0.01 (p99)
NOISE_MAX_P50, NOISE_MAX_P99 = 0.01, 0.1
# a small quantisation step, fp16 -> Q8_0, measures ~0.065 / ~0.4
DRIFT_MAX_P50, DRIFT_MAX_P99 = 0.15, 1.0
# Fallback when log-probabilities sit on a coarse grid and small drifts read as zero.
COARSE_GRID = 1.0 / 256
NOISE_MIN_AGREEMENT, DRIFT_MIN_AGREEMENT = 0.999, 0.97

WIDTH = 92  # the report wraps here so that it reads well in a terminal and in a code block

SEVERITY = {"identical": 0, "noise": 1, "drift": 2, "different": 3, "differs-unsized": 3, "empty": 0}

# Reference points for scale: median top-2 log-odds drift, in nats, measured with
# llama.cpp b12244 on Qwen2.5-0.5B-Instruct. See benchmarks/README.md.
REFERENCES = [
    (0.0015, "the rounding noise of fp16 weights under a different batch shape"),
    (0.065, "switching the weights from fp16 to Q8_0"),
    (0.33, "switching the weights from Q8_0 to Q4_K_M"),
]


def verdict(c: Comparison) -> str:
    if c.prompts == 0:
        return "empty"
    if not c.with_logprobs:
        return "identical" if c.identical_outputs == c.prompts else "differs-unsized"
    if c.bitwise_like:
        return "identical"
    d50, d99 = percentile(c.drift, 0.5), percentile(c.drift, 0.99)
    coarse = c.resolution is not None and c.resolution >= COARSE_GRID
    if d50 is None or coarse:
        agreement = c.agreement if c.agreement is not None else 0.0
        if agreement >= NOISE_MIN_AGREEMENT:
            return "noise"
        return "drift" if agreement >= DRIFT_MIN_AGREEMENT else "different"
    # a low median with a heavy tail means "mostly fine, sometimes far off": the p99
    # bound keeps that from being reported as noise
    if d50 < NOISE_MAX_P50 and d99 < NOISE_MAX_P99:
        return "noise"
    if d50 < DRIFT_MAX_P50 and d99 < DRIFT_MAX_P99:
        return "drift"
    return "different"


def scale(drift_p50: Optional[float]) -> Optional[str]:
    """Express a drift as a multiple of the nearest reference point."""
    if not drift_p50:
        return None
    ref, label = min(REFERENCES, key=lambda r: abs(_log_ratio(drift_p50, r[0])))
    ratio = drift_p50 / ref
    if 0.8 <= ratio <= 1.25:
        return f"about the size of {label}"
    return f"about {ratio:.1f}x {label}"


def _log_ratio(a: float, b: float) -> float:
    return math.log(a / b)


def _pct(x: Optional[float]) -> str:
    return "   n/a" if x is None else f"{100 * x:6.2f}%"


def _num(x: Optional[float]) -> str:
    return "    n/a" if x is None else f"{x:7.4f}"


def explain(name: str, c: Comparison) -> List[str]:
    v = verdict(c)
    s = c.summary()
    changed = c.prompts - c.identical_outputs
    lines: List[str] = []
    if v == "empty":
        return [f"{name}: nothing to compare."]
    if v == "identical" and c.with_logprobs:
        lines.append(f"{name}: IDENTICAL. Same tokens and same log-probabilities on all "
                     f"{c.positions} compared positions.")
    elif v == "identical":
        lines.append(f"{name}: IDENTICAL TEXT for all {c.prompts} prompts. The endpoint returned "
                     "no log-probabilities, so smaller differences cannot be seen.")
    elif v == "differs-unsized":
        lines.append(f"{name}: OUTPUTS DIFFER for {changed} of {c.prompts} prompts. The endpoint "
                     "returned no log-probabilities, so the size of the difference is unknown.")
    else:
        label = {"noise": "ROUNDING NOISE", "drift": "DRIFT", "different": "DIFFERENT"}[v]
        lines.append(f"{name}: {label}. {changed} of {c.prompts} outputs changed; on "
                     f"{c.positions} positions with an identical context the argmax agrees "
                     f"{_pct(s['argmax_agreement']).strip()} of the time; the top-2 log-odds "
                     f"move by {_num(s['drift_p50']).strip()} nat at the median and "
                     f"{_num(s['drift_p99']).strip()} nat at the 99th percentile.")
        if not s["drift_p50"] and s["drift_p99"]:
            lines.append("  Most positions match exactly; the difference is concentrated in a few.")
        size = scale(s["drift_p50"])
        if size:
            lines.append(f"  For scale: {size}.")
        if v == "noise":
            lines.append("  Outputs change only where two tokens were nearly tied.")
        if v == "different":
            lines.append("  If these were meant to be the same model and configuration, look closer.")
    if c.resolution and v != "identical":
        lines.append(f"  Note: this endpoint reports log-probabilities on a {c.resolution:g}-nat "
                     "grid, so drift below that reads as 0. Rely on the argmax agreement.")
    wrapped: List[str] = []
    for line in lines:
        indent = "  " if line.startswith("  ") else ""
        wrapped += textwrap.wrap(line.strip(), width=WIDTH, initial_indent=indent,
                                 subsequent_indent="  ")
    return wrapped


def table(rows: Dict[str, Comparison]) -> List[str]:
    width = max([len(n) for n in rows] + [5])
    out = [f"{'check':{width}}  {'same':>7}  {'argmax':>7}  {'drift50':>7}  {'drift99':>7}  "
           f"{'KL99':>7}  {'forks':>5}  verdict"]
    for name, c in rows.items():
        s = c.summary()
        out.append(f"{name:{width}}  {c.identical_outputs:>3}/{c.prompts:<3}  "
                   f"{_pct(s['argmax_agreement'])}  {_num(s['drift_p50'])}  {_num(s['drift_p99'])}  "
                   f"{_num(s['kl_p99'])}  {len(c.forks):>5}  {verdict(c)}")
    return out


def render(rows: Dict[str, Comparison], forks: int = 0) -> str:
    lines = table(rows) + [""]
    for name, c in rows.items():
        lines += explain(name, c)
        shown = sorted(c.forks, key=lambda f: -(f.shift if f.shift is not None else 1e9))[:forks]
        for f in shown:
            shift = "n/a" if f.shift is None else f"{f.shift:.4f}"
            lines.append(f"    fork in {f.prompt_id} at token {f.position}: {f.token_a!r} -> "
                         f"{f.token_b!r}, log-odds moved {shift} nat")
    return "\n".join(lines)


def to_json(rows: Dict[str, Comparison]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, c in rows.items():
        s = c.summary()
        s["verdict"] = verdict(c)
        s["forks_detail"] = [f.__dict__ for f in c.forks]
        out[name] = s
    return out


def worst(rows: Dict[str, Comparison]) -> int:
    return max([SEVERITY[verdict(c)] for c in rows.values()] + [0])
