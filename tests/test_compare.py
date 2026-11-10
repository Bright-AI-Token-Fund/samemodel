import math

from samemodel.compare import compare, logprob_resolution, percentile
from samemodel.report import scale, verdict


def tok(token_id, alts, with_ids=True):
    """alts: list of (id, logprob), best first; the first one is the emitted token."""
    top = [[i if with_ids else None, f"t{i}", lp] for i, lp in alts]
    return {"id": token_id if with_ids else None, "token": f"t{token_id}", "logprob": alts[0][1], "top": top}


def rec(pid, tokens):
    return {"id": pid, "text": "".join(t["token"] for t in tokens), "tokens": tokens}


def test_identical_runs():
    a = [rec("p0", [tok(1, [(1, -0.1), (2, -2.5)]), tok(3, [(3, -0.2), (4, -1.8)])])]
    c = compare(a, a)
    assert c.prompts == 1 and c.identical_outputs == 1
    assert c.positions == 2 and c.agreement == 1.0
    assert c.bitwise_like
    assert verdict(c) == "identical"


def test_small_logprob_difference_without_fork_is_noise():
    a = [rec("p0", [tok(1, [(1, -0.1), (2, -2.5)]) for _ in range(30)])]
    b = [rec("p0", [tok(1, [(1, -0.1005), (2, -2.4990)]) for _ in range(30)])]
    c = compare(a, b)
    assert c.identical_outputs == 1 and not c.bitwise_like
    assert math.isclose(percentile(c.drift, 0.5), 0.0015, abs_tol=1e-9)
    assert verdict(c) == "noise"


def test_fork_position_margin_and_shift():
    shared = tok(1, [(1, -0.1), (2, -2.5)])
    # at the fork A prefers token 5 over 6 by 0.10 nat, B prefers 6 over 5 by 0.30 nat
    a = [rec("p0", [shared, tok(5, [(5, -0.60), (6, -0.70)]), tok(9, [(9, -0.1), (8, -3.0)])])]
    b = [rec("p0", [shared, tok(6, [(6, -0.50), (5, -0.80)]), tok(7, [(7, -0.1), (8, -3.0)])])]
    c = compare(a, b)
    assert c.identical_outputs == 0
    assert c.positions == 2          # the shared token and the fork position; nothing after it
    assert c.agreeing == 1
    assert len(c.forks) == 1
    fork = c.forks[0]
    assert fork.position == 1
    assert math.isclose(fork.margin_a, 0.10, abs_tol=1e-9)
    assert math.isclose(fork.shift, 0.40, abs_tol=1e-9)


def test_length_difference_is_not_identical_but_has_no_fork():
    t = tok(1, [(1, -0.1), (2, -2.5)])
    c = compare([rec("p0", [t, t, t])], [rec("p0", [t, t])])
    assert c.identical_outputs == 0 and not c.forks and c.positions == 2


def test_token_strings_are_used_when_ids_are_missing():
    a = [rec("p0", [tok(1, [(1, -0.1), (2, -2.5)], with_ids=False)])]
    b = [rec("p0", [tok(2, [(2, -0.4), (1, -1.1)], with_ids=False)])]
    c = compare(a, b)
    assert len(c.forks) == 1 and c.forks[0].token_a == "t1" and c.forks[0].token_b == "t2"
    assert math.isclose(c.forks[0].shift, 2.4 + 0.7, abs_tol=1e-9)


def test_text_fallback_without_logprobs():
    a = [{"id": "p0", "text": "hello", "tokens": []}, {"id": "p1", "text": "x", "tokens": []}]
    b = [{"id": "p0", "text": "hello", "tokens": []}, {"id": "p1", "text": "y", "tokens": []}]
    c = compare(a, b)
    assert not c.with_logprobs and c.identical_outputs == 1
    assert verdict(c) == "differs-unsized"
    assert verdict(compare(a, a)) == "identical"


def test_records_are_paired_by_id():
    t = tok(1, [(1, -0.1), (2, -2.5)])
    c = compare([rec("p0", [t]), rec("p1", [t])], [rec("p1", [t]), rec("zz", [t])])
    assert c.prompts == 1


def test_coarse_logprob_grid_is_detected():
    coarse = [rec("p0", [tok(1, [(1, -k / 16), (2, -3.0)]) for k in range(1, 40)])]
    fine = [rec("p0", [tok(1, [(1, -0.1 - k / 97), (2, -3.0)]) for k in range(1, 40)])]
    assert logprob_resolution(coarse) == 1 / 16
    assert logprob_resolution(fine) is None


def test_scale_names_the_nearest_reference():
    assert "fp16 to Q8_0" in scale(0.07)
    assert "Q8_0 to Q4_K_M" in scale(0.4)
    assert scale(0.0) is None and scale(None) is None


def test_heavy_tail_is_not_reported_as_noise():
    # 97 positions agree to the last digit, 3 are far off: the median drift is 0
    good = tok(1, [(1, -0.1), (2, -2.5)])
    bad = tok(1, [(1, -1.2), (2, -0.9)])
    a = [rec("p0", [good] * 100)]
    b = [rec("p0", [good] * 97 + [bad] * 3)]
    c = compare(a, b)
    assert percentile(c.drift, 0.5) == 0.0
    assert verdict(c) == "different"
