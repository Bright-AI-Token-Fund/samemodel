"""End-to-end tests against the built-in mock endpoint. No model, no GPU."""
import json

import pytest

from samemodel import checks
from samemodel.cli import main
from samemodel.client import Client, EndpointError
from samemodel.compare import percentile
from samemodel.mock import MockServer
from samemodel.prompts import PROMPTS
from samemodel.report import verdict

FAST = {"window_ms": 40}


def run_selfcheck(n=12, max_tokens=48, **server):
    with MockServer(**FAST, **server) as url:
        client = Client(url, max_tokens=max_tokens)
        return checks.selfcheck(client, PROMPTS[:n])


def test_stable_endpoint_is_identical_everywhere():
    rows = run_selfcheck()
    assert list(rows) == [checks.REUSE, checks.FIRST_LOADED, checks.REPEAT_LOADED]
    assert [verdict(c) for c in rows.values()] == ["identical"] * 3
    assert rows[checks.REUSE].prompts == 6 and rows[checks.FIRST_LOADED].prompts == 6
    assert rows[checks.REPEAT_LOADED].prompts == 12
    assert rows[checks.REUSE].cache_hit_ratio == 1.0
    assert rows[checks.REUSE].scratch_hit_ratio == 0.0


def test_cache_dependent_endpoint_is_caught_by_cache_reuse_only():
    rows = run_selfcheck(n=20, max_tokens=64, cache_drift=0.08)
    repeat, busy = rows[checks.REUSE], rows[checks.REPEAT_LOADED]
    assert verdict(repeat) == "drift"
    assert verdict(busy) == "identical"
    # each logit gets N(0, 0.08) noise, so the top-2 log-odds move by |N(0, 0.08*sqrt(2))|,
    # whose median is 0.954 * 0.08 = 0.076
    assert 0.055 < percentile(repeat.drift, 0.5) < 0.10


def test_batch_dependent_endpoint_is_caught_by_cobatching_only():
    rows = run_selfcheck(n=20, max_tokens=64, batch_drift=0.08)
    assert verdict(rows[checks.REUSE]) == "identical"
    assert verdict(rows[checks.FIRST_LOADED]) == "drift"      # the first request was the loaded one
    assert verdict(rows[checks.REPEAT_LOADED]) == "drift"


def test_tiny_drift_is_noise_and_large_drift_is_different():
    assert verdict(run_selfcheck(cache_drift=0.002)["cache reuse"]) == "noise"
    assert verdict(run_selfcheck(cache_drift=1.5)["cache reuse"]) == "different"


def test_coarse_grid_is_reported():
    rows = run_selfcheck(cache_drift=0.08, resolution=1 / 16)
    assert rows[checks.REUSE].resolution == 1 / 16


def test_endpoint_without_logprobs_falls_back_to_text():
    assert verdict(run_selfcheck(logprobs=False)["cache reuse"]) == "identical"
    assert verdict(run_selfcheck(logprobs=False, cache_drift=1.5)["cache reuse"]) == "differs-unsized"


def test_endpoint_without_token_ids_still_works():
    rows = run_selfcheck(n=20, max_tokens=64, cache_drift=0.08, token_ids=False)
    assert verdict(rows[checks.REUSE]) == "drift"


def test_record_diff_and_exit_codes(tmp_path, capsys):
    a, b, out = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl", tmp_path / "r.json"
    with MockServer(**FAST, cache_drift=0.08) as url:
        common = ["--base-url", url, "-n", "10", "--max-tokens", "48"]
        assert main(["record", *common, "-o", str(a)]) == 0     # cold
        assert main(["record", *common, "-o", str(b)]) == 0     # same prompts again: cache path
    assert main(["diff", str(a), str(a)]) == 0
    assert "IDENTICAL" in capsys.readouterr().out
    assert main(["diff", str(a), str(b), "--fail-on", "different"]) == 0
    assert main(["diff", str(a), str(b), "--fail-on", "drift", "--json", str(out), "--forks", "2"]) == 1
    text = capsys.readouterr().out
    assert "DRIFT" in text and "fork in p" in text
    result = json.loads(out.read_text())
    assert result["checks"]["b vs a"]["verdict"] == "drift"
    assert result["checks"]["b vs a"]["prompts"] == 10


def test_compare_two_endpoints(capsys):
    with MockServer(**FAST) as url_a, MockServer(**FAST) as url_b:
        code = main(["compare", "--a-base-url", url_a, "--b-base-url", url_b, "-n", "6",
                     "--max-tokens", "32", "--fail-on", "noise"])
    assert code == 0 and "IDENTICAL" in capsys.readouterr().out


def test_selfcheck_cli_prints_cache_ratio(capsys):
    with MockServer(**FAST, batch_drift=0.08) as url:
        assert main(["selfcheck", "--base-url", url + "/v1", "-n", "8", "--max-tokens", "32"]) == 0
    out = capsys.readouterr().out
    assert "repeat under load" in out and "0% first request, 100% repeat" in out
    assert "unique conversations" in out and "--shared-system" in out


def test_unreachable_endpoint_is_a_clean_error(capsys):
    assert main(["record", "--base-url", "http://127.0.0.1:9", "-n", "1", "-o", "/dev/null",
                 "--timeout", "2"]) == 2
    assert "cannot reach" in capsys.readouterr().err


def test_bad_response_shape_raises():
    from samemodel.client import parse_response
    with pytest.raises(EndpointError):
        parse_response("p0", "x", {"error": "nope"})


def test_markers_differ_from_the_first_characters():
    marks = [checks.marker("run1", str(i)) for i in range(50)]
    assert len({m[:3] for m in marks}) > 40          # almost no shared prefix between prompts
    assert checks.marker("run1", "7") == checks.marker("run1", "7")
    assert checks.marker(None, "7") is None


def test_shared_system_prompt_mode(capsys):
    with MockServer(**FAST, cache_drift=0.08) as url:
        assert main(["selfcheck", "--base-url", url, "-n", "8", "--max-tokens", "32",
                     "--shared-system"]) == 0
    out = capsys.readouterr().out
    assert "shared system prompt" in out and "--shared-system to test" not in out


def test_conversation_shapes():
    system, user = checks.conversation("Hi", "run1", "3")
    assert user == "Hi" and system.endswith(checks.SHARED_SYSTEM) and system != checks.SHARED_SYSTEM
    system, user = checks.conversation("Hi", "run1", "3", shared_system=True)
    assert system == checks.SHARED_SYSTEM and user.endswith("] Hi") and user.startswith("[")
    assert checks.conversation("Hi", None, "3") == (None, "Hi")
