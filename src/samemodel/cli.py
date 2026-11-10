"""Command line interface."""
from __future__ import annotations

import argparse
import json
import secrets
import sys
from typing import Dict, List, Optional, Sequence

from . import __version__, checks, report
from .client import Client, EndpointError
from .compare import Comparison, compare
from .prompts import PROMPTS

EXIT_THRESHOLDS = {"never": 99, "noise": 1, "drift": 2, "different": 3}


def _endpoint_args(p: argparse.ArgumentParser, prefix: str = "") -> None:
    flag = f"--{prefix}" if prefix else "--"
    p.add_argument(f"{flag}base-url", required=True, metavar="URL",
                   help="OpenAI-compatible endpoint, e.g. http://localhost:8080")
    p.add_argument(f"{flag}model", metavar="NAME",
                   help="model name to request (default: the first one the endpoint lists)")
    p.add_argument(f"{flag}api-key", metavar="KEY", help="bearer token, if the endpoint needs one")


def _run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-n", "--prompts", type=int, default=24, metavar="N",
                   help="how many built-in prompts to use (default: 24, max: %d)" % len(PROMPTS))
    p.add_argument("--prompt-file", metavar="FILE",
                   help="use your own prompts instead: a JSON list of strings, or one prompt per line")
    p.add_argument("--max-tokens", type=int, default=128, help="tokens to generate per prompt (default: 128)")
    p.add_argument("--top-logprobs", type=int, default=10,
                   help="alternatives to request per token (default: 10; 0 compares text only)")
    p.add_argument("--seed", type=int, metavar="N",
                   help="send this seed with every request (default: none; greedy decoding needs "
                        "none, and some servers stop batching requests that carry one)")
    p.add_argument("--extra-body", metavar="JSON",
                   help='extra fields merged into every request, e.g. \'{"cache_prompt": false}\'')
    p.add_argument("--timeout", type=float, default=600, help="seconds per request (default: 600)")


def _report_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", metavar="FILE", help="also write the results as JSON")
    p.add_argument("--forks", type=int, default=0, metavar="N",
                   help="list the N largest forks of each check")
    p.add_argument("--fail-on", choices=sorted(EXIT_THRESHOLDS), default="never",
                   help="exit with status 1 when any check is at least this bad (default: never)")


def _load_prompts(args) -> List[str]:
    if args.prompt_file:
        text = open(args.prompt_file, encoding="utf-8").read()
        try:
            prompts = [str(x) for x in json.loads(text)]
        except ValueError:
            prompts = [line.strip() for line in text.splitlines() if line.strip()]
        if not prompts:
            raise SystemExit(f"no prompts found in {args.prompt_file}")
        return prompts
    return PROMPTS[: max(1, min(args.prompts, len(PROMPTS)))]


def _client(args, prefix: str = "") -> Client:
    get = lambda name: getattr(args, (prefix + name).replace("-", "_"))  # noqa: E731
    extra = json.loads(args.extra_body) if args.extra_body else None
    return Client(get("base_url"), model=get("model"), api_key=get("api_key"),
                  max_tokens=args.max_tokens, top_logprobs=args.top_logprobs, seed=args.seed,
                  extra_body=extra, timeout=args.timeout)


def _progress(label: str):
    def show(done: int, total: int) -> None:
        if sys.stderr.isatty():
            sys.stderr.write(f"\r{label}: {done}/{total}")
            if done == total:
                sys.stderr.write("\n")
            sys.stderr.flush()
    return show


def _finish(rows: Dict[str, Comparison], args, header: str, meta: dict) -> int:
    print(header)
    print()
    print(report.render(rows, forks=args.forks))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"tool": "samemodel", "version": __version__, **meta,
                       "checks": report.to_json(rows)}, f, indent=1)
    return 1 if report.worst(rows) >= EXIT_THRESHOLDS[args.fail_on] else 0


def cmd_selfcheck(args) -> int:
    client = _client(args)
    prompts = _load_prompts(args)
    rows = checks.selfcheck(client, prompts, companions=args.companions,
                            nonce=None if args.no_nonce else "auto",
                            shared_system=args.shared_system, progress=_progress("selfcheck"))
    kind = "shared system prompt" if args.shared_system else "unique conversations"
    header = (f"samemodel selfcheck  {args.base_url}  model={client.model}  "
              f"{len(prompts)} prompts x {args.max_tokens} tokens  ({kind})")
    code = _finish(rows, args, header, {"endpoint": args.base_url, "model": client.model,
                                        "prompts": len(prompts), "max_tokens": args.max_tokens,
                                        "shared_system": args.shared_system, "seed": args.seed})
    reuse = rows[checks.REUSE]
    if reuse.cache_hit_ratio is not None:
        first = reuse.scratch_hit_ratio or 0.0
        print(f"\nPrompt tokens served from cache, as the endpoint reports it: "
              f"{100 * first:.0f}% first request, {100 * reuse.cache_hit_ratio:.0f}% repeat.")
        if reuse.cache_hit_ratio < 0.5:
            print("The repeat was mostly recomputed, so 'cache reuse' says little about caching here.")
    if not args.shared_system:
        print("Every conversation here starts differently. Run again with --shared-system to test "
              "requests\nthat share a system prompt, as most deployments do.")
    return code


def cmd_record(args) -> int:
    client = _client(args)
    prompts = _load_prompts(args)
    records = checks.record_run(client, prompts, nonce=args.tag, companions=args.companions,
                                shared_system=args.shared_system, progress=_progress("record"))
    checks.save(records, args.out)
    print(f"recorded {len(records)} prompts from {args.base_url} (model={client.model}) to {args.out}")
    return 0


def cmd_diff(args) -> int:
    rows = {"b vs a": compare(checks.load(args.a), checks.load(args.b))}
    return _finish(rows, args, f"samemodel diff  a={args.a}  b={args.b}", {"a": args.a, "b": args.b})


def cmd_compare(args) -> int:
    prompts = _load_prompts(args)
    nonce = secrets.token_hex(4)  # the same marker on both sides keeps the prompts identical
    a = checks.record_run(_client(args, "a-"), prompts, nonce, progress=_progress("endpoint a"))
    b = checks.record_run(_client(args, "b-"), prompts, nonce, progress=_progress("endpoint b"))
    rows = {"b vs a": compare(a, b)}
    header = (f"samemodel compare  a={args.a_base_url}  b={args.b_base_url}  "
              f"{len(prompts)} prompts x {args.max_tokens} tokens")
    return _finish(rows, args, header, {"a": args.a_base_url, "b": args.b_base_url})


def cmd_mock(args) -> int:
    from .mock import MockServer
    server = MockServer(host=args.host, port=args.port, cache_drift=args.cache_drift,
                        batch_drift=args.batch_drift, resolution=args.resolution,
                        logprobs=not args.no_logprobs)
    print(f"mock endpoint listening on {server.url}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="samemodel",
        description="Measure whether an inference endpoint still serves the same model.")
    ap.add_argument("--version", action="version", version=f"samemodel {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("selfcheck", help="one endpoint: do prompt-cache reuse and concurrent load "
                                         "change its answers?")
    _endpoint_args(p)
    _run_args(p)
    p.add_argument("--companions", type=int, default=3,
                   help="unrelated requests sent alongside each measured one (default: 3; 0 skips the check)")
    p.add_argument("--shared-system", action="store_true",
                   help="give every conversation the same system message, as most deployments do; "
                        "the first request for a prompt then already reuses the cached system prefix")
    p.add_argument("--no-nonce", action="store_true",
                   help="do not add a per-run system message "
                        "(the first request may then reuse a cached prefix)")
    _report_args(p)
    p.set_defaults(func=cmd_selfcheck)

    p = sub.add_parser("record", help="record one run to a file, to compare later")
    _endpoint_args(p)
    _run_args(p)
    p.add_argument("-o", "--out", required=True, metavar="FILE")
    p.add_argument("--tag", metavar="TEXT",
                   help="start every conversation with a system message containing TEXT, so that "
                        "nothing is reused from the server's prompt cache; use the same tag for "
                        "the runs you want to compare, and a new one per server lifetime")
    p.add_argument("--companions", type=int, default=0,
                   help="send this many unrelated requests alongside each recorded one (default: 0)")
    p.add_argument("--shared-system", action="store_true",
                   help="with --tag: give every conversation the same system message and put the "
                        "marker at the start of the user message instead")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("diff", help="compare two recorded runs")
    p.add_argument("a")
    p.add_argument("b")
    _report_args(p)
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("compare", help="two endpoints: record both and compare")
    _endpoint_args(p, "a-")
    _endpoint_args(p, "b-")
    _run_args(p)
    _report_args(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("mock", help="run a toy endpoint for trying the tool without a model")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8642)
    p.add_argument("--cache-drift", type=float, default=0.0, metavar="NATS")
    p.add_argument("--batch-drift", type=float, default=0.0, metavar="NATS")
    p.add_argument("--resolution", type=float, metavar="NATS",
                   help="round reported log-probabilities to this grid")
    p.add_argument("--no-logprobs", action="store_true")
    p.set_defaults(func=cmd_mock)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except EndpointError as e:
        print(f"samemodel: {e}", file=sys.stderr)
        return 2
