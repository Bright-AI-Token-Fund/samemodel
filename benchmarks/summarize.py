#!/usr/bin/env python3
"""Turn the output directories of llamacpp.sh / mlx.sh into the Markdown tables of
benchmarks/README.md, so that no number in that file is typed by hand.

Usage:
  summarize.py selfcheck LABEL=DIR [LABEL=DIR ...] [--weights fp16 q8_0 q4_k_m]
  summarize.py overview  LABEL=DIR [LABEL=DIR ...] --weights fp16 q8_0 q4_k_m
  summarize.py weights   LABEL=DIR [LABEL=DIR ...] --steps fp16:q8_0 q8_0:q4_k_m
  summarize.py across    LABEL_A=DIR_A LABEL_B=DIR_B --weights q4_k_m q8_0 fp16
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from samemodel import checks
from samemodel.compare import compare
from samemodel.report import verdict, verdict_of


def pct(x):
    return "n/a" if x is None else f"{100 * x:.2f}%"


def num(x):
    return "n/a" if x is None else f"{x:.4f}"


def cell(s):
    """verdict / identical outputs and argmax agreement / median and 99th percentile drift"""
    v = verdict_of(s)  # recomputed, so that a table never mixes threshold versions
    lines = [f"**{v}**", f"{s['identical_outputs']}/{s['prompts']} same"]
    if v != "identical":
        lines[1] += f", argmax {pct(s['argmax_agreement'])}"
        lines.append(f"drift {num(s['drift_p50'])} / {num(s['drift_p99'])}")
    return "<br>".join(lines)


def split(pairs):
    return [p.split("=", 1) for p in pairs]


def cmd_selfcheck(args):
    dirs = split(args.dirs)
    lead = ["setup"] if len(dirs) > 1 else []
    head = lead + ["weights", "conversations", "cache reuse", "first request under load",
                   "repeat under load", "cached on first / repeat request"]
    print("| " + " | ".join(head) + " |")
    print("| " + " | ".join("---" for _ in head) + " |")
    for label, d in dirs:
        found = {}
        for path in glob.glob(os.path.join(d, "*.selfcheck.*.json")):
            w, _, kind, _ = os.path.basename(path).split(".")
            found[w, kind] = path
        order = args.weights or sorted({w for w, _ in found})
        for w, kind in [(w, k) for w in order for k in ("unique", "shared") if (w, k) in found]:
            c = json.load(open(found[w, kind]))["checks"]
            reuse = c["cache reuse"]
            first, rep = reuse.get("scratch_hit_ratio"), reuse.get("cache_hit_ratio")
            cached = "n/a" if rep is None else f"{100 * (first or 0):.0f}% / {100 * rep:.0f}%"
            cells = [cell(c[k]) if k in c else "n/a"
                     for k in ("cache reuse", "first request under load", "repeat under load")]
            row = ([label] if lead else []) + [w, kind, *cells, cached]
            print("| " + " | ".join(row) + " |")


def brief(s):
    v = verdict_of(s)
    if v == "identical":
        return "identical"
    changed = s["prompts"] - s["identical_outputs"]
    if s.get("logprob_resolution"):  # drift is quantised by the grid: show the agreement instead
        size = f"argmax {pct(s['argmax_agreement'])}"
    else:
        size = f"drift {num(s['drift_p50'])}"
    return f"{v}: {changed} of {s['prompts']} answers changed, {size}"


def cmd_overview(args):
    """The short table of the top-level README: unique conversations only."""
    print("| setup | weights | reusing the prompt cache | sharing the server with other requests |")
    print("| --- | --- | --- | --- |")
    for label, d in split(args.dirs):
        for w in args.weights:
            path = os.path.join(d, f"{w}.selfcheck.unique.json")
            if os.path.exists(path):
                c = json.load(open(path))["checks"]
                print(f"| {label} | {w} | {brief(c['cache reuse'])} | {brief(c['repeat under load'])} |")


def row(label, name, c):
    s = c.summary()
    print(f"| {label} | {name} | {c.identical_outputs}/{c.prompts} | {pct(s['argmax_agreement'])} | "
          f"{num(s['drift_p50'])} | {num(s['drift_p99'])} | {verdict(c)} |")


HEADER = ("| setup | comparison | identical outputs | argmax agreement | drift p50 | drift p99 | verdict |\n"
          "| --- | --- | --- | --- | --- | --- | --- |")


def scratch(d, w):
    return checks.load(os.path.join(d, f"{w}.scratch.jsonl.gz"))


def cmd_weights(args):
    print(HEADER)
    for label, d in split(args.dirs):
        for step in args.steps:
            a, b = step.split(":")
            row(label, f"{a} -> {b}", compare(scratch(d, a), scratch(d, b)))


def cmd_across(args):
    (la, da), (lb, db) = split(args.dirs)
    print(HEADER)
    for w in args.weights:
        row(f"{la} vs {lb}", w, compare(scratch(da, w), scratch(db, w)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("selfcheck")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--weights", nargs="+", help="row order (default: alphabetical)")
    p.set_defaults(f=cmd_selfcheck)

    p = sub.add_parser("overview")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--weights", nargs="+", required=True)
    p.set_defaults(f=cmd_overview)

    p = sub.add_parser("weights")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--steps", nargs="+", required=True)
    p.set_defaults(f=cmd_weights)

    p = sub.add_parser("across")
    p.add_argument("dirs", nargs=2)
    p.add_argument("--weights", nargs="+", required=True)
    p.set_defaults(f=cmd_across)

    args = ap.parse_args()
    args.f(args)


if __name__ == "__main__":
    main()
