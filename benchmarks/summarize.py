#!/usr/bin/env python3
"""Turn the output directories of llamacpp.sh / mlx.sh into the Markdown tables of
benchmarks/README.md, so that no number in that file is typed by hand.

Usage:
  summarize.py selfcheck LABEL=DIR [LABEL=DIR ...]
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
from samemodel.report import verdict


def pct(x):
    return "n/a" if x is None else f"{100 * x:.2f}%"


def num(x):
    return "n/a" if x is None else f"{x:.4f}"


def cell(s):
    return f"{s['verdict']} ({s['identical_outputs']}/{s['prompts']}, {num(s['drift_p50'])})"


def split(pairs):
    return [p.split("=", 1) for p in pairs]


def cmd_selfcheck(args):
    print("| setup | weights | conversations | cache reuse | first request under load | "
          "repeat under load | cached on first / repeat request |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for label, d in split(args.dirs):
        for path in sorted(glob.glob(os.path.join(d, "*.selfcheck.*.json"))):
            w, _, kind, _ = os.path.basename(path).split(".")
            c = json.load(open(path))["checks"]
            reuse = c["cache reuse"]
            first, rep = reuse.get("scratch_hit_ratio"), reuse.get("cache_hit_ratio")
            cached = "n/a" if rep is None else f"{100 * (first or 0):.0f}% / {100 * rep:.0f}%"
            cells = [cell(c[k]) if k in c else "n/a"
                     for k in ("cache reuse", "first request under load", "repeat under load")]
            print(f"| {label} | {w} | {kind} | {' | '.join(cells)} | {cached} |")


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
    p.set_defaults(f=cmd_selfcheck)

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
