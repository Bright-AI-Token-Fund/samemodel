#!/usr/bin/env python3
"""Print the scan results as the Markdown table used in README.md.

Usage: scan_table.py data/scan.jsonl
"""
import json
import sys


def main(path):
    print("| file | `general.name` in the file | norm tensors compared | identical | different | "
          "largest difference |")
    print("| --- | --- | --- | --- | --- | --- |")
    for line in open(path):
        r = json.loads(line)
        if "error" in r:
            print(f"| `{r['gguf']}` | | could not be read: {r['error']} | | | |")
            continue
        worst = "0" if r["max_abs_diff"] == 0 else f"{r['max_abs_diff']:.3g}"
        same = str(r["identical"])
        if r["rounding_only"]:
            same += f" (+{r['rounding_only']} within rounding)"
        print(f"| `{r['gguf']}` | {r['name']} | {r['norm_tensors_compared']} | {same} | {r['different']} | "
              f"{worst} |")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.strip())
    main(sys.argv[1])
