#!/usr/bin/env python3
"""Run remote_norm_check.py over a list of Hugging Face repositories.

Each line of the target file is: label, GGUF repository, file name pattern, checkpoint repository.
The first GGUF file matching the pattern is checked (for split files, the first part).
One JSON object per target is printed.

Usage: scan.py targets.txt > data/scan.jsonl
"""
import json
import os
import re
import subprocess
import sys
import urllib.request

HF = "https://huggingface.co"


def files(repo):
    req = urllib.request.Request(f"{HF}/api/models/{repo}", headers={"User-Agent": "gguf-header-reader"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return [s["rfilename"] for s in json.load(r).get("siblings", [])]


def main(target_file):
    check = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote_norm_check.py")
    for line in open(target_file):
        if not line.strip() or line.startswith("#"):
            continue
        label, gguf_repo, pattern, ckpt_repo = line.split()
        res = {}
        try:
            gguf = sorted(f for f in files(gguf_repo) if f.endswith(".gguf") and re.search(pattern, f, re.I))
            ckpt = sorted(f for f in files(ckpt_repo) if f.endswith(".safetensors") and f.startswith("model"))
            if not gguf or not ckpt:
                raise RuntimeError(f"no matching file: {len(gguf)} GGUF, {len(ckpt)} safetensors")
            cmd = [sys.executable, check, f"{HF}/{gguf_repo}/resolve/main/{gguf[0]}"]
            cmd += [f"{HF}/{ckpt_repo}/resolve/main/{f}" for f in ckpt]
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            if run.returncode:
                raise RuntimeError(run.stderr.strip().splitlines()[-1])
            res = json.loads(run.stdout.strip().splitlines()[-1])
            res["gguf"] = f"{gguf_repo}/{gguf[0]}"
        except Exception as e:  # noqa: BLE001 - one failed target must not stop the scan
            res = {"gguf": gguf_repo, "error": str(e)[:200]}
        res.update(label=label, checkpoint=ckpt_repo)
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.strip())
    main(sys.argv[1])
