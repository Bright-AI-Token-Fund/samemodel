#!/usr/bin/env python3
"""Run remote_norm_check.py on the model layer of a tag in the Ollama registry.

Usage: ollama_registry_check.py library/qwen2.5:7b Qwen/Qwen2.5-7B-Instruct
"""
import json
import os
import subprocess
import sys
import urllib.request

REGISTRY = "https://registry.ollama.ai/v2"
HF = "https://huggingface.co"


def get_json(url, accept=None):
    headers = {"User-Agent": "gguf-header-reader"}
    if accept:
        headers["Accept"] = accept
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        return json.load(r)


def main(ref, ckpt_repo):
    name, tag = ref.split(":")
    manifest = get_json(f"{REGISTRY}/{name}/manifests/{tag}",
                        "application/vnd.docker.distribution.manifest.v2+json")
    layer = next(x for x in manifest["layers"] if x["mediaType"].endswith("image.model"))
    siblings = get_json(f"{HF}/api/models/{ckpt_repo}")["siblings"]
    ckpt = sorted(s["rfilename"] for s in siblings
                  if s["rfilename"].endswith(".safetensors") and s["rfilename"].startswith("model"))
    check = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote_norm_check.py")
    cmd = [sys.executable, check, f"{REGISTRY}/{name}/blobs/{layer['digest']}"]
    cmd += [f"{HF}/{ckpt_repo}/resolve/main/{f}" for f in ckpt]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if run.returncode:
        res = {"error": run.stderr.strip().splitlines()[-1][:200]}
    else:
        res = json.loads(run.stdout.strip().splitlines()[-1])
    res.update(label=f"ollama {ref}", gguf=f"{ref} ({layer['digest'][:19]})", checkpoint=ckpt_repo)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__.strip())
    main(sys.argv[1], sys.argv[2])
