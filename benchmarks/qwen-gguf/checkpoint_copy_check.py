#!/usr/bin/env python3
"""Is a local safetensors file a faithful copy of the published checkpoint?

Fetches a sample of tensors from the published file with HTTP range requests and compares values.

Usage: checkpoint_copy_check.py local.safetensors PUBLISHED_URL
"""
import json
import struct
import sys
import urllib.request

import numpy as np
from ggufio import Safetensors, decode

SAMPLE = [
    "model.layers.0.self_attn.q_proj.weight", "model.layers.11.self_attn.v_proj.weight",
    "model.layers.23.self_attn.o_proj.weight", "model.layers.11.input_layernorm.weight",
    "model.layers.23.post_attention_layernorm.weight", "model.norm.weight",
    "model.layers.5.self_attn.k_proj.bias",
]


def fetch(url, a, b):
    headers = {"Range": f"bytes={a}-{b - 1}", "User-Agent": "gguf-header-reader"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def main(local_path, url):
    local = Safetensors(local_path)
    n = struct.unpack("<Q", fetch(url, 0, 8))[0]
    header = {k: v for k, v in json.loads(fetch(url, 8, 8 + n)).items() if k != "__metadata__"}
    print(f"tensors: local {len(local.header)}, published {len(header)}; "
          f"published dtypes {sorted({v['dtype'] for v in header.values()})}")
    for name in SAMPLE:
        v = header[name]
        start, end = v["data_offsets"]
        published = decode(fetch(url, 8 + n + start, 8 + n + end), v["dtype"])
        d = np.max(np.abs(local.get(name).ravel() - published))
        print(f"{name:50s} max |local - published| = {d:.1e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__.strip())
    main(sys.argv[1], sys.argv[2])
