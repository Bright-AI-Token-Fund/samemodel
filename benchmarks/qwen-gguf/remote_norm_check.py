#!/usr/bin/env python3
"""Does a GGUF file hold the same normalisation weights as the checkpoint it was made from?

Norm weights are stored unquantised, so the comparison is exact, and they are tiny. The script
reads the GGUF header and the norm tensors with HTTP range requests, and the matching tensors from
the safetensors checkpoint the same way. Neither file is downloaded: about 10 MB is fetched.

Prints one JSON object. `identical` counts tensors equal in every element.

Usage: remote_norm_check.py GGUF_URL SAFETENSORS_URL [SAFETENSORS_URL ...]
"""
import http.client
import json
import struct
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ggufio import GGML_TYPES, decode, read_gguf_header

CKPT_NORM = {"attn_norm.weight": "input_layernorm.weight",
             "ffn_norm.weight": "post_attention_layernorm.weight"}


class Remote:
    """A read-only file over HTTP range requests, buffered for sequential header parsing."""

    def __init__(self, url, chunk=1 << 22):
        self.url, self.chunk, self.pos, self.start, self.buf, self.fetched = url, chunk, 0, 0, b"", 0

    def fetch(self, a, b):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b - 1}",
                                                        "User-Agent": "gguf-header-reader"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = r.read()
                break
            except (OSError, http.client.HTTPException):
                if attempt == 3:
                    raise
                time.sleep(2 + 3 * attempt)
        self.fetched += len(data)
        return data

    def tell(self):
        return self.pos

    def read(self, n):
        if not (self.start <= self.pos and self.pos + n <= self.start + len(self.buf)):
            self.start, self.buf = self.pos, self.fetch(self.pos, self.pos + max(n, self.chunk))
        o = self.pos - self.start
        self.pos += n
        return self.buf[o:o + n]


def checkpoint_name(gguf_name):
    p = gguf_name.split(".")
    if p[0] == "blk" and ".".join(p[2:]) in CKPT_NORM:
        return f"model.layers.{p[1]}.{CKPT_NORM['.'.join(p[2:])]}"
    return "model.norm.weight" if gguf_name == "output_norm.weight" else None


def gguf_norms(url):
    f = Remote(url)
    meta, infos, base = read_gguf_header(f)
    want = [x for x in infos if x[0].endswith("norm.weight") and len(x[1]) == 1 and x[2] in (0, 1)]

    def one(x):
        name, dims, t, off = x
        raw = f.fetch(base + off, base + off + dims[0] * (4 if t == 0 else 2))
        return name, decode(raw, "F32" if t == 0 else "F16")

    with ThreadPoolExecutor(12) as pool:
        norms = dict(pool.map(one, want))
    types = {}
    for _name, _dims, t, _off in infos:
        types[GGML_TYPES.get(t, str(t))] = types.get(GGML_TYPES.get(t, str(t)), 0) + 1
    return meta, norms, types, len(infos), f


def checkpoint_norms(urls, wanted):
    found = {}
    for url in urls:
        f = Remote(url, chunk=1 << 20)
        n = struct.unpack("<Q", f.read(8))[0]
        header, base = json.loads(f.read(n)), 8 + n

        def one(kv, f=f, base=base):
            start, end = kv[1]["data_offsets"]
            return kv[0], decode(f.fetch(base + start, base + end), kv[1]["dtype"])

        todo = [(k, v) for k, v in header.items() if k in wanted and k not in found]
        with ThreadPoolExecutor(12) as pool:
            found.update(pool.map(one, todo))
        if len(found) == len(wanted):
            break
    return found


def main(gguf_url, checkpoint_urls):
    meta, norms, types, n_tensors, f = gguf_norms(gguf_url)
    names = {n: checkpoint_name(n) for n in norms if checkpoint_name(n)}
    ckpt = checkpoint_norms(checkpoint_urls, set(names.values()))
    identical = rounding = different = 0
    worst = 0.0
    for name, ck in names.items():
        if ck not in ckpt or len(ckpt[ck]) != len(norms[name]):
            continue
        d = float(np.max(np.abs(norms[name] - ckpt[ck])))
        worst = max(worst, d)
        if d == 0:
            identical += 1
        elif d < 2e-3 * max(1.0, float(np.max(np.abs(ckpt[ck])))):
            rounding += 1
        else:
            different += 1
    print(json.dumps({
        "name": meta.get("general.name"), "quantized_by": meta.get("general.quantized_by"),
        "file_type": meta.get("general.file_type"), "imatrix": meta.get("quantize.imatrix.file"),
        "tensors": n_tensors, "types": dict(sorted(types.items(), key=lambda kv: -kv[1])),
        "norm_tensors_compared": identical + rounding + different, "identical": identical,
        "rounding_only": rounding, "different": different, "max_abs_diff": float(f"{worst:.3g}"),
        "gguf_bytes_fetched": f.fetched}))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__.strip())
    main(sys.argv[1], sys.argv[2:])
