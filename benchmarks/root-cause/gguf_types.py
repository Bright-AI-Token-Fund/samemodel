#!/usr/bin/env python3
"""List the tensor types in a GGUF file: how many tensors of each type, and which kinds.

Usage: gguf_types.py model.gguf
"""
import collections
import re
import struct
import sys

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
         10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16"}
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def rstr(f):
    return f.read(rd(f, "Q")).decode("utf-8", "replace")


def rval(f, t):
    if t in SCALAR:
        return rd(f, SCALAR[t])
    if t == 8:
        return rstr(f)
    if t == 9:
        et, n = rd(f, "I"), rd(f, "Q")
        return [rval(f, et) for _ in range(n)]
    raise ValueError(f"unknown GGUF value type {t}")


def main(path):
    by_type = collections.defaultdict(list)
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path} is not a GGUF file")
        _version, n_tensors, n_kv = rd(f, "I"), rd(f, "Q"), rd(f, "Q")
        for _ in range(n_kv):
            rstr(f)
            rval(f, rd(f, "I"))
        for _ in range(n_tensors):
            name = rstr(f)
            dims = [rd(f, "Q") for _ in range(rd(f, "I"))]
            t, _offset = rd(f, "I"), rd(f, "Q")
            by_type[TYPES.get(t, str(t))].append((name, dims))
    for t, items in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        print(f"{t:5s} {len(items):4d} tensors")
        kinds = collections.Counter((re.sub(r"blk\.\d+\.", "blk.N.", n), tuple(d)) for n, d in items)
        for (kind, dims), count in sorted(kinds.items(), key=lambda kv: -kv[1]):
            if len(dims) > 1:
                print(f"        {count:3d} x {kind:28s} {list(dims)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.strip())
    main(sys.argv[1])
