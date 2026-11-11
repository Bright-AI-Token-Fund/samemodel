#!/usr/bin/env python3
"""Compare two GGUF files tensor by tensor: which tensors hold the same bytes?

Usage: tensor_diff.py a.gguf b.gguf
"""
import collections
import hashlib
import re
import sys

from ggufio import GGUF


def main(path_a, path_b):
    a, b = GGUF(path_a), GGUF(path_b)
    print("only in a:", sorted(set(a.info) - set(b.info)))
    print("only in b:", sorted(set(b.info) - set(a.info)))
    count, example = collections.Counter(), {}
    for name in sorted(set(a.info) & set(b.info)):
        same = hashlib.sha256(a.raw(name)).digest() == hashlib.sha256(b.raw(name)).digest()
        key = (a.type_name(name), b.type_name(name), "same bytes" if same else "different bytes")
        count[key] += 1
        example.setdefault(key, re.sub(r"blk\.\d+\.", "blk.N.", name))
    for key, n in sorted(count.items(), key=lambda kv: -kv[1]):
        print(f"{n:4d}  a={key[0]:5s} b={key[1]:5s} {key[2]:16s} e.g. {example[key]}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__.strip())
    main(sys.argv[1], sys.argv[2])
