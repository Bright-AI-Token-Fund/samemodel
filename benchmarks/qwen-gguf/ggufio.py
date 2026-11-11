"""Minimal readers for GGUF and safetensors files, enough to compare weights. Needs numpy."""
import json
import os
import struct

import numpy as np

GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
              10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16"}
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read_gguf_header(f):
    """Parse a GGUF header from a file-like object. Returns (metadata, tensor infos, data offset).

    Metadata arrays are skipped: only scalar and string values are kept.
    A tensor info is (name, dims, ggml type, offset relative to the data section).
    """
    def u(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

    def string():
        return f.read(u("Q")).decode("utf-8", "replace")

    def value(t):
        if t in _SCALAR:
            return u(_SCALAR[t])
        if t == 8:
            return string()
        if t == 9:
            elem, n = u("I"), u("Q")
            if elem in _SCALAR:
                f.read(n * struct.calcsize("<" + _SCALAR[elem]))
            else:
                for _ in range(n):
                    value(elem)
            return None
        raise ValueError(f"unknown GGUF value type {t}")

    if f.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    u("I")
    n_tensors, n_kv = u("Q"), u("Q")
    meta = {}
    for _ in range(n_kv):
        key = string()
        val = value(u("I"))
        if val is not None:
            meta[key] = val
    infos = []
    for _ in range(n_tensors):
        name = string()
        dims = [u("Q") for _ in range(u("I"))]
        infos.append((name, dims, u("I"), u("Q")))
    align = meta.get("general.alignment", 32)
    return meta, infos, (f.tell() + align - 1) // align * align


def decode(raw, dtype):
    """Raw little-endian bytes of dtype F32, F16 or BF16 as a float64 array."""
    if dtype == "BF16":
        return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return np.frombuffer(raw, dtype={"F16": "<f2", "F32": "<f4"}[dtype]).astype(np.float64)


class GGUF:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.meta, infos, self.base = read_gguf_header(self.f)
        self.info = {name: (dims, t, off) for name, dims, t, off in infos}
        order = sorted(infos, key=lambda x: x[3])
        size = os.path.getsize(path) - self.base
        ends = [x[3] for x in order[1:]] + [size]
        self.nbytes = {x[0]: end - x[3] for x, end in zip(order, ends, strict=True)}

    def type_name(self, name):
        return GGML_TYPES.get(self.info[name][1], str(self.info[name][1]))

    def raw(self, name):
        self.f.seek(self.base + self.info[name][2])
        return self.f.read(self.nbytes[name])

    def get(self, name):
        """An unquantised tensor as float64, in [out, in] layout."""
        dims, t, off = self.info[name]
        if t not in (0, 1):
            raise ValueError(f"{name} is {self.type_name(name)}, not F32 or F16")
        self.f.seek(self.base + off)
        n = int(np.prod(dims))
        return decode(self.f.read(n * (4 if t == 0 else 2)), "F32" if t == 0 else "F16").reshape(dims[::-1])


class Safetensors:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = {k: v for k, v in json.loads(self.f.read(n)).items() if k != "__metadata__"}
        self.base = 8 + n

    def get(self, name):
        v = self.header[name]
        start, end = v["data_offsets"]
        self.f.seek(self.base + start)
        return decode(self.f.read(end - start), v["dtype"]).reshape(v["shape"])
