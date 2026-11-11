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
    """One GGUF file, or the parts of a split one (pass a list of paths)."""

    def __init__(self, paths):
        self.paths = [paths] if isinstance(paths, str) else list(paths)
        self.meta, self.info, self.nbytes, self._where = {}, {}, {}, {}
        for path in self.paths:
            f = open(path, "rb")
            meta, infos, base = read_gguf_header(f)
            self.meta = self.meta or meta
            order = sorted(infos, key=lambda x: x[3])
            ends = [x[3] for x in order[1:]] + [os.path.getsize(path) - base]
            for (name, dims, t, off), end in zip(order, ends, strict=True):
                self.info[name] = (dims, t, off)
                self.nbytes[name] = end - off
                self._where[name] = (f, base)

    def type_name(self, name):
        return GGML_TYPES.get(self.info[name][1], str(self.info[name][1]))

    def offset(self, name):
        """(open file, absolute offset) of a tensor's data."""
        f, base = self._where[name]
        return f, base + self.info[name][2]

    def raw(self, name):
        f, pos = self.offset(name)
        f.seek(pos)
        return f.read(self.nbytes[name])

    def get(self, name):
        """An unquantised tensor as float64, in [out, in] layout."""
        dims, t, _off = self.info[name]
        if t not in (0, 1):
            raise ValueError(f"{name} is {self.type_name(name)}, not F32 or F16")
        f, pos = self.offset(name)
        f.seek(pos)
        n = int(np.prod(dims))
        return decode(f.read(n * (4 if t == 0 else 2)), "F32" if t == 0 else "F16").reshape(dims[::-1])


class Safetensors:
    """One safetensors file, or the shards of a checkpoint (pass a list of paths)."""

    def __init__(self, paths):
        self.header, self._where = {}, {}
        for path in [paths] if isinstance(paths, str) else list(paths):
            f = open(path, "rb")
            n = struct.unpack("<Q", f.read(8))[0]
            for k, v in json.loads(f.read(n)).items():
                if k != "__metadata__":
                    self.header[k] = v
                    self._where[k] = (f, 8 + n)

    def get(self, name):
        v = self.header[name]
        f, base = self._where[name]
        start, end = v["data_offsets"]
        f.seek(base + start)
        return decode(f.read(end - start), v["dtype"]).reshape(v["shape"])
