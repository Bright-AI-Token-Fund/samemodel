#!/usr/bin/env python3
"""Is a full-precision GGUF file a rescaled copy of a checkpoint, and what else was done to it?

For every block the script rebuilds each weight matrix from the checkpoint as

    checkpoint_weight * column_scale * row_scale

with the scales read off the GGUF file itself: the ratio of the norm weights for the layers that
read a norm output, medians of the element ratios otherwise. It then reports what in the file is
not explained by that rescale.

With --unclip COPY, COPY (a copy of the GGUF file) is patched in place so that every matrix is
exactly the rescaled checkpoint: whatever the file contains beyond a rescale is removed.

Usage: rescale_check.py full-precision.gguf model.safetensors [--unclip COPY.gguf]
"""
import sys

import numpy as np
from ggufio import GGUF, Safetensors

# layers that read a norm output: GGUF name -> (GGUF norm, checkpoint norm, checkpoint layer)
NORMED = {
    "attn_q": ("attn_norm", "input_layernorm", "self_attn.q_proj"),
    "attn_k": ("attn_norm", "input_layernorm", "self_attn.k_proj"),
    "attn_v": ("attn_norm", "input_layernorm", "self_attn.v_proj"),
    "ffn_gate": ("ffn_norm", "post_attention_layernorm", "mlp.gate_proj"),
    "ffn_up": ("ffn_norm", "post_attention_layernorm", "mlp.up_proj"),
}
OTHER = {"attn_output": "self_attn.o_proj", "ffn_down": "mlp.down_proj"}
GROUP = 128   # AutoAWQ clips per output row and per group of this many input channels


def median_ratio(a, b, axis):
    ok = np.abs(b) > 1e-5
    return np.nanmedian(np.where(ok, a / np.where(ok, b, 1), np.nan), axis=axis)


def rebuild(g, s, blk, kind):
    """Returns (matrix in the GGUF file, plain checkpoint matrix, rescaled checkpoint matrix)."""
    w_file = g.get(f"blk.{blk}.{kind}.weight")
    if kind in NORMED:
        norm, ck_norm, ck_layer = NORMED[kind]
        col = s.get(f"model.layers.{blk}.{ck_norm}.weight") / g.get(f"blk.{blk}.{norm}.weight")
        w_ck = s.get(f"model.layers.{blk}.{ck_layer}.weight")
        scaled = w_ck * col[None, :]
        return w_file, w_ck, scaled * median_ratio(w_file, scaled, 1)[:, None]
    w_ck = s.get(f"model.layers.{blk}.{OTHER[kind]}.weight")
    return w_file, w_ck, w_ck * median_ratio(w_file, w_ck, 0)[None, :]


def main(gguf_path, ckpt_path, unclip_path=None):
    g, s = GGUF(gguf_path), Safetensors(ckpt_path)
    n_blk = 1 + max(int(n.split(".")[1]) for n in g.info if n.startswith("blk."))
    out = open(unclip_path, "r+b") if unclip_path else None
    patched = GGUF(unclip_path) if unclip_path else None
    rows = {}
    for blk in range(n_blk):
        for kind in list(NORMED) + list(OTHER):
            w_file, w_ck, target = rebuild(g, s, blk, kind)
            r = rows.setdefault(kind, {"raw": [], "rescaled": [], "n": 0, "changed": 0, "smaller": 0,
                                       "groups": 0, "one_ceiling": 0})
            r["raw"].append(np.linalg.norm(w_file - w_ck) / np.linalg.norm(w_ck))
            r["rescaled"].append(np.linalg.norm(w_file - target) / np.linalg.norm(target))
            changed = np.abs(w_file - target) > 0.01 * np.abs(target) + 1e-6
            r["n"] += changed.size
            r["changed"] += int(changed.sum())
            r["smaller"] += int((changed & (np.abs(w_file) < np.abs(target))).sum())
            # within a row and a group of input channels: do the changed weights share one magnitude,
            # and is it the largest magnitude in the group?
            mag = np.abs(w_file).reshape(w_file.shape[0], -1, GROUP)
            ch = changed.reshape(w_file.shape[0], -1, GROUP)
            several = ch.sum(-1) >= 2
            hi = np.where(ch, mag, 0).max(-1)
            lo = np.where(ch, mag, np.inf).min(-1)
            flat = (hi - lo <= 1e-3 * hi) & np.isclose(hi, mag.max(-1), rtol=2e-3)
            r["groups"] += int(several.sum())
            r["one_ceiling"] += int((several & flat).sum())
            if out:
                dims, t, off = patched.info[f"blk.{blk}.{kind}.weight"]
                if t != 1:
                    raise SystemExit("--unclip expects F16 tensors")
                out.seek(patched.base + off)
                out.write(target.astype("<f2").tobytes())
    print(f"{n_blk} blocks. Relative differences are Frobenius norms, |file - reference| / |reference|.\n")
    print("layer        file vs checkpoint   file vs rescaled checkpoint   weights not explained   "
          "of those, smaller   groups where they share one ceiling")
    print("             (median over blocks) (largest over blocks)         by the rescale (>1%)    "
          "in magnitude")
    for kind, r in rows.items():
        share = f"{r['one_ceiling']}/{r['groups']}" if r["groups"] else "-"
        smaller = f"{r['smaller'] / r['changed']:.1%}" if r["changed"] else "-"
        print(f"{kind:12s} {np.median(r['raw']):12.3f} {max(r['rescaled']):22.1e} {r['changed']:20d} = "
              f"{r['changed'] / r['n']:5.2%} {smaller:>14s}      {share}")
    if out:
        out.close()
        print(f"\npatched {unclip_path}: every matrix is now the rescaled checkpoint")


if __name__ == "__main__":
    args = sys.argv[1:]
    unclip = None
    if "--unclip" in args:
        i = args.index("--unclip")
        unclip = args[i + 1]
        del args[i:i + 2]
    if len(args) != 2:
        raise SystemExit(__doc__.strip())
    main(args[0], args[1], unclip)
