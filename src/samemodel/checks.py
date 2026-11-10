"""The measurements samemodel runs against live endpoints."""
from __future__ import annotations

import gzip
import hashlib
import json
import secrets
from typing import Callable, Dict, List, Optional, Sequence

from .client import Client, Record
from .compare import Comparison, compare

Progress = Optional[Callable[[int, int], None]]


def marker(nonce: Optional[str], label: str) -> Optional[str]:
    """A system message that makes a conversation unique from its very first tokens.

    A server with a prefix cache reuses whatever prefix a request shares with earlier
    ones, and that alone can change the computation. Putting a per-run marker at the
    start of the system message leaves only the first few template tokens shared, so
    the first request for each prompt is computed (almost) from scratch.
    """
    if not nonce:
        return None
    # hash so that two markers of the same run already differ in their first characters
    unique = hashlib.blake2b(f"{nonce}:{label}".encode(), digest_size=5).hexdigest()
    return f"{unique} is the id of this session. You are a helpful assistant."


def record_run(client: Client, prompts: Sequence[str], nonce: Optional[str] = None,
               progress: Progress = None) -> List[Record]:
    out = []
    for i, p in enumerate(prompts):
        out.append(client.record(f"p{i:03d}", p, system=marker(nonce, str(i))))
        if progress:
            progress(i + 1, len(prompts))
    return out


def selfcheck(client: Client, prompts: Sequence[str], companions: int = 3,
              nonce: Optional[str] = "auto", progress: Progress = None) -> Dict[str, Comparison]:
    """Does one endpoint answer the same way under different serving conditions?

    Each prompt is sent three times:
      scratch  first time the server sees this conversation
      cached   the identical request again (reusing the prompt cache, if any)
      busy     again, at the same moment as ``companions`` unrelated requests

    ``cache reuse`` compares cached against scratch; ``co-batched`` compares busy
    against cached, so that both sides of that comparison start from the same cache
    state.
    """
    if nonce == "auto":
        nonce = secrets.token_hex(4)
    scratch: List[Record] = []
    cached: List[Record] = []
    busy: List[Record] = []
    for i, p in enumerate(prompts):
        pid, system = f"p{i:03d}", marker(nonce, str(i))
        scratch.append(client.record(pid, p, system=system))
        cached.append(client.record(pid, p, system=system))
        if companions > 0:
            # companions carry their own marker so they never warm the cache for a
            # prompt that is measured from scratch later
            others = [(marker(nonce or "c", f"c{i}-{k}"), prompts[(i + 7 * k) % len(prompts)])
                      for k in range(1, companions + 1)]
            busy.append(client.record(pid, p, system=system, companions=others))
        if progress:
            progress(i + 1, len(prompts))
    rows = {"cache reuse": compare(scratch, cached)}
    if companions > 0:
        rows[f"co-batched x{companions + 1}"] = compare(cached, busy)
    hits = [r["cached_tokens"] for r in cached if r.get("cached_tokens") is not None]
    total = [r["prompt_tokens"] for r in cached if r.get("prompt_tokens")]
    rows["cache reuse"].cache_hit_ratio = (sum(hits) / sum(total)) if hits and total else None
    first = [r["cached_tokens"] for r in scratch if r.get("cached_tokens") is not None]
    first_total = [r["prompt_tokens"] for r in scratch if r.get("prompt_tokens")]
    rows["cache reuse"].scratch_hit_ratio = (sum(first) / sum(first_total)) if first and first_total else None
    return rows


def _open(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def save(records: Sequence[Record], path: str) -> None:
    """Write one JSON record per line; a path ending in .gz is compressed."""
    with _open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load(path: str) -> List[Record]:
    with _open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]
