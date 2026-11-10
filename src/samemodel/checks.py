"""The measurements samemodel runs against live endpoints."""
from __future__ import annotations

import gzip
import hashlib
import json
import secrets
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .client import Client, Record
from .compare import Comparison, compare

Progress = Optional[Callable[[int, int], None]]


SHARED_SYSTEM = "You are a helpful assistant."


def _unique(nonce: str, label: str) -> str:
    # hash so that two markers of the same run already differ in their first characters
    return hashlib.blake2b(f"{nonce}:{label}".encode(), digest_size=5).hexdigest()


def conversation(prompt: str, nonce: Optional[str], label: str,
                 shared_system: bool = False) -> Tuple[Optional[str], str]:
    """Return (system message, user message) for one measured prompt.

    A server with a prefix cache reuses whatever prefix a request shares with earlier
    ones, and that alone can change the computation. By default the per-run marker
    opens the system message, so only the first few template tokens are shared and the
    first request for each prompt is computed (almost) from scratch.

    With ``shared_system`` every conversation has the same system message and the
    marker opens the user message instead. That is how most deployments look: the
    first request for a prompt then already reuses the cached system prefix.
    """
    if not nonce:
        return None, prompt
    unique = _unique(nonce, label)
    if shared_system:
        return SHARED_SYSTEM, f"[{unique}] {prompt}"
    return f"{unique} is the id of this session. {SHARED_SYSTEM}", prompt


def marker(nonce: Optional[str], label: str) -> Optional[str]:
    """The system message used for a unique conversation (see ``conversation``)."""
    return conversation("", nonce, label)[0]


def companions_for(prompts: Sequence[str], nonce: Optional[str], i: int, count: int,
                   shared_system: bool = False) -> List[Tuple[Optional[str], str]]:
    """Unrelated conversations to send alongside prompt ``i``.

    They carry their own markers so they never warm the cache for a prompt that is
    measured later.
    """
    return [conversation(prompts[(i + 7 * k) % len(prompts)], nonce or "c", f"c{i}-{k}", shared_system)
            for k in range(1, count + 1)]


def warm_up(client: Client, nonce: Optional[str], shared_system: bool) -> None:
    """With a shared system message, let the server see it once before measuring.

    On a freshly started server the very first conversation would otherwise be the
    only one that finds nothing to reuse, an outlier among requests that all do.
    """
    if shared_system and nonce:
        system, user = conversation("Say hello.", nonce, "warm-up", shared_system=True)
        client.warm(user, system)


def record_run(client: Client, prompts: Sequence[str], nonce: Optional[str] = None,
               companions: int = 0, shared_system: bool = False,
               progress: Progress = None) -> List[Record]:
    warm_up(client, nonce, shared_system)
    out = []
    for i, p in enumerate(prompts):
        system, user = conversation(p, nonce, str(i), shared_system)
        out.append(client.record(f"p{i:03d}", user, system=system,
                                 companions=companions_for(prompts, nonce, i, companions, shared_system)))
        if progress:
            progress(i + 1, len(prompts))
    return out


REUSE = "cache reuse"
FIRST_LOADED = "first request under load"
REPEAT_LOADED = "repeat under load"


def selfcheck(client: Client, prompts: Sequence[str], companions: int = 3,
              nonce: Optional[str] = "auto", shared_system: bool = False,
              progress: Progress = None) -> Dict[str, Comparison]:
    """Does one endpoint answer the same way under different serving conditions?

    Each prompt is sent three times:
      first   the first time the server sees this conversation
      repeat  the identical request again, alone (reusing the prompt cache, if any)
      loaded  again, at the same moment as ``companions`` unrelated requests

    A caching server can compute a conversation for the first time only once, so the
    prompts are split: every second prompt makes its *first* request under load.

      cache reuse               first (alone)  vs repeat, on the prompts that started alone
      first request under load  first (loaded) vs repeat, on the other prompts; this also
                                contains the cache reuse effect, so read it against the row above
      repeat under load         repeat vs loaded, on all prompts
    """
    if nonce == "auto":
        nonce = secrets.token_hex(4)
    warm_up(client, nonce, shared_system)
    first: List[Record] = []
    repeat: List[Record] = []
    loaded: List[Record] = []
    started_loaded = set()
    for i, p in enumerate(prompts):
        pid = f"p{i:03d}"
        system, user = conversation(p, nonce, str(i), shared_system)
        if companions > 0 and i % 2 == 1:
            started_loaded.add(pid)
            others = companions_for(prompts, nonce, i, companions, shared_system)
            first.append(client.record(pid, user, system=system, companions=others))
        else:
            first.append(client.record(pid, user, system=system))
        repeat.append(client.record(pid, user, system=system))
        if companions > 0:
            # new companions, so that the load again includes prompt processing
            others = companions_for(prompts, f"{nonce}-again", i, companions, shared_system)
            loaded.append(client.record(pid, user, system=system, companions=others))
        if progress:
            progress(i + 1, len(prompts))

    alone = [r for r in first if r["id"] not in started_loaded]
    rows = {REUSE: compare(alone, repeat)}
    if started_loaded:
        rows[FIRST_LOADED] = compare([r for r in first if r["id"] in started_loaded], repeat)
    if companions > 0:
        rows[REPEAT_LOADED] = compare(repeat, loaded)

    def ratio(records: Sequence[Record]) -> Optional[float]:
        hits = [r["cached_tokens"] for r in records if r.get("cached_tokens") is not None]
        total = [r["prompt_tokens"] for r in records if r.get("prompt_tokens")]
        return (sum(hits) / sum(total)) if hits and total else None

    rows[REUSE].cache_hit_ratio = ratio(repeat)
    rows[REUSE].scratch_hit_ratio = ratio(first)
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
