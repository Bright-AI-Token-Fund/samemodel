"""A tiny OpenAI-compatible server backed by a deterministic toy language model.

It exists so that samemodel can be demonstrated and tested without a GPU or a real
model. The toy model maps a context to a next-token distribution through a hash, so
identical requests give identical answers. Two knobs make the answers depend on the
serving path, the way real engines can:

  cache_drift   perturb the logits when the same prompt has been seen before
  batch_drift   perturb the logits when other requests are in flight

Both are in nats (standard deviation of the perturbation added to each logit).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional, Tuple

_SYLLABLES = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]
VOCAB = 4096
CANDIDATES = 12


def token_text(token_id: int) -> str:
    n = len(_SYLLABLES)
    return " " + _SYLLABLES[token_id % n] + _SYLLABLES[(token_id // n) % n]


def _rng(*parts: bytes) -> random.Random:
    return random.Random(hashlib.blake2b(b"\x00".join(parts), digest_size=16).digest())


class ToyModel:
    def __init__(self, cache_drift: float = 0.0, batch_drift: float = 0.0,
                 resolution: Optional[float] = None):
        self.cache_drift = cache_drift
        self.batch_drift = batch_drift
        self.resolution = resolution

    def step(self, context: bytes, cached: bool, batched: bool) -> List[Tuple[int, float]]:
        """Top candidates for the next token as (token id, logprob), best first."""
        base = _rng(context)
        ids = base.sample(range(VOCAB), CANDIDATES)
        logits = [base.gauss(0.0, 2.0) for _ in ids]
        for active, amount, label in ((cached, self.cache_drift, b"cache"),
                                      (batched, self.batch_drift, b"batch")):
            if active and amount > 0:
                noise = _rng(context, label)
                logits = [x + noise.gauss(0.0, amount) for x in logits]
        norm = math.log(sum(math.exp(x) for x in logits))
        scored = [(i, x - norm) for i, x in zip(ids, logits, strict=True)]
        if self.resolution:
            scored = [(i, round(lp / self.resolution) * self.resolution) for i, lp in scored]
        scored.sort(key=lambda p: (-p[1], p[0]))
        return scored

    def generate(self, prompt: str, max_tokens: int, cached: bool, batched: bool):
        context = prompt.encode()
        out = []
        for _ in range(max_tokens):
            scored = self.step(context, cached, batched)
            out.append(scored)
            context += b"|" + str(scored[0][0]).encode()
        return out


class MockServer:
    """Run with ``with MockServer(...) as url:`` or call start()/stop()."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, cache_drift: float = 0.0,
                 batch_drift: float = 0.0, resolution: Optional[float] = None,
                 logprobs: bool = True, token_ids: bool = True, window_ms: float = 30.0):
        self.model = ToyModel(cache_drift, batch_drift, resolution)
        self.logprobs = logprobs
        self.token_ids = token_ids
        self.window = window_ms / 1000.0
        self._seen = set()
        self._in_flight = 0
        self._arrivals = 0
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output quiet
                pass

            def _send(self, code: int, body: dict) -> None:
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path.rstrip("/") == "/v1/models":
                    self._send(200, {"object": "list", "data": [{"id": "toy", "object": "model"}]})
                elif self.path.rstrip("/") == "/health":
                    self._send(200, {"status": "ok"})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                if self.path.rstrip("/") != "/v1/chat/completions":
                    self._send(404, {"error": "not found"})
                    return
                try:
                    req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    prompt = "\n".join(str(m.get("content", "")) for m in req["messages"])
                except (ValueError, KeyError, TypeError):
                    self._send(400, {"error": "invalid request"})
                    return
                if req.get("stream"):
                    self._send(400, {"error": "the mock server does not stream"})
                    return
                self._send(200, outer.complete(prompt, req))

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._httpd.daemon_threads = True
        self.url = f"http://{host}:{self._httpd.server_address[1]}"
        self._thread: Optional[threading.Thread] = None

    def complete(self, prompt: str, req: dict) -> dict:
        with self._lock:
            self._in_flight += 1
            self._arrivals += 1
            arrivals = self._arrivals
            batched = self._in_flight > 1
            cached = prompt in self._seen
            self._seen.add(prompt)
        try:
            # a batching window: anything that arrives while we wait shares the batch
            time.sleep(self.window)
            with self._lock:
                batched = batched or self._arrivals > arrivals
            max_tokens = int(req.get("max_tokens") or 16)
            steps = self.model.generate(prompt, max_tokens, cached, batched)
        finally:
            with self._lock:
                self._in_flight -= 1
        text = "".join(token_text(s[0][0]) for s in steps)
        choice = {"index": 0, "finish_reason": "length",
                  "message": {"role": "assistant", "content": text}}
        if self.logprobs and req.get("logprobs"):
            k = max(1, int(req.get("top_logprobs") or 1))
            content = []
            for scored in steps:
                top = [self._entry(i, lp) for i, lp in scored[:k]]
                content.append(dict(top[0], top_logprobs=top))
            choice["logprobs"] = {"content": content}
        n_prompt = len(prompt.split())
        return {"object": "chat.completion", "model": "toy", "choices": [choice],
                "usage": {"prompt_tokens": n_prompt, "completion_tokens": len(steps),
                          "total_tokens": n_prompt + len(steps),
                          "prompt_tokens_details": {"cached_tokens": n_prompt if cached else 0}}}

    def _entry(self, token_id: int, logprob: float) -> dict:
        entry = {"token": token_text(token_id), "logprob": logprob}
        if self.token_ids:
            entry["id"] = token_id
        return entry

    def start(self) -> str:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.url

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def serve_forever(self) -> None:
        self._httpd.serve_forever()

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
