"""Minimal OpenAI-compatible chat client that records tokens and their logprobs.

Standard library only, so the tool installs anywhere a Python interpreter exists.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

Record = Dict[str, Any]


class EndpointError(RuntimeError):
    pass


def _url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base + "/v1" + path


def _request(url: str, payload: Optional[dict], api_key: Optional[str], timeout: float) -> dict:
    headers = {"Content-Type": "application/json", "User-Agent": "samemodel"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:500]
        raise EndpointError(f"{url} returned HTTP {e.code}: {body}") from None
    except urllib.error.URLError as e:
        raise EndpointError(f"cannot reach {url}: {e.reason}") from None


def default_model(base_url: str, api_key: Optional[str] = None, timeout: float = 30) -> str:
    """First model id the endpoint advertises, or "default" when it lists none."""
    try:
        data = _request(_url(base_url, "/models"), None, api_key, timeout)
        models = data.get("data") or []
        if models and models[0].get("id"):
            return str(models[0]["id"])
    except (EndpointError, ValueError):
        pass
    return "default"


class Client:
    def __init__(self, base_url: str, model: Optional[str] = None, api_key: Optional[str] = None,
                 max_tokens: int = 128, top_logprobs: int = 10, seed: Optional[int] = None,
                 extra_body: Optional[dict] = None, timeout: float = 600):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model or default_model(base_url, api_key)
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs
        self.seed = seed
        self.extra_body = extra_body or {}
        self.timeout = timeout

    def _payload(self, prompt: str, logprobs: bool, system: Optional[str] = None) -> dict:
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        # No seed unless asked for. Greedy decoding does not need one, and a request
        # that carries one can be served on another path: mlx_lm.server takes it out
        # of batching, which hides the very effect being measured.
        if self.seed is not None:
            payload["seed"] = self.seed
        if logprobs and self.top_logprobs > 0:
            payload["logprobs"] = True
            payload["top_logprobs"] = self.top_logprobs
        payload.update(self.extra_body)
        return payload

    def record(self, prompt_id: str, prompt: str, system: Optional[str] = None,
               companions: Sequence[Tuple[Optional[str], str]] = ()) -> Record:
        """Greedy-decode one prompt and return its tokens with top-k logprobs.

        ``companions`` are unrelated (system, prompt) pairs sent at the same moment,
        so that a batching server processes the measured sequence together with
        them. Their responses are discarded.
        """
        threads: List[threading.Thread] = []
        for other_system, other in companions:
            th = threading.Thread(target=self._discard, args=(other, other_system), daemon=True)
            th.start()
            threads.append(th)
        data = _request(_url(self.base_url, "/chat/completions"),
                        self._payload(prompt, True, system), self.api_key, self.timeout)
        for th in threads:
            th.join()
        return parse_response(prompt_id, prompt, data)

    def warm(self, prompt: str, system: Optional[str] = None) -> None:
        """Send one throwaway request, so that the server has seen ``system`` before."""
        payload = self._payload(prompt, False, system)
        payload["max_tokens"] = 1
        _request(_url(self.base_url, "/chat/completions"), payload, self.api_key, self.timeout)

    def _discard(self, prompt: str, system: Optional[str]) -> None:
        try:
            _request(_url(self.base_url, "/chat/completions"),
                     self._payload(prompt, False, system), self.api_key, self.timeout)
        except EndpointError:
            pass  # a failed companion only weakens the co-batching; the measured request decides


def parse_response(prompt_id: str, prompt: str, data: dict) -> Record:
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError):
        raise EndpointError(f"unexpected response shape: {json.dumps(data)[:300]}") from None
    tokens = []
    for entry in ((choice.get("logprobs") or {}).get("content") or []):
        if not entry:
            continue
        top = [[alt.get("id"), alt.get("token"), alt.get("logprob")]
               for alt in (entry.get("top_logprobs") or [])]
        tokens.append({"id": entry.get("id"), "token": entry.get("token"),
                       "logprob": entry.get("logprob"), "top": top})
    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "id": prompt_id,
        "prompt": prompt,
        "text": (choice.get("message") or {}).get("content"),
        "finish": choice.get("finish_reason"),
        "tokens": tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
    }
