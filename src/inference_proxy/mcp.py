"""MCP server (stdio) that lets an agent such as Claude Code, Codex or Kilo hand small jobs to the proxy.

It talks to a running proxy over HTTP, so the account pools, cooldowns and prices stay in one place. A delegated call
never goes to a Claude or Codex CLI provider unless asked, so an agent cannot loop back into itself.

Environment: INFERENCE_PROXY_URL (default http://127.0.0.1:8480), INFERENCE_PROXY_API_KEY (if the proxy needs one).
"""

from __future__ import annotations

import json
import os
import sys

import httpx

from .cli import current_depth

PROTOCOL = "2025-06-18"
MAX_DEPTH = 2  # an agent may delegate, and a delegate may delegate once more; deeper is refused
TIERS = ["auto", "trivial", "moderate", "complex", "hardcore"]


def _tool(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {"name": name, "description": description,
            "inputSchema": {"type": "object", "properties": properties or {}, "required": required or []}}


TOOLS = [
    _tool(
        "delegate",
        "Hand one self-contained job to a cheaper model through the inference proxy and get its answer back as text. "
        "Good for summaries, reformatting, classification, boilerplate, drafting tests and bulk text edits. The prompt must "
        "carry all the context the model needs (it cannot see your files), and the answer is untrusted text to check before use. "
        "tier 'auto' lets a local judge pick the model by difficulty; 'trivial' keeps it on the cheapest tier.",
        {"prompt": {"type": "string"}, "system": {"type": "string"},
         "tier": {"type": "string", "enum": TIERS, "description": "default auto"},
         "model": {"type": "string", "description": "pin a provider or provider:model (overrides tier)"},
         "max_tokens": {"type": "integer"}, "temperature": {"type": "number"},
         "local_only": {"type": "boolean", "description": "only models on this machine or network (use for private content)"},
         "allow_cli": {"type": "boolean", "description": "allow Claude/Codex CLI providers (off by default)"}},
        ["prompt"]),
    _tool("delegate_status", "Which providers and accounts are healthy or cooling down, which endpoints are pinned, and which model judges difficulty."),
    _tool("delegate_ladder", "The suggested model ladder: models ordered by independent score and price, and which fill each tier.",
          {"rungs": {"type": "integer"}, "tolerance": {"type": "number"}}),
    _tool("delegate_usage", "What delegated requests cost, by provider, model and tier.", {"days": {"type": "number"}}),
]


class Proxy:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self.base = os.environ.get("INFERENCE_PROXY_URL", "http://127.0.0.1:8480").rstrip("/")
        key = os.environ.get("INFERENCE_PROXY_API_KEY")
        self.client = client or httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0))
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}

    def get(self, path: str, **params) -> dict:
        r = self.client.get(self.base + path, params=params, headers=self.headers)
        r.raise_for_status()
        return r.json()

    def delegate(self, a: dict) -> dict:
        messages = ([{"role": "system", "content": a["system"]}] if a.get("system") else []) + [{"role": "user", "content": a["prompt"]}]
        body: dict = {"model": a.get("model") or a.get("tier") or "auto", "messages": messages}
        for key in ("max_tokens", "temperature"):
            if a.get(key) is not None:
                body[key] = a[key]
        options: dict = {}
        if not a.get("allow_cli"):
            options["exclude_types"] = ["claude", "codex"]
        if a.get("local_only"):
            options["local_only"] = True
        body["inference_proxy"] = options
        r = self.client.post(self.base + "/v1/chat/completions", json=body, headers=self.headers)
        data = r.json()
        if r.status_code != 200:
            raise RuntimeError(str((data.get("error") or {}).get("message") or data)[:500])
        meta = data.get("inference_proxy") or {}
        return {"text": data["choices"][0]["message"].get("content") or "", "tier": meta.get("tier"), "provider": meta.get("provider"),
                "model": meta.get("model"), "difficulty": meta.get("difficulty"), "usage": data.get("usage"), "cost_usd": meta.get("cost_usd")}


def call(proxy: Proxy, name: str, a: dict):
    if name == "delegate":
        depth = current_depth()
        if depth >= MAX_DEPTH:
            raise RuntimeError(f"delegation depth limit reached ({depth}): answer this yourself instead of delegating again")
        return proxy.delegate(a)
    if name == "delegate_status":
        pools = proxy.get("/admin/pools")
        models = proxy.get("/admin/api/models")
        return {"pools": pools, "pinned_endpoints": models.get("pins"), "judge": models.get("judge")}
    if name == "delegate_ladder":
        d = proxy.get("/admin/api/ladder", rungs=a.get("rungs", 10), tolerance=a.get("tolerance", 0.03))
        return {"rungs": [{k: r.get(k) for k in ("rung", "difficulty_from", "difficulty_to", "provider", "model", "score")} for r in d["rungs"]],
                "tiers": d["tiers"], "unscored": len(d["unscored"])}
    if name == "delegate_usage":
        return proxy.get("/admin/api/usage", days=a.get("days", 7))
    raise ValueError(f"unknown tool {name}")


def handle(msg: dict, proxy: Proxy) -> dict | None:
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:
        return None
    if method == "initialize":
        want = (msg.get("params") or {}).get("protocolVersion", PROTOCOL)
        return {"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": want, "capabilities": {"tools": {}},
                "serverInfo": {"name": "inference-proxy", "version": "0.1.0"}}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        p = msg.get("params") or {}
        try:
            out = call(proxy, p.get("name"), p.get("arguments") or {})
            return {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False, default=str)}]}}
        except Exception as exc:  # report to the agent instead of dying
            return {"jsonrpc": "2.0", "id": mid, "result": {"isError": True, "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"[:600]}]}}
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown method {method}"}}


def run() -> None:
    proxy = Proxy()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            resp = handle(json.loads(line), proxy)
        except Exception as exc:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run()
