#!/usr/bin/env python3
"""ninfer-ui-proxy — the llama.cpp web UI front end for a ninfer-serve backend.

The llama.cpp dist talks to llama-server's HTTP surface: `/props`, `/slots`,
`/v1/models`, `/v1/chat/completions` (SSE) and a few control paths.  ninfer-serve
speaks the OpenAI/Anthropic API only, so this process serves the dist statically
and adapts the llama.cpp-shaped discovery/control paths, proxying generation to
the engine on an internal port.

Layout:  LAN :8081 -> this proxy -> ninfer-serve on 127.0.0.1:18091
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import aiohttp
import mimetypes
import os
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web

LLAMA_PARAM_DEFAULTS = {
    "id": 0, "scale": 1.0, "seed": -1, "temperature": 0.8, "dynatemp_range": 0.0,
    "dynatemp_exponent": 1.0, "top_k": 40, "top_p": 0.95, "min_p": 0.05,
    "top_n_sigma": -1.0, "xtc_probability": 0.0, "xtc_threshold": 0.1, "typical_p": 1.0,
    "repeat_last_n": 64, "repeat_penalty": 1.0, "presence_penalty": 0.0,
    "frequency_penalty": 0.0, "dry_multiplier": 0.0, "dry_base": 1.75,
    "dry_allowed_length": 2, "dry_penalty_last_n": -1, "mirostat": 0, "mirostat_tau": 5.0,
    "mirostat_eta": 0.1, "adaptive_target": -1.0, "adaptive_decay": 0.9,
    "max_tokens": -1, "n_predict": -1, "n_keep": 0, "samplers": [],
}

HOP_HEADERS = {
    "host", "content-length", "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade",
    "accept-encoding",
}


class Proxy:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.upstream = args.upstream.rstrip("/")
        self.dist = Path(args.dist).resolve()
        self.template = ""
        if args.template_file:
            try:
                self.template = Path(args.template_file).read_text(encoding="utf-8")
            except OSError as error:
                print(f"warning: cannot read template {args.template_file}: {error}", flush=True)
        self.session: ClientSession | None = None
        self._session_lock = asyncio.Lock()

    async def session_or_start(self) -> ClientSession:
        # One session for all in-flight requests: concurrent first requests must not each
        # create (and leak) their own connector.
        async with self._session_lock:
            if self.session is None or self.session.closed:
                self.session = ClientSession(
                    timeout=ClientTimeout(total=None, sock_connect=10, sock_read=3600),
                    auto_decompress=False,
                )
            return self.session

    # ---- llama.cpp-shaped discovery ------------------------------------------------
    def model_entry(self) -> dict:
        return {
            "id": self.args.model,
            "object": "model",
            "created": 0,
            "owned_by": "local",
            "status": {"value": "loaded"},
            "meta": {"n_ctx": self.args.ctx, "n_ctx_train": self.args.ctx},
            "architecture": {
                "input_modalities": ["text", "image"] if self.args.vision else ["text"],
                "output_modalities": ["text"],
            },
            "capabilities": ["completion"] + (["multimodal"] if self.args.vision else []),
        }

    def props(self) -> dict:
        params = dict(LLAMA_PARAM_DEFAULTS)
        params["n_predict"] = -1
        params["max_tokens"] = -1
        caps = {"tools": True, "tool_choice": True, "parallel_tool_calls": True,
                "thinking": True, "vision": bool(self.args.vision)}
        return {
            "default_generation_settings": {"n_ctx": self.args.ctx, "params": params},
            "total_slots": self.args.slots,
            "model_alias": self.args.model,
            "model_path": self.args.artifact,
            "chat_template": self.template,
            "chat_template_caps": caps,
            "modalities": {"vision": bool(self.args.vision), "audio": False},
            "models_autoload": False,
            "is_sleeping": False,
            "n_ctx": self.args.ctx,
            "n_ctx_train": self.args.ctx,
            "build_info": "ninfer",
        }

    # ---- static dist ---------------------------------------------------------------
    def static_response(self, path: str) -> web.StreamResponse:
        relative = path.lstrip("/") or "index.html"
        candidate = (self.dist / relative).resolve()
        try:
            candidate.relative_to(self.dist)
        except ValueError:
            raise web.HTTPNotFound()
        if not candidate.is_file():
            # SPA routes have no file of their own; assets keep their 404.
            if "." in Path(relative).name:
                raise web.HTTPNotFound()
            candidate = self.dist / "index.html"
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        headers = {"Cache-Control": "no-cache"}
        return web.FileResponse(candidate, headers=headers)

    # ---- upstream proxy ------------------------------------------------------------
    async def forward(self, request: web.Request) -> web.StreamResponse:
        session = await self.session_or_start()
        url = self.upstream + request.path_qs
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
        body = await request.read() if request.can_read_body else None
        # The llama.cpp web UI posts generation requests without `model` (llama-server never needs
        # it); ninfer-serve requires it, so fill in the served alias when the client omits it.
        if body and request.path in ("/v1/chat/completions", "/v1/responses", "/v1/messages"):
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                payload = None
            if isinstance(payload, dict) and not payload.get("model"):
                payload["model"] = self.args.model
                body = json.dumps(payload).encode("utf-8")
                headers["Content-Type"] = "application/json"
            # Temporary diagnosis aid: keep the last generation request (media payloads elided).
            if isinstance(payload, dict) and os.environ.get("NINFER_UI_DUMP", "1") == "1":
                try:
                    masked = json.loads(body)
                    for message in masked.get("messages", []):
                        content = message.get("content")
                        if isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and isinstance(part.get("image_url"), dict):
                                    image_url = part["image_url"].get("url", "")
                                    if isinstance(image_url, str) and image_url.startswith("data:"):
                                        try:
                                            header, payload = image_url.split(",", 1)
                                            Path("/tmp/ninfer-ui-last-image.meta").write_text(header)
                                            Path("/tmp/ninfer-ui-last-image.bin").write_bytes(
                                                base64.b64decode(payload))
                                        except Exception:
                                            pass
                                        part["image_url"]["url"] = f"<data {len(image_url)} chars>"
                    Path("/tmp/ninfer-ui-last-request.json").write_text(
                        json.dumps(masked, ensure_ascii=False, indent=1))
                except Exception:
                    pass
        upstream = await session.request(request.method, url, headers=headers, data=body)
        try:
            content_type = upstream.headers.get("Content-Type", "")
            if "text/event-stream" in content_type:
                response = web.StreamResponse(
                    status=upstream.status,
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )
                await response.prepare(request)
                try:
                    async for chunk in upstream.content.iter_any():
                        await response.write(chunk)
                    await response.write_eof()
                except (aiohttp.ClientConnectionError, ConnectionResetError):
                    # The browser closed the tab or cancelled: nothing left to write.
                    pass
                return response
            payload = await upstream.read()
            passthrough = {
                k: v for k, v in upstream.headers.items()
                if k.lower() not in HOP_HEADERS and k.lower() != "content-type"
            }
            return web.Response(status=upstream.status, body=payload,
                                content_type=content_type.split(";")[0] or None,
                                headers=passthrough)
        finally:
            upstream.close()

    # ---- router --------------------------------------------------------------------
    async def handle(self, request: web.Request) -> web.StreamResponse:
        path = request.path
        method = request.method
        if path == "/health":
            return web.json_response({"status": "ok"})
        if path == "/props":
            return web.json_response(self.props())
        if path in ("/v1/models", "/models") and method == "GET":
            entry = self.model_entry()
            return web.json_response({"object": "list", "data": [entry], "models": [entry]})
        if path.startswith("/v1/models/") and method == "GET":
            return web.json_response(self.model_entry())
        if path == "/slots" and method == "GET":
            busy = False
            slots = [{"id": i, "n_ctx": self.args.ctx, "is_processing": busy,
                      "n_prompt_tokens": 0} for i in range(self.args.slots)]
            return web.json_response(slots)
        if path.startswith("/slots") and method == "POST":
            return web.json_response({"success": True})
        if path in ("/models/load", "/models/unload") and method == "POST":
            return web.json_response({"status": "ok"})
        if path == "/models/sse" and method == "GET":
            response = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await response.prepare(request)
            await response.write(b"data: {}\n\n")
            return response
        if path == "/v1/chat/completions/control" and method == "POST":
            return web.json_response({"success": True})
        if path == "/tools" and method == "GET":
            return web.json_response({"tools": []})
        if path.startswith("/v1/") and path not in (
                "/v1/chat/completions", "/v1/responses", "/v1/responses/input_tokens",
                "/v1/messages", "/v1/messages/count_tokens"):
            return web.json_response({"error": {"message": "not found"}}, status=404)
        if path.startswith(("/v1/", "/api", "/props", "/slots", "/health", "/metrics",
                            "/status", "/tokenize", "/detokenize")):
            return await self.forward(request)
        if method in ("GET", "HEAD"):
            return self.static_response(path)
        raise web.HTTPNotFound()


def build_app(proxy: Proxy) -> web.Application:
    app = web.Application(client_max_size=1024 ** 3)
    app.router.add_route("*", "/{path:.*}", proxy.handle)
    return app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--upstream", default="http://127.0.0.1:18091")
    parser.add_argument("--dist", default="/tank/Strata/web-llama")
    parser.add_argument("--model", default="Qwen3.8-27B")
    parser.add_argument("--ctx", type=int, default=524288)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--vision", action="store_true", default=True)
    parser.add_argument("--artifact", default="")
    parser.add_argument("--template-file", default="/home/ai-agent/chat_template_qwen38-27b.jinja")
    args = parser.parse_args()
    proxy = Proxy(args)
    web.run_app(build_app(proxy), host=args.host, port=args.port,
                print=lambda *a, **k: print(*a, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
