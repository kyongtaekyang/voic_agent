"""영어 Agent 화면 서버: 127.0.0.1:7101 (프롬프트 2.6절).

정적 화면(web/static)을 내보내고, 브라우저 WebSocket /ws 를 Core(:7102 /session)로 그대로 이어 준다.
/api/status, /api/dev/gpu 는 Core 에 물어서 넘긴다. 개발자 화면은 설정 web.developer_panel 이 true 일 때만 켤 수 있다.
접속은 SSH 터널(localhost)로 한다: 브라우저 마이크 권한은 HTTPS 또는 localhost 에서만 허용된다.
실행: .venv/bin/python -m web.server [--config config/services.yaml]
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import aiohttp
import yaml
from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
HTTP_KEY = web.AppKey("http", aiohttp.ClientSession)


def make_app(cfg: dict) -> web.Application:
    core = f"http://127.0.0.1:{int(cfg['web']['core_port'])}"
    core_ws = core.replace("http://", "ws://") + "/session"
    dev_allowed = bool(cfg["web"].get("developer_panel", False))
    app = web.Application()

    async def on_startup(app):
        app[HTTP_KEY] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=5))

    async def on_cleanup(app):
        await app[HTTP_KEY].close()

    async def index(request):
        return web.FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    async def proxy_json(request, path):
        try:
            async with app[HTTP_KEY].get(core + path, timeout=aiohttp.ClientTimeout(total=8)) as r:
                data = await r.json()
        except Exception:  # noqa: BLE001
            data = {"type": "eva.status", "maintenance": True, "core": "unreachable"}
        if path == "/api/status":
            data["developer_panel"] = dev_allowed
            data["korean_agent_url"] = cfg["web"].get("korean_agent_url")
        return web.json_response(data)

    async def status(request):
        return await proxy_json(request, "/api/status")

    async def gpu(request):
        if not dev_allowed:
            raise web.HTTPForbidden()
        return await proxy_json(request, "/api/dev/gpu")

    async def ws(request):
        client = web.WebSocketResponse(max_msg_size=1 << 20)
        await client.prepare(request)
        try:
            upstream = await app[HTTP_KEY].ws_connect(core_ws, max_msg_size=0)
        except Exception:  # noqa: BLE001
            await client.send_json({"type": "eva.unavailable", "reason": "core_unreachable"})
            await client.close()
            return client

        async def pipe(src, dst):
            async for m in src:
                if dst.closed:
                    break
                if m.type == aiohttp.WSMsgType.BINARY:
                    await dst.send_bytes(m.data)
                elif m.type == aiohttp.WSMsgType.TEXT:
                    await dst.send_str(m.data)
                else:
                    break

        tasks = [asyncio.create_task(pipe(client, upstream)), asyncio.create_task(pipe(upstream, client))]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        # Core 가 세션 요약을 보낼 시간을 조금 준다
        await asyncio.wait(tasks, timeout=2)
        for t in tasks:
            t.cancel()
        for s in (upstream, client):
            if not s.closed:
                await s.close()
        return client

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws)
    app.router.add_get("/api/status", status)
    app.router.add_get("/api/dev/gpu", gpu)
    app.router.add_static("/static", STATIC)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.environ.get("EVA_CONFIG", str(ROOT / "config/services.yaml")))
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    web.run_app(make_app(cfg), host="127.0.0.1", port=int(cfg["web"]["screen_port"]), print=None)


if __name__ == "__main__":
    main()
