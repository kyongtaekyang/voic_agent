"""Gateway 통합 시험: 가짜 Moshi(bench/mock_moshi.py) ↔ Core(gateway/server.py) ↔ 시험 클라이언트. GPU 를 쓰지 않는다."""

import asyncio
import json
import sys
from pathlib import Path

import aiohttp
import numpy as np
from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import mock_moshi  # noqa: E402

import gateway.server as gs  # noqa: E402

SCRIPT = [
    ["Hello", " there", "!"],
    [" You", " are", " so", " stupid"],  # 마지막 단어 뒤에 경계 토큰 없이 침묵 → idle() 로 확정돼야 함
    [" Great", " job", "."],
]


async def _serve(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return runner, port


def _cfg(moshi_port, tmp_path, record=True):
    return {
        "moshi": {"url": f"ws://127.0.0.1:{moshi_port}/api/chat"},
        "safety": {"rules": "config/safety_rules.yaml", "output_delay_ms": 0, "idle_ms": 40, "unmute_quiet_ms": 100},
        "asr": {"enabled": False},
        "gateway": {"require_mode": False, "record": record, "handshake_timeout_s": 2},
        "web": {"screen_port": 0, "core_port": 0, "korean_agent_url": "http://127.0.0.1:7201/"},
        "queue_dir": str(tmp_path / "queue"),
    }


async def _talk(core_port, seconds=3.0, frame_ms=20):
    """20ms 마다 80ms 분량(1920 샘플)을 보낸다 (실시간의 4배 속도). 받은 이벤트와 음성 바이트 수를 돌려준다."""
    events, audio_bytes = [], 0
    pcm = (0.05 * np.random.default_rng(0).standard_normal(1920)).clip(-1, 1)
    frame = (pcm * 32767).astype("<i2").tobytes()
    async with aiohttp.ClientSession() as http:
        async with http.ws_connect(f"http://127.0.0.1:{core_port}/session") as ws:
            async def reader():
                nonlocal audio_bytes
                async for m in ws:
                    if m.type == aiohttp.WSMsgType.TEXT:
                        events.append(json.loads(m.data))
                    elif m.type == aiohttp.WSMsgType.BINARY:
                        audio_bytes += len(m.data)
            rt = asyncio.create_task(reader())
            for _ in range(int(seconds * 1000 / frame_ms)):
                if ws.closed:
                    break
                await ws.send_bytes(frame)
                await asyncio.sleep(frame_ms / 1000)
            await asyncio.sleep(0.3)
            await ws.close()
            await asyncio.wait_for(rt, 5)
    return events, audio_bytes


def test_session_text_audio_and_safety_block(tmp_path):
    async def main():
        m_runner, m_port = await _serve(mock_moshi.make_app(SCRIPT, gap=8))  # 줄 사이 무음 8프레임 ≈ 160ms(4배속)
        c_runner, c_port = await _serve(gs.make_app(_cfg(m_port, tmp_path)))
        try:
            return await _talk(c_port)
        finally:
            await c_runner.cleanup()
            await m_runner.cleanup()

    events, audio_bytes = asyncio.run(main())
    types = [e["type"] for e in events]
    assert "eva.session.started" in types
    text = "".join(e["delta"] for e in events if e["type"] == "eva.agent_text")
    assert "Hello there" in text
    assert "stupid" not in text  # 위반 토큰은 자막으로 내보내지 않는다
    blocks = [e for e in events if e["type"] == "eva.safety_block"]
    assert blocks and blocks[0]["category"] == "insult"
    assert "eva.safety_resumed" in types  # 침묵 뒤 출력 재개
    assert "Great job" in text  # 재개 뒤 정상 자막
    assert audio_bytes > 0
    sess = list((tmp_path / "queue").iterdir())
    assert len(sess) == 1
    names = {p.name for p in sess[0].iterdir()}
    assert {"user.wav", "agent.wav", "events.jsonl", "meta.json"} <= names
    logged = [json.loads(line)["type"] for line in (sess[0] / "events.jsonl").read_text().splitlines()]
    assert "eva.safety_block" in logged and logged[-1] == "eva.session.ended"
    meta = json.loads((sess[0] / "meta.json").read_text())
    assert meta["user_s"] > 1.0 and meta["status"] == "queued"


def test_output_delay_buffer_drops_unplayed_audio(tmp_path):
    """출력 지연 버퍼를 두면 위반 판정 시 아직 내보내지 않은 음성을 버린다 (dropped_ms > 0)."""
    async def main():
        m_runner, m_port = await _serve(mock_moshi.make_app([[" You", " are", " stupid", "."]], gap=3))
        cfg = _cfg(m_port, tmp_path, record=False)
        cfg["safety"]["output_delay_ms"] = 300
        c_runner, c_port = await _serve(gs.make_app(cfg))
        try:
            return await _talk(c_port, seconds=1.5)
        finally:
            await c_runner.cleanup()
            await m_runner.cleanup()

    events, _ = asyncio.run(main())
    blocks = [e for e in events if e["type"] == "eva.safety_block"]
    assert blocks and blocks[0]["dropped_ms"] > 0


def test_busy_and_maintenance(tmp_path, monkeypatch):
    async def main():
        m_runner, m_port = await _serve(mock_moshi.make_app(SCRIPT))
        cfg = _cfg(m_port, tmp_path, record=False)
        c_runner, c_port = await _serve(gs.make_app(cfg))
        out = {}
        try:
            async with aiohttp.ClientSession() as http:
                async with http.ws_connect(f"http://127.0.0.1:{c_port}/session") as ws1:
                    m = await ws1.receive(timeout=5)
                    assert json.loads(m.data)["type"] == "eva.session.started"
                    async with http.ws_connect(f"http://127.0.0.1:{c_port}/session") as ws2:
                        out["busy"] = json.loads((await ws2.receive(timeout=5)).data)
                await asyncio.sleep(0.2)
        finally:
            await c_runner.cleanup()
        cfg["gateway"]["require_mode"] = True
        monkeypatch.setattr(gs, "current_mode", lambda: "batch")
        c_runner, c_port = await _serve(gs.make_app(cfg))
        try:
            async with aiohttp.ClientSession() as http:
                async with http.ws_connect(f"http://127.0.0.1:{c_port}/session") as ws:
                    out["maint"] = json.loads((await ws.receive(timeout=5)).data)
                async with http.get(f"http://127.0.0.1:{c_port}/api/status") as r:
                    out["status"] = await r.json()
        finally:
            await c_runner.cleanup()
            await m_runner.cleanup()
        return out

    out = asyncio.run(main())
    assert out["busy"] == {"type": "eva.unavailable", "reason": "busy"}
    assert out["maint"]["reason"] == "maintenance"
    assert out["status"]["maintenance"] is True and out["status"]["mode"] == "batch"


def test_screen_server_relays(tmp_path):
    """화면 서버(:7101 역할)가 /ws 를 Core 로 잇고, /api/status 에 화면용 필드를 붙인다."""
    import web.server as ws_mod

    async def main():
        m_runner, m_port = await _serve(mock_moshi.make_app(SCRIPT))
        c_runner, c_port = await _serve(gs.make_app(_cfg(m_port, tmp_path, record=False)))
        cfg = _cfg(m_port, tmp_path, record=False)
        cfg["web"]["core_port"] = c_port
        w_runner, w_port = await _serve(ws_mod.make_app(cfg))
        out = {}
        try:
            async with aiohttp.ClientSession() as http:
                async with http.get(f"http://127.0.0.1:{w_port}/api/status") as r:
                    out["status"] = await r.json()
                async with http.get(f"http://127.0.0.1:{w_port}/") as r:
                    out["index"] = r.status
                async with http.ws_connect(f"http://127.0.0.1:{w_port}/ws") as ws:
                    out["first"] = json.loads((await ws.receive(timeout=5)).data)
        finally:
            await w_runner.cleanup()
            await c_runner.cleanup()
            await m_runner.cleanup()
        return out

    out = asyncio.run(main())
    assert out["index"] == 200
    assert out["status"]["korean_agent_url"] == "http://127.0.0.1:7201/"
    assert out["status"]["developer_panel"] is False
    assert out["first"]["type"] == "eva.session.started"


def test_moshi_unreachable(tmp_path):
    async def main():
        c_runner, c_port = await _serve(gs.make_app(_cfg(1, tmp_path, record=False)))
        try:
            async with aiohttp.ClientSession() as http:
                async with http.ws_connect(f"http://127.0.0.1:{c_port}/session") as ws:
                    return json.loads((await ws.receive(timeout=5)).data)
        finally:
            await c_runner.cleanup()

    ev = asyncio.run(main())
    assert ev["type"] == "eva.unavailable" and ev["reason"] == "moshi_unreachable"
