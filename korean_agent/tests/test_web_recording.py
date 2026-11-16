"""화면 서버 중계 경로에 세션 녹음이 붙어 있는지: 가짜 Realtime 서버 ↔ web.server ↔ TestClient."""

import asyncio
import base64
import json
import threading

from fastapi.testclient import TestClient
from websockets.asyncio.server import serve

import web.server as ws_mod

PCM = base64.b64encode(b"\x01\x00" * 1600).decode()
SCRIPT = [
    {"type": "input_audio_buffer.speech_started", "audio_start_ms": 0, "item_id": "u1"},
    {"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 1000, "item_id": "u1"},
    {"type": "conversation.item.input_audio_transcription.completed", "item_id": "u1", "transcript": "안녕하세요"},
    {"type": "response.created"},
    {"type": "response.output_audio_transcript.delta", "delta": "반가워요."},
    {"type": "response.output_audio.delta", "delta": PCM},
    {"type": "response.done"},
]


def _start_upstream():
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    box = {}

    async def handler(conn):
        await conn.send(json.dumps({"type": "session.created"}))
        async for raw in conn:
            if json.loads(raw).get("type") == "input_audio_buffer.append":
                for ev in SCRIPT:
                    await conn.send(json.dumps(ev))

    async def main():
        async with serve(handler, "127.0.0.1", 0) as server:
            box["port"] = server.sockets[0].getsockname()[1]
            box["stop"] = loop.create_future()
            ready.set()
            await box["stop"]

    th = threading.Thread(target=lambda: loop.run_until_complete(main()), daemon=True)
    th.start()
    ready.wait(5)
    return box, loop, th


def test_relay_records_session(tmp_path, monkeypatch):
    box, loop, th = _start_upstream()
    try:
        monkeypatch.setattr(ws_mod, "UPSTREAM", f"ws://127.0.0.1:{box['port']}")
        monkeypatch.setattr(ws_mod, "IGNORE_MODE", True)
        monkeypatch.setattr(ws_mod, "EVENT_LOG", tmp_path / "events_missing.jsonl")
        monkeypatch.setattr(ws_mod, "SLOW", {"record": True, "queue_dir": str(tmp_path / "queue")})
        with TestClient(ws_mod.app) as client, client.websocket_connect("/ws") as ws:
            seen = []
            while "session.created" not in seen:
                seen.append(json.loads(ws.receive_text())["type"])
            ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": PCM}))
            while "response.done" not in seen:
                seen.append(json.loads(ws.receive_text())["type"])
    finally:
        loop.call_soon_threadsafe(box["stop"].set_result, None)
        th.join(5)
    sessions = list((tmp_path / "queue").iterdir())
    assert len(sessions) == 1
    d = sessions[0]
    meta = json.loads((d / "meta.json").read_text())
    assert meta["status"] == "queued" and meta["user_s"] == 0.1 and meta["agent_audio_s"] == 0.1
    types = [json.loads(x)["type"] for x in (d / "events.jsonl").read_text().splitlines()]
    assert "conversation.item.input_audio_transcription.completed" in types and "response.done" in types


def test_recording_off_by_default(tmp_path, monkeypatch):
    assert ws_mod.load_cfg()["slow_path"]["record"] is False
