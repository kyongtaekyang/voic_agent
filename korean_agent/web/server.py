"""한국어 Agent 전용 화면 서버 (포트 7201, 프롬프트 2.3절).

- GET  /               사용자 화면 (주소 끝에 ?dev=1 을 붙이면 개발자 화면도 표시, services.yaml web.dev_screen 이 true 일 때만)
- GET  /api/status     운영 모드, 점검 여부, 모델 정보, 영어 Agent 주소
- GET  /api/dev/gpu    GPU 메모리 (개발자 화면용)
- WS   /ws             화면용 WebSocket. Realtime 서버(ws://127.0.0.1:8766/v1/realtime)로 중계하면서
                       kva.* 이벤트(언어 안내, 지연, 출력 검사, 점검)를 끼워 넣는다. 형식은 docs/event_schema.md.

실행: .venv/bin/python -m web.server   (포트·주소는 config/services.yaml)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from websockets.asyncio.client import connect as ws_connect

from web.recorder import SessionRecorder
from web.relay import EventTail, PlaybackTracker, TurnMetrics, audio_ms_of_append, filter_client_event, read_mode, to_screen_events

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
logger = logging.getLogger("kva.web")


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "config" / "services.yaml").read_text(encoding="utf-8"))


CFG = load_cfg()
WEB = CFG.get("web", {})
UPSTREAM = os.environ.get("KVA_REALTIME_URL", f"ws://127.0.0.1:{CFG['ports']['realtime_server']}/v1/realtime")
EVENT_LOG = Path(os.environ.get("KVA_EVENT_LOG", ROOT / "logs" / "events.jsonl"))
IGNORE_MODE = os.environ.get("KVA_WEB_IGNORE_MODE") == "1"
SLOW = CFG.get("slow_path", {})  # record 가 true 일 때만 queue/korean/ 에 세션 녹음 (web/recorder.py)  # 개발 시험용: 운영 모드와 무관하게 대화 허용

app = FastAPI(title="한국어 Agent 화면", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def current_status() -> dict:
    mode = read_mode(WEB.get("ops_dir", ROOT.parent / "ada-ops"))
    maintenance = bool(WEB.get("require_realtime_mode", True)) and mode != "realtime" and not IGNORE_MODE
    llm, tts, stt = CFG["llm"], CFG["tts"], CFG["stt"]
    return {
        "mode": mode,
        "maintenance": maintenance,
        "dev_screen": bool(WEB.get("dev_screen", False)),
        "english_agent_url": WEB.get("english_agent_url"),
        "models": {
            "llm": llm["model"], "llm_precision": llm.get("quantization"),
            "stt": Path(stt["model"]).name, "stt_device": stt["device"],
            "tts": tts["model"], "tts_backend": f"{tts.get('qwen3_backend', '')} / {tts['device']}",
        },
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/status")
def api_status() -> JSONResponse:
    return JSONResponse(current_status())


@app.get("/api/dev/gpu")
def api_gpu() -> JSONResponse:
    if not WEB.get("dev_screen"):
        return JSONResponse({"error": "개발자 화면이 꺼져 있습니다."}, status_code=403)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            timeout=5).decode().strip().splitlines()[0]
        name, used, total = [x.strip() for x in out.split(",")]
        return JSONResponse({"name": name, "used_mib": int(used), "total_mib": int(total)})
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"{type(e).__name__}"}, status_code=500)


async def _send(ws: WebSocket, obj: dict) -> None:
    await ws.send_text(json.dumps(obj, ensure_ascii=False))


@app.websocket("/ws")
async def relay(ws: WebSocket) -> None:
    await ws.accept()
    status = current_status()
    await _send(ws, {"type": "kva.status", **status})
    if status["maintenance"]:
        await _send(ws, {"type": "kva.maintenance", "mode": status["mode"]})
        await ws.close()
        return
    try:
        upstream = await ws_connect(UPSTREAM, max_size=None, open_timeout=5)
    except Exception as e:  # noqa: BLE001
        logger.warning("upstream connect failed: %s", e)
        await _send(ws, {"type": "kva.unavailable", "reason": "connect_failed"})
        await ws.close()
        return

    metrics = TurnMetrics(speech_pad_ms=float(CFG.get("turn", {}).get("speech_pad_ms", 0)))
    playback = PlaybackTracker()
    tail = EventTail(EVENT_LOG, from_end=True)
    session_started = asyncio.Event()
    rec = SessionRecorder(ROOT / SLOW.get("queue_dir", "queue/korean")) if SLOW.get("record") else None

    async def browser_to_upstream() -> None:
        while True:
            raw = await ws.receive_text()
            try:
                msg = filter_client_event(json.loads(raw))
            except ValueError:
                msg = None
            if msg is None:
                continue
            if rec:
                rec.client_event(msg)
            if msg["type"] == "input_audio_buffer.append":
                metrics.add_audio(audio_ms_of_append(msg))
            elif msg["type"] == "conversation.item.truncate":
                # 들려준 문장만 이력에 남긴다 (speech-to-speech 1.1.0 은 truncate 로 이력을 고치지 않음)
                rewrites = playback.rewrite(msg["item_id"], msg["audio_end_ms"])
                for ev in rewrites:
                    await upstream.send(json.dumps(ev))
                if rewrites:
                    await _send(ws, {"type": "kva.history_truncated", "item_id": msg["item_id"],
                                     "audio_end_ms": msg["audio_end_ms"], "kept": len(rewrites) > 1})
                continue
            await upstream.send(json.dumps(msg))

    async def upstream_to_browser() -> None:
        async for raw in upstream:
            text = raw if isinstance(raw, str) else raw.decode("utf-8", "replace")
            await ws.send_text(text)
            try:
                event = json.loads(text)
            except ValueError:
                continue
            if rec:
                rec.server_event(event)
            if event.get("type") == "session.created":
                session_started.set()
            playback.feed(event)
            for extra in metrics.feed(event):
                if rec:
                    rec.server_event(extra)
                await _send(ws, extra)

    async def side_events() -> None:
        last_mode_check = 0.0
        while True:
            for row in tail.read():
                for extra in to_screen_events(row):
                    if rec:
                        rec.server_event(extra)
                    await _send(ws, extra)
            if time.monotonic() - last_mode_check > 5:
                last_mode_check = time.monotonic()
                if current_status()["maintenance"]:
                    await _send(ws, {"type": "kva.maintenance", "mode": read_mode(WEB.get("ops_dir", ""))})
                    return
            await asyncio.sleep(0.2)

    tasks = [asyncio.create_task(t()) for t in (browser_to_upstream, upstream_to_browser, side_events)]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            exc = t.exception()
            if exc and not isinstance(exc, WebSocketDisconnect):
                logger.info("relay ended: %s: %s", type(exc).__name__, exc)
        if not session_started.is_set():
            # speech-to-speech 는 동시에 한 세션만 받는다(num_pipelines=1). 세션이 시작되지 않고 끊기면 사용 중으로 본다.
            try:
                await _send(ws, {"type": "kva.unavailable", "reason": "busy"})
            except Exception:  # noqa: BLE001
                pass
    finally:
        if rec:
            rec.close()  # 취소돼도 녹음은 닫히도록 await 보다 먼저
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await upstream.close()
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=WEB.get("host", "127.0.0.1"), port=int(CFG["ports"]["web"]), log_level="info")


if __name__ == "__main__":
    main()
