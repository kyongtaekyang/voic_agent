"""영어 Agent Core (Gateway) 서버: 127.0.0.1:7102.

- WebSocket /session : 대화 세션 하나 (gateway/session.py). Moshi 서버가 한 번에 한 연결만 받으므로 Core 도 한 세션만 받는다.
- GET /api/status   : 운영 모드, 점검 중 여부, 사용 중 여부, 설정 요약 (화면 서버가 그대로 넘긴다)
- GET /api/dev/gpu  : GPU 메모리 (개발자 화면용)
화면 서버(web/server.py, :7101)만 이 서버에 붙는다. 외부에 열지 않는다.
실행: .venv/bin/python -m gateway.server [--config config/services.yaml]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import wave
from pathlib import Path

import aiohttp
import numpy as np
import yaml
from aiohttp import web

from core.asr_monitor import LanguageMismatchDetector
from core.safety_monitor import SafetyMonitor, SafetyRules
from gateway.asr_worker import AsrWorker, NemotronRecognizer, resample_linear
from gateway.recorder import Recorder
from gateway.session import RATE, Session, SessionConfig

ROOT = Path(__file__).resolve().parent.parent
OPS_MODE = ROOT.parent / "ada-ops" / "run" / "mode"
ALLOWED_MODES = ("realtime", "en-solo")
log = logging.getLogger("eva.core")


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def current_mode() -> str | None:
    try:
        return OPS_MODE.read_text().strip() or None
    except OSError:
        return None


def load_wav_24k(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: PCM16 wav 만 지원")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
        return resample_linear(x, w.getframerate(), RATE)


def gpu_info() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, check=True).stdout.strip().splitlines()[0]
        name, used, total = [s.strip() for s in out.split(",")]
        return {"name": name, "used_mib": int(used), "total_mib": int(total)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"nvidia-smi 실패: {e}"[:120]}


class Core:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        g = cfg.get("gateway", {})
        s = cfg.get("safety", {})
        a = cfg.get("asr", {})
        self.require_mode = g.get("require_mode", True)
        self.rules = SafetyRules.from_yaml(ROOT / s.get("rules", "config/safety_rules.yaml"))
        self.redirect_pcm = load_wav_24k(ROOT / (self.rules.redirect_audio or "assets/redirect_en.wav"))
        self.scfg = SessionConfig(moshi_url=cfg["moshi"]["url"], output_delay_ms=int(s.get("output_delay_ms", 0)),
                                  idle_ms=int(s.get("idle_ms", 160)), unmute_quiet_ms=int(s.get("unmute_quiet_ms", 1000)),
                                  handshake_timeout_s=float(g.get("handshake_timeout_s", 60)))
        self.queue_dir = ROOT / cfg.get("queue_dir", "queue/english") if g.get("record", True) else None
        self.asr_cfg = a
        self.recognizer = None
        if a.get("enabled", True):
            self.recognizer = NemotronRecognizer(a["model"], a.get("language", "auto"), int(a.get("threads", 8)))
        self.korean_url = cfg.get("web", {}).get("korean_agent_url")
        self.busy = asyncio.Lock()
        self.http: aiohttp.ClientSession | None = None
        self.last_session: dict | None = None

    def maintenance(self) -> bool:
        return self.require_mode and current_mode() not in ALLOWED_MODES

    def status(self) -> dict:
        return {"type": "eva.status", "mode": current_mode(), "maintenance": self.maintenance(), "busy": self.busy.locked(),
                "moshi_url": self.scfg.moshi_url, "output_delay_ms": self.scfg.output_delay_ms,
                "asr": bool(self.recognizer), "redirect_audio": self.redirect_pcm is not None,
                "last_session": self.last_session}

    async def on_startup(self, app):
        self.http = aiohttp.ClientSession()
        if self.recognizer and self.asr_cfg.get("preload", True):
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(None, self.recognizer.load)
            fut.add_done_callback(lambda f: log.info("ASR 모델 준비: %s", "실패" if f.exception() else "완료"))

    async def on_cleanup(self, app):
        if self.http:
            await self.http.close()

    async def ws_session(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=1 << 20)
        await ws.prepare(request)
        if self.maintenance():
            await ws.send_json({"type": "eva.unavailable", "reason": "maintenance", "mode": current_mode()})
            await ws.close()
            return ws
        if self.busy.locked():
            await ws.send_json({"type": "eva.unavailable", "reason": "busy"})
            await ws.close()
            return ws
        async with self.busy:
            rec = Recorder(self.queue_dir) if self.queue_dir else None
            asr = None
            if self.recognizer:
                det = LanguageMismatchDetector(int(self.asr_cfg.get("mismatch_turns", 2)),
                                               float(self.asr_cfg.get("hangul_ratio_min", 0.3)))
                asr = AsrWorker(self.recognizer, det, korean_agent_url=self.korean_url)
            sess = Session(ws, SafetyMonitor(self.rules), self.scfg, asr=asr, recorder=rec,
                           redirect_pcm=self.redirect_pcm, redirect_text=self.rules.redirect_text)
            await sess.run(self.http)
            st = sess.stats
            self.last_session = {"user_s": round(st.user_ms / 1000, 1), "agent_s": round(st.agent_ms / 1000, 1),
                                 "text_tokens": st.text_tokens, "safety_blocks": st.blocks,
                                 "detect_ms": st.detect_ms[-10:], "already_played_ms": st.leaked_ms[-10:]}
        if not ws.closed:
            await ws.close()
        return ws

    async def api_status(self, request):
        return web.json_response(self.status())

    async def api_gpu(self, request):
        return web.json_response(await asyncio.to_thread(gpu_info))


CORE_KEY = web.AppKey("core", Core)


def make_app(cfg: dict) -> web.Application:
    core = Core(cfg)
    app = web.Application()
    app[CORE_KEY] = core
    app.on_startup.append(core.on_startup)
    app.on_cleanup.append(core.on_cleanup)
    app.router.add_get("/session", core.ws_session)
    app.router.add_get("/api/status", core.api_status)
    app.router.add_get("/api/dev/gpu", core.api_gpu)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.environ.get("EVA_CONFIG", str(ROOT / "config/services.yaml")))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(a.config)
    web.run_app(make_app(cfg), host="127.0.0.1", port=int(cfg["web"]["core_port"]), print=None)


if __name__ == "__main__":
    main()
