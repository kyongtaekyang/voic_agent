"""대화 세션 하나: 브라우저 ↔ Gateway ↔ Moshi 서버 (프롬프트 2.2·2.3·2.6절).

브라우저와는 PCM16(24kHz 모노, little-endian) 바이너리와 JSON 이벤트로 주고받고, Moshi 서버와는 Moshi 바이너리 규격
(0x00 handshake, 0x01 Ogg/Opus, 0x02 텍스트)으로 주고받는다. Opus 변환은 Gateway 가 sphn 으로 한다.
그래서 Gateway 가 원음 PCM 을 손에 쥐고 다음 일을 한 곳에서 한다.

- 안전 감시: Moshi 텍스트 토큰을 SafetyMonitor 에 넣고, 위반이면 아직 내보내지 않은 음성을 버리고 안내 음성을 보낸다.
  출력 음성은 output_delay_ms 만큼 붙잡아 두었다가 내보낸다 (0 이면 바로). 텍스트 토큰이 idle_ms 동안 없으면 idle() 을 부른다.
  차단 뒤에는 Moshi 가 unmute_quiet_ms 동안 조용해질 때까지 음성과 자막을 내보내지 않는다.
- ASR 병행 감시: 사용자 음성을 AsrWorker 로 넘긴다 (자막, language_mismatch).
- 녹음: 사용자·Agent 음성과 이벤트를 queue/english/ 에 남긴다 (Slow Path 용, 서버 밖으로 보내지 않음).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field

import aiohttp
import numpy as np
import sphn

from core.safety_monitor import SafetyMonitor, Violation

log = logging.getLogger("eva.session")
RATE = 24000


@dataclass
class SessionConfig:
    moshi_url: str = "ws://127.0.0.1:8998/api/chat"
    handshake_timeout_s: float = 60.0
    output_delay_ms: int = 0
    idle_ms: int = 160
    unmute_quiet_ms: int = 1000
    speaking_rms: float = 0.01
    tick_ms: int = 20


@dataclass
class Stats:
    started: float = field(default_factory=time.monotonic)
    user_ms: float = 0.0
    agent_ms: float = 0.0
    text_tokens: int = 0
    blocks: int = 0
    dropped_ms: float = 0.0  # 차단으로 버린 음성 길이
    leaked_ms: list = field(default_factory=list)  # 위반 텍스트 도착 전에 이미 내보낸 그 발화의 음성 길이 (참고값)
    detect_ms: list = field(default_factory=list)  # 위반 단어 첫 토큰 도착 → 판정


def pcm16_to_f32(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0


def f32_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class Session:
    def __init__(self, client_ws, monitor: SafetyMonitor, cfg: SessionConfig, *, asr=None, recorder=None,
                 redirect_pcm: np.ndarray | None = None, redirect_text: str = ""):
        self.client = client_ws
        self.monitor = monitor
        self.cfg = cfg
        self.asr = asr
        self.rec = recorder
        self.redirect_pcm = redirect_pcm
        self.redirect_text = redirect_text
        self.stats = Stats()
        self.moshi = None
        self._writer = sphn.OpusStreamWriter(RATE)
        self._reader = sphn.OpusStreamReader(RATE)
        self._out: deque[tuple[float, np.ndarray]] = deque()  # (내보낼 시각, pcm)
        self._last_text = 0.0
        self._idle_pending = False
        self._held: str | None = None  # 아직 자막으로 내보내지 않은 마지막 텍스트 토큰
        self._muted = False
        self._state = "listening"
        self._last_loud = 0.0
        self._tok_times: deque[float] = deque(maxlen=64)  # 최근 텍스트 토큰 도착 시각 (차단 판정 지연 계산용)
        self._utt_sent_ms = 0.0  # 현재 발화(마지막 침묵 이후)에서 이미 내보낸 음성 길이
        self._closed = False

    # 브라우저로 보내기 --------------------------------------------------------------------------------------------
    async def emit(self, ev: dict) -> None:
        if self.rec:
            self.rec.event(ev)
        if not self.client.closed:
            await self.client.send_str(json.dumps(ev, ensure_ascii=False))

    async def _send_audio(self, pcm: np.ndarray) -> None:
        if pcm.size == 0 or self.client.closed:
            return
        ms = 1000.0 * pcm.size / RATE
        now = time.monotonic()
        loud = float(np.sqrt(np.mean(pcm * pcm))) >= self.cfg.speaking_rms
        if loud and now - self._last_loud > 0.5:
            self._utt_sent_ms = 0.0  # 0.5초 넘게 조용했다가 다시 말하기 시작: 새 발화
        if loud:
            self._last_loud = now
        self.stats.agent_ms += ms
        self._utt_sent_ms += ms
        if self.rec:
            self.rec.agent(pcm)
        await self.client.send_bytes(f32_to_pcm16(pcm))

    # 실행 -------------------------------------------------------------------------------------------------------
    async def run(self, http: aiohttp.ClientSession) -> None:
        try:
            self.moshi = await http.ws_connect(self.cfg.moshi_url, max_msg_size=0)
        except Exception as e:  # noqa: BLE001
            await self.emit({"type": "eva.unavailable", "reason": "moshi_unreachable", "detail": str(e)[:200]})
            return
        try:
            msg = await asyncio.wait_for(self.moshi.receive(), self.cfg.handshake_timeout_s)
            if msg.type != aiohttp.WSMsgType.BINARY or not msg.data or msg.data[0] != 0:
                await self.emit({"type": "eva.unavailable", "reason": "moshi_bad_handshake"})
                return
        except asyncio.TimeoutError:
            await self.emit({"type": "eva.unavailable", "reason": "moshi_busy"})  # 서버는 한 번에 한 연결만 받는다
            return
        await self.emit({"type": "eva.session.started", "sample_rate": RATE})
        await self._set_state("listening", force=True)
        tasks = [asyncio.create_task(t) for t in (self._pump_client(), self._pump_moshi(), self._ticker())]
        if self.asr:
            tasks.append(asyncio.create_task(self.asr.run(self.emit)))
        try:
            await asyncio.wait(tasks[:2], return_when=asyncio.FIRST_COMPLETED)
        finally:
            self._closed = True
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if not self.moshi.closed:
                await self.moshi.close()
            await self.emit_summary()

    async def emit_summary(self) -> None:
        s = self.stats
        ev = {"type": "eva.session.ended", "duration_s": round(time.monotonic() - s.started, 1),
              "user_s": round(s.user_ms / 1000, 1), "agent_s": round(s.agent_ms / 1000, 1),
              "text_tokens": s.text_tokens, "safety_blocks": s.blocks}
        if self.rec:
            self.rec.event(ev)
            self.rec.close()
        if not self.client.closed:
            await self.client.send_str(json.dumps(ev))

    async def _pump_client(self) -> None:
        async for msg in self.client:
            if msg.type == aiohttp.WSMsgType.BINARY:
                pcm = pcm16_to_f32(msg.data)
                if pcm.size == 0:
                    continue
                self.stats.user_ms += 1000.0 * pcm.size / RATE
                if self.rec:
                    self.rec.user(pcm)
                if self.asr:
                    self.asr.feed(pcm)
                opus = self._writer.append_pcm(pcm)
                if opus:
                    await self.moshi.send_bytes(b"\x01" + opus)
            elif msg.type == aiohttp.WSMsgType.TEXT:
                pass  # 화면 → Core 제어 이벤트는 아직 없다 (종료는 WebSocket 닫기)
            else:
                break

    async def _pump_moshi(self) -> None:
        async for msg in self.moshi:
            if msg.type != aiohttp.WSMsgType.BINARY or not msg.data:
                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
                continue
            kind, payload = msg.data[0], msg.data[1:]
            now = time.monotonic()
            if kind == 1:
                pcm = self._reader.append_bytes(payload)
                if pcm.size:
                    await self._on_agent_audio(pcm, now)
            elif kind == 2:
                await self._on_text(payload.decode("utf-8", "replace"), now)
            elif kind == 5:
                await self.emit({"type": "eva.error", "source": "moshi", "detail": payload.decode("utf-8", "replace")[:200]})

    async def _on_agent_audio(self, pcm: np.ndarray, now: float) -> None:
        if self._idle_pending and now - self._last_text >= self.cfg.idle_ms / 1000:
            self._idle_pending = False
            v = self.monitor.idle()
            if v:
                self._held = None
                await self._block(v, now)
            else:
                await self._flush_held()
        if self._muted:
            if now - self._last_text >= self.cfg.unmute_quiet_ms / 1000:
                self._muted = False
                self._held = None
                await self.emit({"type": "eva.safety_resumed"})
            else:
                self.stats.dropped_ms += 1000.0 * pcm.size / RATE
                return
        self._out.append((now + self.cfg.output_delay_ms / 1000, pcm))
        await self._release(now)

    async def _on_text(self, text: str, now: float) -> None:
        self.stats.text_tokens += 1
        self._last_text = now
        self._idle_pending = True
        self._tok_times.append(now)
        v = self.monitor.feed(text)
        if v:
            self._held = None  # 판정을 기다리던 토큰은 자막으로 내보내지 않는다
            await self._block(v, now)
            return
        # 자막은 토큰 하나 늦게 낸다: word 규칙은 다음 토큰(또는 idle)에서 확정되므로, 마지막 토큰은 아직 위반일 수 있다
        prev, self._held = self._held, text
        if prev is not None and not self._muted:
            await self.emit({"type": "eva.agent_text", "delta": prev})

    async def _flush_held(self) -> None:
        prev, self._held = self._held, None
        if prev is not None and not self._muted:
            await self.emit({"type": "eva.agent_text", "delta": prev})

    async def _block(self, violations: list[Violation], now: float) -> None:
        dropped = sum(p.size for _, p in self._out)
        self._out.clear()
        self.stats.blocks += 1
        self.stats.dropped_ms += 1000.0 * dropped / RATE
        self.stats.leaked_ms.append(round(self._utt_sent_ms))
        v = violations[0]
        # 판정 지연: 위반이 확정된 토큰 기준. 토큰 도착 시각은 최근 64개만 보관한다
        back = self.monitor.tokens - 1 - v.token_index
        t_tok = self._tok_times[-1 - back] if 0 <= back < len(self._tok_times) else now
        self.stats.detect_ms.append(round(1000 * (now - t_tok), 1))
        was_muted, self._muted = self._muted, True
        if was_muted:
            return
        await self.emit({"type": "eva.safety_block", "rule": v.rule, "category": v.category,
                         "token_index": v.token_index, "dropped_ms": round(1000.0 * dropped / RATE),
                         "already_played_ms": self.stats.leaked_ms[-1], "redirect_text": self.redirect_text})
        if self.redirect_pcm is not None:
            await self._send_audio(self.redirect_pcm)

    async def _release(self, now: float) -> None:
        while self._out and self._out[0][0] <= now:
            _, pcm = self._out.popleft()
            await self._send_audio(pcm)

    async def _set_state(self, state: str, force: bool = False) -> None:
        if state != self._state or force:
            self._state = state
            await self.emit({"type": "eva.state", "state": state})

    async def _ticker(self) -> None:
        while not self._closed:
            await asyncio.sleep(self.cfg.tick_ms / 1000)
            now = time.monotonic()
            await self._release(now)
            await self._set_state("speaking" if now - self._last_loud < 0.3 else "listening")
