"""한국어 Agent 화면(7201)과 Realtime 서버(8766) 사이 중계 로직. 서버 없이 시험할 수 있게 web/server.py 와 나눴다.

- filter_client_event : 브라우저가 보낸 이벤트 가운데 허용된 것만, 허용된 필드만 서버로 넘긴다.
  시스템 프롬프트를 바꾸는 instructions, tools, 텍스트 입력(conversation.item.create), response.create 는 막는다.
- TurnMetrics         : 화면 기준 지연(발화 끝 → 턴 판정 통보, 발화 끝 → 첫 음성)을 잰다. 단계별 지연은
                        speech-to-speech 의 추측 실행 때문에 화면에서 잴 수 없으므로 서버 로그(turn_latency)를 쓴다.
- EventTail           : realtime_server 가 남기는 logs/events.jsonl 에서 새 줄을 읽는다
                        (speech-to-speech 는 num_pipelines=1 이라 동시에 한 세션만 처리하므로 세션 구분이 필요 없다).
- read_mode           : ada-ops 운영 모드(run/mode)를 읽는다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SNAPSHOT_EXT = "speech_to_speech.input_audio_transcription.snapshot"
FIXED_TURN_DETECTION = {"type": "server_vad", "interrupt_response": True}
ALLOWED_CLIENT_TYPES = {"session.update", "input_audio_buffer.append", "input_audio_buffer.clear",
                        "response.cancel", "conversation.item.truncate"}
MAX_APPEND_B64 = 256 * 1024  # 16kHz PCM16 약 6초 분량. 브라우저는 32~128ms 단위로 보낸다
PCM16_BYTES_PER_MS = 32  # 16kHz * 2 bytes / 1000


def filter_client_event(msg: Any) -> dict | None:
    """허용하지 않는 이벤트는 None. session.update 는 오디오 형식과 확장만 남기고 턴 판정은 고정한다."""
    if not isinstance(msg, dict) or msg.get("type") not in ALLOWED_CLIENT_TYPES:
        return None
    kind = msg["type"]
    if kind == "input_audio_buffer.append":
        audio = msg.get("audio")
        if not isinstance(audio, str) or not audio or len(audio) > MAX_APPEND_B64:
            return None
        return {"type": kind, "audio": audio}
    if kind == "session.update":
        session = msg.get("session") if isinstance(msg.get("session"), dict) else {}
        audio = session.get("audio") if isinstance(session.get("audio"), dict) else {}
        ain = audio.get("input") if isinstance(audio.get("input"), dict) else {}
        aout = audio.get("output") if isinstance(audio.get("output"), dict) else {}
        new_in: dict[str, Any] = {"turn_detection": dict(FIXED_TURN_DETECTION)}
        new_out: dict[str, Any] = {}
        if isinstance(ain.get("format"), dict):
            new_in["format"] = ain["format"]
        if isinstance(aout.get("format"), dict):
            new_out["format"] = aout["format"]
        out: dict[str, Any] = {"type": "realtime", "audio": {"input": new_in, "output": new_out}}
        if SNAPSHOT_EXT in (session.get("extensions") or []):
            out["extensions"] = [SNAPSHOT_EXT]
        return {"type": kind, "session": out}
    if kind == "conversation.item.truncate":
        try:
            return {"type": kind, "item_id": str(msg["item_id"]), "content_index": int(msg.get("content_index", 0)),
                    "audio_end_ms": max(0, int(msg["audio_end_ms"]))}
        except (KeyError, TypeError, ValueError):
            return None
    return {"type": kind}


def audio_ms_of_append(msg: dict) -> float:
    """input_audio_buffer.append 한 개가 담은 음성 길이(ms). base64 길이로 계산한다."""
    b64 = msg.get("audio", "")
    n_bytes = len(b64) * 3 // 4 - (len(b64) - len(b64.rstrip("=")))
    return n_bytes / PCM16_BYTES_PER_MS


@dataclass
class TurnMetrics:
    """서버 이벤트로 턴별 지연을 잰다. 시각은 중계 서버의 monotonic 시계(ms)."""

    audio_ms_sent: float = 0.0  # 지금까지 서버로 보낸 사용자 음성 길이 (실시간으로 보내므로 음성 시간축과 같다)
    speech_pad_ms: float = 0.0  # VAD 가 audio_start_ms 를 앞당긴 양 (services.yaml turn.speech_pad_ms)
    _t: dict[str, float] = field(default_factory=dict)
    _v: dict[str, float | None] = field(default_factory=dict)
    turn: int = 0

    def add_audio(self, ms: float) -> None:
        self.audio_ms_sent += ms

    def feed(self, event: dict, now_ms: float | None = None) -> list[dict]:
        """서버 이벤트 하나를 넣는다. 화면으로 보낼 kva.* 이벤트 목록을 돌려준다."""
        now = time.monotonic() * 1000 if now_ms is None else now_ms
        kind = event.get("type")
        out: list[dict] = []
        if kind == "input_audio_buffer.speech_started":
            start = event.get("audio_start_ms")
            if isinstance(start, (int, float)):
                # 실제 발화 시작 = audio_start_ms + speech_pad_ms. 감지 지연 = 지금까지 보낸 음성 길이 - 실제 발화 시작
                out.append({"type": "kva.vad", "event": "speech_started",
                            "detect_delay_ms": round(self.audio_ms_sent - start - self.speech_pad_ms)})
        elif kind == "input_audio_buffer.speech_stopped":
            self._t = {"stopped": now}
            self._v = {}
            end = event.get("audio_end_ms")
            self._v["turn_decision_ms"] = (self.audio_ms_sent - end) if isinstance(end, (int, float)) else None
        elif kind == "response.output_audio.delta" and "stopped" in self._t and "audio" not in self._t:
            # 단계별 값(STT, LLM, TTS)은 추측 실행 때문에 화면 쪽에서 잴 수 없다 → 서버 기준 kva.server_latency 를 쓴다
            self._t["audio"] = now
            decision = self._v.get("turn_decision_ms") or 0.0
            self._v["e2e_ms"] = now - self._t["stopped"] + decision
        elif kind == "response.done" and "stopped" in self._t:
            self.turn += 1
            status = (event.get("response") or {}).get("status")
            out.append({"type": "kva.metrics", "turn": self.turn, "status": status,
                        **{k: (round(v) if isinstance(v, (int, float)) else None) for k, v in self._v.items()}})
            self._t, self._v = {}, {}
        return out


class PlaybackTracker:
    """응답 음성 안에서 각 문장이 시작하는 위치(ms)를 기록해, 끼어들기 때 실제로 들려준 문장만 이력에 남긴다.

    speech-to-speech 1.1.0 은 conversation.item.truncate 를 받기만 하고 이력을 고치지 않는다. 또 TTS 가 실시간보다 빨라
    response.done 이 재생보다 먼저 오므로, 응답 전체가 이력에 남는다. 그래서 중계 서버가 truncate 를 받으면
    그 assistant 항목을 지우고(conversation.item.delete) 끝까지 들려준 문장만으로 다시 만든다(conversation.item.create).
    끝까지 듣지 못한 문장은 넣지 않는다.
    """

    def __init__(self, max_items: int = 8):
        self.items: dict[str, dict] = {}
        self.max_items = max_items

    def _item(self, item_id: str) -> dict:
        if item_id not in self.items:
            if len(self.items) >= self.max_items:
                self.items.pop(next(iter(self.items)))
            self.items[item_id] = {"audio_ms": 0.0, "sentences": []}
        return self.items[item_id]

    def feed(self, event: dict) -> None:
        kind = event.get("type")
        item_id = event.get("item_id")
        if not item_id:
            return
        if kind == "response.output_audio_transcript.delta" and event.get("delta"):
            it = self._item(item_id)
            it["sentences"].append([it["audio_ms"], event["delta"]])
        elif kind == "response.output_audio.delta":
            it = self._item(item_id)
            it["audio_ms"] += audio_ms_of_append({"audio": event.get("delta", "")})

    def rewrite(self, item_id: str, audio_end_ms: int) -> list[dict]:
        """truncate 를 이력 수정 이벤트로 바꾼다. 다 들었거나 모르는 항목이면 빈 목록."""
        it = self.items.get(item_id)
        if not it or audio_end_ms >= it["audio_ms"]:
            return []
        starts = [s[0] for s in it["sentences"]] + [it["audio_ms"]]
        played = [text for (start, text), end in zip(it["sentences"], starts[1:]) if end <= audio_end_ms]
        self.items.pop(item_id, None)
        events: list[dict] = [{"type": "conversation.item.delete", "item_id": item_id}]
        text = " ".join(t.strip() for t in played if t.strip())
        if text:
            events.append({"type": "conversation.item.create", "item": {
                "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}})
        return events


class EventTail:
    """JSONL 파일 끝에서부터 새 줄만 읽는다. 파일이 지워지거나 새로 만들어지면 처음부터 읽는다."""

    def __init__(self, path: str | Path, from_end: bool = True):
        self.path = Path(path)
        self.pos = self.path.stat().st_size if (from_end and self.path.exists()) else 0

    def read(self) -> list[dict]:
        if not self.path.exists():
            self.pos = 0
            return []
        size = self.path.stat().st_size
        if size < self.pos:
            self.pos = 0
        if size == self.pos:
            return []
        with self.path.open("rb") as f:
            f.seek(self.pos)
            data = f.read()
        end = data.rfind(b"\n")
        if end < 0:
            return []
        self.pos += end + 1
        rows = []
        for line in data[:end].splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        return rows


def to_screen_events(row: dict) -> list[dict]:
    """events.jsonl 한 줄을 화면 이벤트로 바꾼다. 사용자 화면에는 language_mismatch 만, 나머지는 개발자 화면용."""
    kind = row.get("type")
    if kind == "language_mismatch":
        return [{"type": "kva.language_mismatch", "streak": row.get("streak")}]
    if kind == "turn_latency":
        return [{"type": "kva.server_latency", **{k: v for k, v in row.items() if k not in ("type", "ts")}}]
    if kind in ("guard_blocked", "history_sanitized"):
        return [{"type": "kva.guard", "event": kind, "rule": row.get("rule"), "matched": row.get("matched"),
                 "ts": row.get("ts")}]
    return []


def read_mode(ops_dir: str | Path) -> str | None:
    p = Path(ops_dir) / "run" / "mode"
    try:
        return p.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None
