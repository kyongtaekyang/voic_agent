"""세션 녹음 (프롬프트 2.2절 1번): queue/korean/<세션>/ 에 Slow Path 입력을 쌓는다.

- user.wav  : 화면이 보낸 마이크 음성 (16kHz PCM16, input_audio_buffer.append 그대로)
- agent.wav : Realtime 서버가 보낸 응답 음성 (response.output_audio.delta). 벽시계 기준으로 빈 구간을 무음으로 채운다.
              끼어들기로 끊긴 부분도 서버가 보낸 만큼은 들어 있다. 실제로 들려준 길이는 events.jsonl 의 truncate 에 있다.
- events.jsonl : 턴 경계, 실시간 STT 확정 결과(CER 기준), Agent 응답 문장, 끼어들기, kva.* 이벤트. 시각은 세션 시작 기준 초.
- meta.json : 길이와 처리 상태 (queued → transcribed → summarized)
  user_s(사용자 녹음 길이), agent_audio_s(서버가 보낸 응답 음성 길이, 정렬용 무음 제외. 끼어들기로 끊긴 부분도 포함),
  agent_track_s(agent.wav 전체 길이, 정렬용 무음 포함). 발화 비율 같은 지표에는 agent_audio_s 를 쓴다.

녹음은 config/services.yaml slow_path.record 가 true 일 때만 한다. 아동 음성이므로 법정대리인 동의와
보관 기간(slow_path.retention_days) 정책이 정해진 뒤에 켠다. 서버 밖으로 보내지 않는다.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
import wave
from pathlib import Path

RATE = 16000
# events.jsonl 에 남기는 Realtime 이벤트와 남길 필드 (음성 데이터는 빼고)
KEEP = {
    "session.created": (),
    "input_audio_buffer.speech_started": ("audio_start_ms", "item_id"),
    "input_audio_buffer.speech_stopped": ("audio_end_ms", "item_id"),
    "conversation.item.input_audio_transcription.completed": ("item_id", "transcript"),
    "response.created": (),
    "response.done": (),
}


class SessionRecorder:
    def __init__(self, root: str | Path, rate: int = RATE):
        self.dir = Path(root) / f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.rate = rate
        self.t0 = time.monotonic()
        self._user = self._open("user.wav")
        self._agent = self._open("agent.wav")
        self._events = open(self.dir / "events.jsonl", "a", encoding="utf-8")
        self._user_frames = self._agent_frames = self._agent_audio_frames = 0
        self._agent_text: list[str] = []
        self._agent_started = False
        self.closed = False

    def _open(self, name):
        w = wave.open(str(self.dir / name), "wb")
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(self.rate)
        return w

    def _t(self) -> float:
        return round(time.monotonic() - self.t0, 3)

    def event(self, ev: dict) -> None:
        if not self.closed:
            self._events.write(json.dumps({"t": self._t(), **ev}, ensure_ascii=False) + "\n")

    def user_b64(self, b64: str) -> None:
        if self.closed or not b64:
            return
        pcm = base64.b64decode(b64)
        self._user.writeframes(pcm)
        self._user_frames += len(pcm) // 2

    def agent_b64(self, b64: str) -> None:
        if self.closed or not b64:
            return
        pcm = base64.b64decode(b64)
        n = len(pcm) // 2
        gap = int((time.monotonic() - self.t0) * self.rate) - self._agent_frames - n
        if gap > self.rate // 10:
            self._agent.writeframes(b"\x00\x00" * gap)
            self._agent_frames += gap
        self._agent.writeframes(pcm)
        self._agent_frames += n
        self._agent_audio_frames += n

    def client_event(self, msg: dict) -> None:
        """화면 → 서버 이벤트 (필터를 통과한 것)."""
        kind = msg.get("type")
        if kind == "input_audio_buffer.append":
            self.user_b64(msg.get("audio", ""))
        elif kind == "conversation.item.truncate":
            self.event({"type": kind, "item_id": msg.get("item_id"), "audio_end_ms": msg.get("audio_end_ms")})

    def server_event(self, ev: dict) -> None:
        """서버 → 화면 이벤트 (Realtime 이벤트와 kva.* 이벤트)."""
        kind = ev.get("type", "")
        if kind == "response.output_audio.delta":
            if not self._agent_started:
                self._agent_started = True
                self.event({"type": "rec.agent_audio_start"})  # 응답 지연 계산용 (턴 끝 → 첫 음성)
            self.agent_b64(ev.get("delta", ""))
        elif kind == "response.output_audio_transcript.delta":
            self._agent_text.append(ev.get("delta", ""))
        elif kind in KEEP:
            row = {"type": kind, **{k: ev.get(k) for k in KEEP[kind] if k in ev}}
            if kind == "response.done":
                row["text"] = "".join(self._agent_text)
                self._agent_text = []
                self._agent_started = False
            self.event(row)
        elif kind.startswith("kva.") and kind not in ("kva.status",):
            self.event(ev)

    def close(self) -> Path:
        if not self.closed:
            self.closed = True
            self._user.close()
            self._agent.close()
            self._events.close()
            meta = {"agent": "korean", "rate": self.rate, "user_s": round(self._user_frames / self.rate, 2),
                    "agent_audio_s": round(self._agent_audio_frames / self.rate, 2),
                    "agent_track_s": round(self._agent_frames / self.rate, 2), "status": "queued",
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
            (self.dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return self.dir
