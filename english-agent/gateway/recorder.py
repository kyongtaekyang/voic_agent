"""세션 녹음 (프롬프트 2.5절). queue/english/<세션>/ 에 user.wav, agent.wav(사용자가 실제로 들은 음성), events.jsonl, meta.json 을 쓴다.

Slow Path(후처리 모드 야간 배치)가 이 폴더를 읽는다. 저장소에 올리지 않고(.gitignore 의 **/queue/), 서버 밖으로 보내지 않는다.
"""

from __future__ import annotations

import json
import time
import uuid
import wave
from pathlib import Path

import numpy as np


class Recorder:
    def __init__(self, root: str | Path, rate: int = 24000):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / f"{stamp}_{uuid.uuid4().hex[:8]}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.rate = rate
        self.t0 = time.monotonic()
        self._user = self._open("user.wav")
        self._agent = self._open("agent.wav")
        self._events = open(self.dir / "events.jsonl", "a", encoding="utf-8")
        self._agent_frames = 0
        self._user_frames = 0
        self.closed = False

    def _open(self, name: str):
        w = wave.open(str(self.dir / name), "wb")
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(self.rate)
        return w

    def _elapsed_frames(self) -> int:
        return int((time.monotonic() - self.t0) * self.rate)

    def _write(self, w, pcm: np.ndarray) -> None:
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())

    def user(self, pcm: np.ndarray) -> None:
        if not self.closed:
            self._write(self._user, pcm)
            self._user_frames += pcm.size

    def agent(self, pcm: np.ndarray) -> None:
        """Agent 음성은 끊겨서 오므로, 벽시계 기준으로 빈 구간을 무음으로 채워 사용자 녹음과 시간을 맞춘다."""
        if self.closed:
            return
        gap = self._elapsed_frames() - self._agent_frames - pcm.size
        if gap > self.rate // 10:
            self._write(self._agent, np.zeros(gap, np.float32))
            self._agent_frames += gap
        self._write(self._agent, pcm)
        self._agent_frames += pcm.size

    def event(self, ev: dict) -> None:
        if not self.closed:
            self._events.write(json.dumps({"t": round(time.monotonic() - self.t0, 3), **ev}, ensure_ascii=False) + "\n")

    def close(self, meta: dict | None = None) -> None:
        if self.closed:
            return
        self.closed = True
        self._user.close()
        self._agent.close()
        self._events.close()
        info = {"rate": self.rate, "user_s": round(self._user_frames / self.rate, 2),
                "agent_s": round(self._agent_frames / self.rate, 2), "status": "queued", **(meta or {})}
        (self.dir / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
