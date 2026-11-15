"""speech-to-speech 가 턴마다 남기는 지연 로그를 events.jsonl 로 옮긴다 (개발자 화면용).

speech-to-speech 1.1.0 은 응답이 끝나면 다음 형식의 INFO 로그를 남긴다 (api/openai_realtime/handlers/response.py).
  Turn turn_1 rev=11 latency: stt=2.34s llm=1.02s tts_ttfa=n/a e2e=n/a vad_decision=0.45s hold=0.00s
      smart_turn_status=complete status=completed source=speech response_key=...
추측 실행(speculative turn) 때문에 화면 쪽 이벤트 시각으로는 단계별 지연을 알 수 없으므로 이 값을 쓴다.
"""

from __future__ import annotations

import logging
import re

from realtime_server.korean_lm_processor import log_event

LOGGER_NAME = "speech_to_speech.api.openai_realtime.handlers.response"
_LINE = re.compile(r"Turn (?P<turn>\S+) rev=(?P<rev>\d+) latency: (?P<rest>.*)")
_PAIR = re.compile(r"(\w+)=(\S+)")
_KEEP = {"stt", "llm", "tts_ttfa", "e2e", "vad_decision", "hold", "smart_turn_status", "status", "source"}


def parse(message: str) -> dict | None:
    m = _LINE.search(message)
    if not m:
        return None
    out: dict = {"turn_id": m.group("turn"), "revision": int(m.group("rev"))}
    for key, value in _PAIR.findall(m.group("rest")):
        if key not in _KEEP:
            continue
        if value.endswith("s") and value[:-1].replace(".", "", 1).isdigit():
            out[f"{key}_ms"] = round(float(value[:-1]) * 1000)
        elif value == "n/a":
            out[f"{key}_ms"] = None
        else:
            out[key] = value
    return out


class TurnLatencyHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            row = parse(record.getMessage())
            if row:
                log_event("turn_latency", **row)
        except Exception:  # noqa: BLE001
            pass


def install() -> None:
    lg = logging.getLogger(LOGGER_NAME)
    if not any(isinstance(h, TurnLatencyHandler) for h in lg.handlers):
        lg.addHandler(TurnLatencyHandler(level=logging.INFO))
