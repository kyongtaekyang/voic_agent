"""speech-to-speech 의 LMOutputProcessor 를 상속해 출력 검사와 텍스트 정규화를 넣는다.

LMOutputProcessor.process() 는 LLM 문장을 TTS 로 넘기는 유일한 통로다 (speech_to_speech/LLM/lm_output_processor.py).
여기서 문장마다
  1) output_guard 로 검사하고, 걸리면 대체 문장으로 바꾸고 같은 응답의 나머지 문장은 버린다.
  2) text_normalizer 로 숫자·기호를 읽는 형태로 바꾼다.
화면 자막(AssistantOutputEvent)도 같은 문장을 받으므로, 차단된 문장은 자막에도 나오지 않는다.

대화 이력에 남는 원문은 LLM 쪽에서 기록되므로 realtime_server/history_guard.py 가 따로 정리한다.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path

from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.messages import AssistantTextPart, EndOfResponse, LLMResponseChunk

from realtime_server.output_guard import OutputGuard
from realtime_server.text_normalizer import normalize

logger = logging.getLogger(__name__)

EVENT_LOG = Path(os.environ.get("KVA_EVENT_LOG", Path(__file__).resolve().parent.parent / "logs" / "events.jsonl"))


def log_event(kind: str, **fields) -> None:
    """개발자 화면과 보고서용 이벤트를 JSONL 로 남긴다. 원문 텍스트는 차단 사유 확인용으로만 남긴다."""
    try:
        EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with EVENT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "type": kind, **fields}, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("event log write failed")


class KoreanLMOutputProcessor(LMOutputProcessor):
    def setup(self, *args, **kwargs) -> None:
        super().setup(*args, **kwargs)
        self.guard = OutputGuard.from_yaml(os.environ.get("KVA_SAFETY_RULES") or
                                           Path(__file__).resolve().parent.parent / "config" / "safety_rules.yaml")
        self._blocked_responses: set[str] = set()
        logger.info("KoreanLMOutputProcessor: output guard + text normalizer enabled")

    def _rewrite(self, chunk: LLMResponseChunk) -> LLMResponseChunk:
        key = chunk.response_key or self._response_key or "?"
        parts = []
        for part in chunk.parts:
            if not isinstance(part, AssistantTextPart) or not part.text.strip():
                parts.append(part)
                continue
            if key in self._blocked_responses:
                logger.info("guard: dropping sentence after block (response=%s)", key)
                continue
            r = self.guard.check(part.text)
            text = r.text
            if not r.allowed:
                self._blocked_responses.add(key)
                logger.warning("guard: blocked rule=%s matched=%r", r.rule, r.matched)
                log_event("guard_blocked", rule=r.rule, matched=r.matched, text=part.text, response_key=key,
                          turn_id=chunk.turn_id, elapsed_ms=round(r.elapsed_ms, 3))
            spoken = normalize(text)
            if spoken != text:
                log_event("normalized", before=text, after=spoken, response_key=key)
            parts.append(part.model_copy(update={"text": spoken}))
        return chunk.model_copy(update={"parts": parts})

    def process(self, lm_output) -> Iterator:
        if isinstance(lm_output, LLMResponseChunk):
            lm_output = self._rewrite(lm_output)
        elif isinstance(lm_output, EndOfResponse):
            key = lm_output.response_key or self._response_key
            if key:
                self._blocked_responses.discard(key)
        yield from super().process(lm_output)

    def on_session_end(self) -> None:
        self._blocked_responses.clear()
        super().on_session_end()
