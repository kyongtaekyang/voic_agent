"""대화 이력 정리: 출력 검사에 걸린 문장이 LLM 대화 이력에 남지 않게 한다.

speech-to-speech 1.1.0 은 LLM 이 만든 응답 원문을 `AssistantMessage.to_chat_item()` 으로 이력에 넣는다
(LLM/base_openai_compatible_language_model.py). TTS 쪽(KoreanLMOutputProcessor)에서 문장을 막아도 이력에는
원문이 남아 다음 턴 프롬프트에 다시 들어가므로, 이 메서드를 감싸서 같은 규칙으로 정리한다.
결과적으로 이력에는 실제로 읽은 내용(차단 문장부터는 대체 문장)이 남는다. 숫자 정규화는 하지 않는다.
"""

from __future__ import annotations

import logging
from typing import Any

from realtime_server.output_guard import OutputGuard

logger = logging.getLogger(__name__)


def _clean_parts(parts: Any, guard: OutputGuard) -> tuple[Any, list]:
    if not isinstance(parts, list):
        return parts, []
    hits, out = [], []
    for part in parts:
        update = {}
        for attr in ("text", "transcript"):
            value = getattr(part, attr, None)
            if isinstance(value, str) and value.strip():
                cleaned, hit = guard.sanitize(value)
                if hit is not None:
                    update[attr] = cleaned
                    hits.append(hit)
        out.append(part.model_copy(update=update) if update else part)
    return out, hits


def clean_assistant_message(msg: Any, guard: OutputGuard) -> tuple[Any, list]:
    """AssistantMessage(pydantic) 의 content 와 response_item.content 를 정리한 사본과 차단 목록을 돌려준다."""
    content, hits = _clean_parts(getattr(msg, "content", None), guard)
    update: dict[str, Any] = {}
    if hits:
        update["content"] = content
    item = getattr(msg, "response_item", None)
    if item is not None:
        item_content, item_hits = _clean_parts(getattr(item, "content", None), guard)
        if item_hits:
            update["response_item"] = item.model_copy(update={"content": item_content})
            hits += item_hits
    return (msg.model_copy(update=update) if update else msg), hits


def install(guard: OutputGuard) -> None:
    from speech_to_speech.LLM.base_openai_compatible_language_model import AssistantMessage

    if getattr(AssistantMessage.to_chat_item, "_kva_wrapped", False):
        return
    original = AssistantMessage.to_chat_item

    def to_chat_item(self):
        cleaned, hits = clean_assistant_message(self, guard)
        if hits:
            from realtime_server.korean_lm_processor import log_event
            logger.warning("history guard: replaced blocked text in history (rule=%s)", hits[0].rule)
            log_event("history_sanitized", rule=hits[0].rule, matched=hits[0].matched, item_id=self.id)
        return original(cleaned)

    to_chat_item._kva_wrapped = True
    AssistantMessage.to_chat_item = to_chat_item
