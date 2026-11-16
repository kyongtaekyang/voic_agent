"""사용자 발화 언어 감시 (프롬프트 2.3절).

Nemotron ASR Streaming 0.6B 의 확정 인식 결과를 발화마다 넣으면, 비영어 발화가 이어질 때 language_mismatch 이벤트를 만든다.
모델 실행은 한국어 Agent 의 transformers 경로(korean_agent/realtime_server/stt_streaming.py)를 그대로 쓸 예정이고,
이 모듈은 판정 규칙만 담는다 (CPU 측정은 bench/ 에서 한다).

인식 언어 설정은 auto 로 한다. en-US 로 고정하면 한국어 발화가 빈 문자열로 나와서 감지할 수 없다
(docs/reports/cpu_prebuild_20261116.md 4.1절: en-US 는 실제 한국어 녹음 7개 중 0개, auto 는 7개 모두 감지).

판정 (발화 하나)
- language=auto 로 인식해서 언어 태그(<en-US>, <ko-KR> ...)가 있으면 태그로 판단한다.
- 태그가 없으면 한글 비율(한글 음절 / (한글 + 라틴 문자))이 hangul_ratio_min 이상일 때 비영어로 본다.
- 인식 결과가 비었거나 너무 짧으면 판단하지 않는다(연속 횟수를 늘리지도, 초기화하지도 않는다).
비영어 발화가 mismatch_turns 번 이어지면 이벤트를 한 번 내고, 영어 발화가 나올 때까지 다시 내지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HANGUL = re.compile(r"[가-힣ㄱ-ㆎ]")
_LATIN = re.compile(r"[A-Za-z]")
_LANG_TAG = re.compile(r"<([a-z]{2})-[A-Z]{2}>")


def split_lang_tag(text: str) -> tuple[str | None, str]:
    """'<ko-KR> 안녕' → ('ko', '안녕'). 태그가 여러 개면 첫 태그를 쓴다."""
    m = _LANG_TAG.search(text)
    return (m.group(1) if m else None), _LANG_TAG.sub("", text).strip()


def hangul_ratio(text: str) -> float | None:
    h, latin = len(_HANGUL.findall(text)), len(_LATIN.findall(text))
    return None if h + latin == 0 else h / (h + latin)


@dataclass
class Utterance:
    text: str
    lang: str | None  # 'en', 'ko', ... 또는 None (판단 불가)
    english: bool | None  # None 이면 판단하지 않음


class LanguageMismatchDetector:
    def __init__(self, mismatch_turns: int = 2, hangul_ratio_min: float = 0.3, min_chars: int = 2):
        self.mismatch_turns = mismatch_turns
        self.hangul_ratio_min = hangul_ratio_min
        self.min_chars = min_chars
        self.reset()

    def reset(self) -> None:
        self.streak = 0
        self.notified = False

    def classify(self, raw_text: str) -> Utterance:
        tag, text = split_lang_tag(raw_text)
        if tag:
            return Utterance(text, tag, tag == "en")
        ratio = hangul_ratio(text)
        if ratio is None or len(text.replace(" ", "")) < self.min_chars:
            return Utterance(text, None, None)
        if ratio >= self.hangul_ratio_min:
            return Utterance(text, "ko", False)
        return Utterance(text, "en", True)

    def feed(self, raw_text: str) -> dict | None:
        """확정 인식 결과 하나를 넣는다. 이벤트를 낼 때만 dict 를 돌려준다 (docs/event_schema.md 의 eva.language_mismatch)."""
        u = self.classify(raw_text)
        if u.english is None:
            return None
        if u.english:
            self.reset()
            return None
        self.streak += 1
        if self.streak >= self.mismatch_turns and not self.notified:
            self.notified = True
            return {"type": "eva.language_mismatch", "detected_language": u.lang, "turns": self.streak,
                    "message": "This program is English only.", "korean_agent_url": None}
        return None
