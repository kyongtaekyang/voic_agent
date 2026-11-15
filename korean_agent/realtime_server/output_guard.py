"""TTS 전 텍스트 검사 (프롬프트 2.1절 4번).

Gemma 4 가 만든 문장을 음성으로 내보내기 전에 금칙어, 개인정보 요청, 개인정보 값을 검사한다.
텍스트 단계에서 막으므로 부적절한 답변은 소리로 나가지 않는다. 규칙은 config/safety_rules.yaml 에 둔다.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_RULES = Path(__file__).resolve().parent.parent / "config" / "safety_rules.yaml"


@dataclass
class GuardResult:
    allowed: bool
    text: str  # 내보낼 문장 (차단되면 대체 문장)
    rule: str | None = None  # 예: banned_words.profanity, pii_request.ask_phone
    matched: str | None = None
    elapsed_ms: float = 0.0


@dataclass
class OutputGuard:
    replacement: str
    banned: dict[str, list[str]] = field(default_factory=dict)
    pii_request: list[tuple[str, re.Pattern]] = field(default_factory=list)
    pii_value: list[tuple[str, re.Pattern]] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_RULES) -> "OutputGuard":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            replacement=cfg["replacement"],
            banned={k: [w.replace(" ", "") for w in v] for k, v in (cfg.get("banned_words") or {}).items()},
            pii_request=[(r["id"], re.compile(r["pattern"])) for r in cfg.get("pii_request_patterns") or []],
            pii_value=[(r["id"], re.compile(r["pattern"])) for r in cfg.get("pii_value_patterns") or []],
        )

    def check(self, sentence: str) -> GuardResult:
        t0 = time.perf_counter()

        def blocked(rule: str, matched: str) -> GuardResult:
            return GuardResult(False, self.replacement, rule, matched, (time.perf_counter() - t0) * 1000)

        compact = re.sub(r"\s+", "", sentence)
        for cat, words in self.banned.items():
            for w in words:
                if w and w in compact:
                    return blocked(f"banned_words.{cat}", w)
        for rid, pat in self.pii_value:
            m = pat.search(sentence)
            if m:
                return blocked(f"pii_value.{rid}", m.group(0))
        for rid, pat in self.pii_request:
            m = pat.search(sentence)
            if m:
                return blocked(f"pii_request.{rid}", m.group(0))
        return GuardResult(True, sentence, elapsed_ms=(time.perf_counter() - t0) * 1000)

    def sanitize(self, text: str) -> tuple[str, GuardResult | None]:
        """응답 전체에 같은 규칙을 적용한다. 처음 걸린 문장부터 끝까지를 대체 문장 하나로 바꾼다.

        대화 이력과 화면 자막에 남는 텍스트를 실제로 읽은 내용과 맞추기 위해 쓴다.
        반환: (정리된 텍스트, 처음 걸린 결과 또는 None)
        """
        sentences = SENTENCE_SPLIT.split(text.strip()) if text.strip() else []
        kept = []
        for s in sentences:
            r = self.check(s)
            if not r.allowed:
                return " ".join([*kept, self.replacement]), r
            kept.append(s)
        return text, None


SENTENCE_SPLIT = re.compile(r"(?<=[.!?。…])\s+|\n+")
