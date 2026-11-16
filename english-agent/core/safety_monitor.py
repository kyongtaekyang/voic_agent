"""Moshi 텍스트 토큰 실시간 검사 (프롬프트 2.2절).

Moshi 는 음성과 함께 텍스트 토큰을 내보낸다. 토큰을 하나씩 feed() 로 넣으면 지금까지 이어 붙인 텍스트를 규칙과 비교하고,
새로 확정된 위반을 돌려준다. 호출하는 쪽(gateway)은 위반을 받으면 출력 음성을 즉시 끊고 안내 음성을 재생한다.

- 토큰 경계에서 잘린 단어("ki" + "ll")도 이어 붙인 뒤 검사하므로 잡힌다.
- match=word 규칙은 단어가 끝났는지(다음 글자가 경계인지) 확인한 뒤 확정한다. "ass" 가 "assistant" 의 앞부분일 수 있기 때문이다.
  턴이 끝나면 flush() 로 남은 후보를 확정한다.
- match=prefix 규칙은 어간이 보이는 즉시 확정한다.
- 규칙은 config/safety_rules.yaml 에 둔다. 차단 성공률은 실제 Moshi 출력으로만 측정할 수 있다(3단계).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_RULES = Path(__file__).resolve().parent.parent / "config" / "safety_rules.yaml"

_WORDCH = "a-z0-9'"
_LB = f"(?<![{_WORDCH}])"
_LA = f"(?![{_WORDCH}])"
_DROP = re.compile(r"[^a-z0-9'@.]+")
_QUOTES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "▁": " "})


def normalize(text: str) -> str:
    """소문자, 곡선 따옴표 → ', SentencePiece 의 ▁ → 공백, 영문·숫자·'·@·. 이외는 공백 하나로."""
    return _DROP.sub(" ", text.translate(_QUOTES).lower())


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    regex: re.Pattern
    prefix: bool  # True 면 단어 끝 확인 없이 바로 확정


@dataclass
class Violation:
    rule: str  # 예: profanity_stem
    category: str  # 예: profanity, pii_request
    matched: str
    token_index: int  # 위반이 확정된 토큰 번호 (0부터)
    start: int  # 턴 텍스트 안의 시작 위치 (정규화 기준)
    detected_at: float  # time.perf_counter()


def _compile(spec: dict) -> Rule:
    prefix = spec.get("match", "word") == "prefix"
    if "words" in spec:
        alts = sorted({normalize(w).strip() for w in spec["words"]}, key=len, reverse=True)
        body = "|".join(re.escape(a) for a in alts)
    else:
        body = spec["pattern"]
    src = f"{_LB}(?:{body})" + ("" if prefix else _LA)
    return Rule(spec["id"], spec.get("category", spec["id"]), re.compile(src), prefix)


@dataclass
class SafetyRules:
    rules: list[Rule]
    exceptions: list[str] = field(default_factory=list)
    redirect_text: str = ""
    redirect_audio: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_RULES) -> "SafetyRules":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        specs = list(cfg.get("rules") or []) + list(cfg.get("pii_value_rules") or [])
        return cls(
            rules=[_compile(s) for s in specs],
            exceptions=[normalize(e).strip() for e in cfg.get("exceptions") or []],
            redirect_text=cfg.get("redirect_text", ""),
            redirect_audio=cfg.get("redirect_audio"),
        )


class SafetyMonitor:
    """한 턴(Moshi 응답 하나) 단위로 쓴다. 턴이 바뀌면 reset()."""

    def __init__(self, rules: SafetyRules | None = None, window: int = 240):
        self.rules = rules or SafetyRules.from_yaml()
        self.window = window  # 매 토큰마다 다시 검사하는 끝부분 길이 (가장 긴 규칙보다 충분히 길게)
        self.reset()

    def reset(self) -> None:
        self.text = " "  # 정규화한 턴 텍스트. 앞 공백은 첫 단어의 경계 역할
        self.tokens = 0
        self.violations: list[Violation] = []
        self._seen: set[tuple[str, int]] = set()

    @property
    def blocked(self) -> bool:
        return bool(self.violations)

    def feed(self, token: str) -> list[Violation]:
        """토큰 하나를 넣고, 이번에 새로 확정된 위반을 돌려준다."""
        piece = normalize(token)
        if piece:
            if self.text.endswith(" ") and piece.startswith(" "):
                piece = piece.lstrip(" ")
            self.text += piece
            if self.text.endswith("  "):
                self.text = self.text.rstrip(" ") + " "
        idx = self.tokens
        self.tokens += 1
        return self._scan(idx, final=False)

    def flush(self) -> list[Violation]:
        """턴 끝: 단어 끝 확인을 기다리던 후보를 확정한다."""
        return self._scan(max(self.tokens - 1, 0), final=True)

    def check_text(self, text: str) -> list[Violation]:
        """문장 하나를 한꺼번에 검사한다 (테스트·사후 검사용). 상태를 초기화한다."""
        self.reset()
        self.feed(text)
        self.flush()
        return list(self.violations)

    # 내부 ---------------------------------------------------------------------------------------------------------

    def _scan(self, idx: int, final: bool) -> list[Violation]:
        buf = self.text
        lo = max(0, len(buf) - self.window)
        new: list[Violation] = []
        for rule in self.rules.rules:
            for m in rule.regex.finditer(buf, lo):
                key = (rule.id, m.start())
                if key in self._seen:
                    continue
                if not rule.prefix and m.end() == len(buf) and not final:
                    continue  # 단어가 아직 끝나지 않았을 수 있다
                exc = self._excepted(buf, m, final)
                if exc is None:
                    continue  # 예외 구절이 이어지는 중일 수 있다
                self._seen.add(key)
                if exc:
                    continue
                v = Violation(rule.id, rule.category, m.group(0).strip(), idx, m.start(), time.perf_counter())
                new.append(v)
        new.sort(key=lambda v: v.start)
        self.violations.extend(new)
        return new

    def _excepted(self, buf: str, m: re.Match, final: bool) -> bool | None:
        """True: 예외 구절 안에 있음, False: 위반, None: 예외 구절인지 아직 모름."""
        word = m.group(0).strip()
        pending = False
        for e in self.rules.exceptions:
            k = e.find(word)
            while k >= 0:
                s = m.start() + (len(m.group(0)) - len(m.group(0).lstrip())) - k
                if s >= 0 and (s == 0 or not re.match(f"[{_WORDCH}]", buf[s - 1])):
                    seg = buf[s:s + len(e)]
                    after = buf[s + len(e):s + len(e) + 1]
                    if seg == e and (after == "" and final or after != "" and not re.match(f"[{_WORDCH}]", after)):
                        return True
                    if not final and e.startswith(buf[s:]) or seg == e and after == "" and not final:
                        pending = True
                k = e.find(word, k + 1)
        return None if pending else False
