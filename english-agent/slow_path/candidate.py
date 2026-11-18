"""LoRA 파인튜닝 데이터 후보 분류와 개인정보 검사 (프롬프트 2.5절 4번, 3절).

전사가 끝난 세션(metrics.json, transcript.json)을 보고 세 가지로 나눈다. 결과는 세션 폴더의 candidate.json 과 meta.json 의 "finetune" 에 쓴다.
- candidate    : 기준을 모두 통과했고 개인정보 규칙에도 걸리지 않음
- needs_review : 품질 기준은 통과했지만 개인정보 규칙에 걸림. 사람이 확인하고 지우거나 가린 뒤에만 학습에 쓴다.
                 확인이 끝나면 meta.json 에 finetune_review: {"approved": true, "by": 이름, "at": 시각} 을 적고 classify 를 다시 돌린다
                 (meta.finetune 을 손으로 바꾸면 다음 classify 가 되돌린다). approved: false 면 rejected
- rejected     : 품질 기준이나 동의 기준을 통과하지 못함 (reasons 에 이유)

음성 파일은 옮기거나 복사하지 않는다 (서버의 queue/english/ 에 그대로 둔다). 학습 데이터 변환은 finetune/ 에서 candidate 세션만 읽는다.
기준값(config/services.yaml slow_path.finetune_candidate)은 초안이다. 실제 세션을 보고 조정한다.

**동의 기준:** 프롬프트 3절에 따라 학습 데이터는 적법하게 확보한 성인 녹음으로 시작하고, 아동 음성은 법정대리인 동의와 내부관리규정 검토 전에는
쓰지 않는다. 그래서 meta.json 에 consent.finetune == true 가 기록된 세션만 후보가 된다. 지금 Gateway 는 이 값을 쓰지 않으므로
(동의 절차 미정) 실제 세션은 모두 rejected("파인튜닝 동의 기록 없음")가 된다. 의도한 동작이다.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from core.safety_monitor import normalize

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PII = ROOT / "config" / "finetune_pii.yaml"
DEFAULT_SAFETY = ROOT / "config" / "safety_rules.yaml"

DEFAULTS = {
    "require_consent": True,
    "min_user_speech_s": 20.0,
    "min_user_turns": 3,
    "max_non_english_turn_ratio": 0.1,
    "user_speech_ratio": [0.2, 0.7],
    "max_agent_cuts_user_rate": 0.2,
    "require_audit": True,  # agent.wav 사후 안전 검사를 했어야 함 (slow_path.audit_agent_audio)
}

_DIGITS = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
           "nine": 9}


class PiiScanner:
    def __init__(self, pii_path: str | Path = DEFAULT_PII, safety_path: str | Path = DEFAULT_SAFETY):
        cfg = yaml.safe_load(Path(pii_path).read_text(encoding="utf-8"))
        specs = list(cfg.get("rules") or [])
        safety = yaml.safe_load(Path(safety_path).read_text(encoding="utf-8"))
        specs += list(safety.get("pii_value_rules") or [])  # 전화번호·이메일 형태는 실시간 검사와 같은 규칙을 쓴다
        self.rules = [(s["id"], re.compile(f"(?<![a-z0-9'])(?:{s['pattern']})")) for s in specs]
        self.spelled_min = int(cfg.get("spelled_digits_min", 7))

    def scan(self, text: str) -> list[dict]:
        t = " " + normalize(text or "") + " "
        found = [{"rule": rid, "at": m.start(), "masked": mask(m.group(0))} for rid, rx in self.rules for m in rx.finditer(t)]
        found += self._spelled_digits(t)
        return sorted(found, key=lambda f: f["at"])

    def _spelled_digits(self, t: str) -> list[dict]:
        out, run, start = [], [], 0
        pos = 0
        for w in t.split(" ") + [""]:
            if w in _DIGITS:
                if not run:
                    start = pos
                run.append(_DIGITS[w])
            else:
                if len(run) >= self.spelled_min and not _counting(run):
                    out.append({"rule": "spelled_digits", "at": start, "masked": f"<숫자 단어 {len(run)}개>"})
                run = []
            pos += len(w) + 1
        return out


def _counting(xs: list[int]) -> bool:
    """1씩 커지거나 작아지는 수열 (숫자 세기 연습)."""
    d = {b - a for a, b in zip(xs, xs[1:])}
    return d in ({1}, {-1})


def mask(s: str) -> str:
    """결과 파일에 개인정보를 다시 적지 않도록 가린다. 숫자나 @ 가 든 단어는 모두, 마지막 단어는 첫 글자만 남긴다 (예: "my name is j***")."""
    words = s.strip().split(" ")
    out = []
    for i, w in enumerate(words):
        if any(ch.isdigit() for ch in w) or "@" in w:
            out.append("***")
        elif i == len(words) - 1:
            out.append(w[:1] + "***")
        else:
            out.append(w)
    return " ".join(out)


def classify(meta: dict, metrics: dict, transcript: dict | None, cfg: dict | None, scanner: PiiScanner) -> dict:
    c = DEFAULTS | (cfg or {})
    reasons: list[str] = []
    if c["require_consent"] and (meta.get("consent") or {}).get("finetune") is not True:
        reasons.append("파인튜닝 동의 기록 없음 (meta.consent.finetune)")
    if transcript is None:
        reasons.append("전사 없음")
    if metrics.get("transcript_truncated"):
        reasons.append("전사가 잘림")
    if metrics.get("errors"):
        reasons.append(f"세션 오류 {metrics['errors']}건")
    s = metrics.get("safety") or {}
    if s.get("blocks"):
        reasons.append(f"실시간 안전 차단 {s['blocks']}건")
    audit = s.get("audit")
    if audit is None and c["require_audit"]:
        reasons.append("사후 안전 검사 안 함 (audit_agent_audio)")
    elif audit and audit.get("violations_in_agent_audio"):
        reasons.append(f"사후 검사 위반 {audit['violations_in_agent_audio']}건")
    tt = metrics.get("turn_taking") or {}
    if (tt.get("user_speech_s") or 0) < c["min_user_speech_s"]:
        reasons.append(f"사용자 발화 {tt.get('user_speech_s')}초 < {c['min_user_speech_s']}")
    if (tt.get("user_turns") or 0) < c["min_user_turns"]:
        reasons.append(f"사용자 턴 {tt.get('user_turns')} < {c['min_user_turns']}")
    ratio = tt.get("user_speech_ratio")
    lo, hi = c["user_speech_ratio"]
    if ratio is None or not lo <= ratio <= hi:
        reasons.append(f"사용자 발화 비율 {ratio} 범위 밖 [{lo}, {hi}]")
    cut = tt.get("agent_cuts_user_rate")
    if cut is not None and cut > c["max_agent_cuts_user_rate"]:
        reasons.append(f"Agent 가 말을 끊은 비율 {cut} > {c['max_agent_cuts_user_rate']}")
    turns = metrics.get("asr_turns") or 0
    if turns and metrics.get("non_english_turns", 0) / turns > c["max_non_english_turn_ratio"]:
        reasons.append(f"비영어 턴 {metrics['non_english_turns']}/{turns}")

    pii = []
    for who in ("user", "agent"):
        for f in scanner.scan(((transcript or {}).get(who) or {}).get("text", "")):
            pii.append({"source": who, **f})
    # 사람 검토 기록 (meta.json finetune_review: {"approved": true|false, "by": ..., "at": ..., "note": ...}).
    # 다시 분류해도 검토 결과를 덮어쓰지 않도록 여기서 반영한다. 개인정보를 가린 뒤 approved 를 true 로 적는다
    review = meta.get("finetune_review") or {}
    if review.get("approved") is False:
        reasons.append("검토자가 제외함 (finetune_review)")
    reviewed = review.get("approved") is True
    decision = "rejected" if reasons else ("needs_review" if pii and not reviewed else "candidate")
    return {"decision": decision, "reasons": reasons, "pii": pii, "pii_reviewed": bool(pii) and reviewed,
            "criteria": {k: c[k] for k in DEFAULTS}, "note": "기준값은 초안 (config/services.yaml slow_path.finetune_candidate)"}
