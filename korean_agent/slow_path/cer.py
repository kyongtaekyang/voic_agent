"""문자 오류율(CER). 한국어는 띄어쓰기와 문장부호가 인식기마다 달라서, 비교 전에 공백과 문장부호를 지운다."""

from __future__ import annotations

import re
import unicodedata

_DROP = re.compile(r"[\s\W_]+", re.UNICODE)
_TAG = re.compile(r"<[^>]{1,20}>")  # <ko-KR> 같은 언어 태그, 특수 토큰


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", _TAG.sub(" ", text or ""))
    return _DROP.sub("", text).lower()


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> dict:
    """reference(축어 전사) 기준으로 hypothesis(실시간 STT) 의 CER. 문자 수 0 이면 cer=None."""
    r, h = normalize(reference), normalize(hypothesis)
    d = edit_distance(r, h)
    return {"cer": round(d / len(r), 4) if r else None, "errors": d, "ref_chars": len(r), "hyp_chars": len(h)}
