"""단어 오류율(WER). 영어 기준: 소문자로 바꾸고, 단어 안의 아포스트로피(don't)만 남기고 문장부호를 지운 뒤 공백으로 나눈다.

한국어 Agent 의 cer.normalize 는 공백까지 지워서 단어 경계가 사라지므로 여기서는 쓰지 않는다.
"""

from __future__ import annotations

import re
import unicodedata

_TAG = re.compile(r"<[^>]{1,20}>")  # <en-US>, <spk:0> 같은 태그, 특수 토큰
_APOS = re.compile(r"[’‘`]")
_PUNCT = re.compile(r"[^\w\s']|_", re.UNICODE)
_EDGE_APOS = re.compile(r"(?<!\w)'|'(?!\w)")


def words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", _TAG.sub(" ", text or ""))
    text = _APOS.sub("'", text).lower()
    text = _EDGE_APOS.sub(" ", _PUNCT.sub(" ", text))
    return text.split()


def edit_distance(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> dict:
    """reference(Transcribe 30B 축어 전사) 기준으로 hypothesis(ASR Streaming 자막) 의 WER. 단어 수 0 이면 wer=None."""
    r, h = words(reference), words(hypothesis)
    d = edit_distance(r, h)
    return {"wer": round(d / len(r), 4) if r else None, "errors": d, "ref_words": len(r), "hyp_words": len(h)}
