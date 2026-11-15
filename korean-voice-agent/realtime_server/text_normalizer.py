"""TTS 직전에 숫자와 기호를 소리 내어 읽는 형태로 바꾼다 (프롬프트 3절, 5단계).

시스템 프롬프트가 이미 "세 개", "열 시 삼십 분"처럼 쓰라고 지시하지만, LLM 이 지키지 않을 때를 대비한다.
1단계 측정에서 두 TTS 모두 "3개", "10:30" 을 잘못 읽었다 (docs/reports/stage1_report.md 2.2절).

규칙
- 고유어 단위명사(개, 명, 살, 시, 마리 ...) 앞의 1~99 는 고유어 관형형으로 읽는다: 3개 -> 세 개, 20살 -> 스무 살.
- 한자어 단위(분, 원, 년, 월, 일, %, km ...) 앞이나 단위가 없는 숫자는 한자어로 읽는다: 30분 -> 삼십 분.
- 시각 HH:MM -> 열 시 삼십 분. 6월 -> 유월, 10월 -> 시월. 1번째 -> 첫 번째.
- 소수 3.5 -> 삼 점 오, 천 단위 쉼표 1,000 -> 천.
- 수식 기호: + 더하기, - 빼기(숫자 사이), x/× 곱하기, ÷ 나누기, = 은/는.
- 마크다운 기호, 이모지, 괄호 문자는 지운다.
"""

from __future__ import annotations

import re

_SINO_DIGITS = "영일이삼사오육칠팔구"
_SMALL_UNITS = ["", "십", "백", "천"]
_BIG_UNITS = ["", "만", "억", "조"]

_NATIVE_ONES_ATTR = ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]
_NATIVE_TENS = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]

# 고유어로 세는 단위명사. 긴 것부터 맞도록 정렬해서 쓴다.
NATIVE_COUNTERS = [
    "번째", "시간", "군데", "가지", "켤레", "송이", "그루", "마리", "개", "명", "살", "시", "권", "잔",
    "장", "번", "달", "줄", "통", "벌", "대", "척", "채", "곳", "판", "알", "쌍", "사람",
]
# 한자어로 세는 단위. 기호 단위는 읽는 말로 바꿔서 붙인다.
SINO_COUNTERS = {
    "퍼센트": "퍼센트", "%": "퍼센트", "킬로미터": "킬로미터", "km": "킬로미터", "센티미터": "센티미터",
    "cm": "센티미터", "밀리미터": "밀리미터", "mm": "밀리미터", "킬로그램": "킬로그램", "kg": "킬로그램",
    "미터": "미터", "m": "미터", "그램": "그램", "g": "그램", "리터": "리터", "L": "리터", "개월": "개월",
    "학년": "학년", "페이지": "페이지", "분": "분", "초": "초", "원": "원", "년": "년", "월": "월", "일": "일",
    "층": "층", "점": "점", "도": "도", "회": "회", "호": "호", "쪽": "쪽", "주": "주", "등": "등", "위": "위",
    "차": "차", "반": "반", "단계": "단계", "과": "과",
}


def sino(n: int) -> str:
    """한자어 수: 1234 -> 천이백삼십사, 10000 -> 만."""
    if n == 0:
        return "영"
    if n < 0:
        return "마이너스 " + sino(-n)
    out = []
    group_idx = 0
    while n > 0:
        group = n % 10000
        n //= 10000
        if group:
            s = ""
            for pos in range(3, -1, -1):
                d = (group // 10 ** pos) % 10
                if d == 0:
                    continue
                if d == 1 and pos > 0:
                    s += _SMALL_UNITS[pos]
                else:
                    s += _SINO_DIGITS[d] + _SMALL_UNITS[pos]
            if group_idx == 1 and group == 1:
                s = ""  # 일만 -> 만
            out.append(s + _BIG_UNITS[group_idx])
        group_idx += 1
    return "".join(reversed(out))


def native_attr(n: int) -> str:
    """고유어 관형형 (1~99). 그 밖의 수는 한자어로 읽는다."""
    if not 1 <= n <= 99:
        return sino(n)
    tens, ones = divmod(n, 10)
    if n == 20:
        return "스무"
    return _NATIVE_TENS[tens] + _NATIVE_ONES_ATTR[ones]


def _has_batchim(word: str) -> bool:
    if not word:
        return False
    ch = word[-1]
    if "가" <= ch <= "힣":
        return (ord(ch) - 0xAC00) % 28 != 0
    return False


def _to_int(s: str) -> int:
    return int(s.replace(",", ""))


_NUM = r"\d{1,3}(?:,\d{3})+|\d+"
_native_alt = "|".join(sorted(NATIVE_COUNTERS, key=len, reverse=True))
_sino_alt = "|".join(re.escape(k) for k in sorted(SINO_COUNTERS, key=len, reverse=True))

_RE_TIME = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])")
_RE_DECIMAL = re.compile(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])")
# (?!월): "3개월" 은 한자어 단위 "개월" 이므로 고유어 "개" 로 보지 않는다.
_RE_RANGE_NATIVE = re.compile(rf"({_NUM})\s*[~∼-]\s*({_NUM})\s*({_native_alt})(?!월)")
_RE_NATIVE = re.compile(rf"({_NUM})\s*({_native_alt})(?!월)")
# 한자어 단위: 영문 기호 단위(m, g, L ...) 뒤에 다른 영문자가 붙으면 단위로 보지 않는다.
_RE_SINO = re.compile(rf"({_NUM})\s*({_sino_alt})(?![A-Za-z])")
_RE_MATH = re.compile(rf"({_NUM})\s*([+\-×x*÷/=])\s*(?=\d)")
_RE_EQUALS = re.compile(r"([가-힣])\s*=\s*")
_RE_BARE = re.compile(rf"{_NUM}")
_RE_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF️‍]+", flags=re.UNICODE
)
_RE_PAREN = re.compile(r"\s*[(\[][^()\[\]]*[)\]]")
_RE_MARKDOWN = re.compile(r"[*_#`>|\[\]{}()<>]")

_MATH_WORDS = {"+": "더하기", "-": "빼기", "×": "곱하기", "x": "곱하기", "*": "곱하기", "÷": "나누기", "/": "나누기"}


def _native_counter(num: str, counter: str) -> str:
    n = _to_int(num)
    if counter == "번째" and n == 1:
        return "첫 번째"
    if counter == "시" and n > 24:  # 시각이 아니면 한자어 (예: 30시 -> 삼십 시)
        return f"{sino(n)} {counter}"
    return f"{native_attr(n)} {counter}"


def _sino_counter(num: str, unit: str) -> str:
    n = _to_int(num)
    word = SINO_COUNTERS[unit]
    if word == "월":
        if n == 6:
            return "유월"
        if n == 10:
            return "시월"
    return f"{sino(n)} {word}"


def _time(m: re.Match) -> str:
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 24 or mi > 59:
        return m.group(0)
    hour = f"{native_attr(h) if h else '영'} 시"
    return hour if mi == 0 else f"{hour} {sino(mi)} 분"


def normalize(text: str) -> str:
    """소리 내어 읽을 문장으로 바꾼다. 이미 읽는 형태인 문장은 그대로 둔다."""
    if not text:
        return text
    s = _RE_EMOJI.sub("", text)
    s = _RE_PAREN.sub("", s)  # 괄호 속 보충 설명은 읽지 않는다 (시스템 프롬프트도 괄호를 금지함)
    s = _RE_MARKDOWN.sub("", s)
    s = _RE_TIME.sub(_time, s)
    s = _RE_DECIMAL.sub(lambda m: f"{sino(int(m.group(1)))} 점 {' '.join(_SINO_DIGITS[int(d)] for d in m.group(2))}", s)
    s = _RE_RANGE_NATIVE.sub(
        lambda m: f"{_native_counter(m.group(1), m.group(3))}에서 {_native_counter(m.group(2), m.group(3))}", s)
    s = _RE_NATIVE.sub(lambda m: _native_counter(m.group(1), m.group(2)), s)
    s = _RE_SINO.sub(lambda m: _sino_counter(m.group(1), m.group(2)), s)

    def _math(m: re.Match) -> str:
        left, op = sino(_to_int(m.group(1))), m.group(2)
        if op == "=":
            return f"{left}{'은' if _has_batchim(left) else '는'} "
        return f"{left} {_MATH_WORDS[op]} "

    s = _RE_MATH.sub(_math, s)
    s = _RE_EQUALS.sub(lambda m: f"{m.group(1)}{'은' if _has_batchim(m.group(1)) else '는'} ", s)
    s = _RE_BARE.sub(lambda m: sino(_to_int(m.group(0))), s)
    s = s.replace("&", " 그리고 ").replace("~", "")
    return re.sub(r"[ \t]{2,}", " ", s).strip()
