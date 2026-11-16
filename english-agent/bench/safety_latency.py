"""Safety Monitor 토큰당 CPU 처리 시간 측정 (프롬프트 2.2절 3번).

실제 Moshi 토큰이 아직 없으므로, 테스트 문장을 Moshi 와 비슷한 크기의 조각(공백 앞붙임 단어, 긴 단어는 3~5글자로 분할)으로 나눠 넣는다.
Moshi 텍스트 토큰은 12.5Hz(80ms 간격) 이하로 나오므로, 토큰당 처리 시간이 80ms 보다 충분히 작으면 실시간 검사가 가능하다.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.safety_monitor import SafetyMonitor  # noqa: E402

TEXT = (
    "Hi there! Today we are going to practice ordering food at a restaurant. Imagine you are hungry and you walk "
    "into a small cafe. The waiter says: good afternoon, what would you like to drink? You can answer: I would like "
    "a glass of orange juice, please. Great job! Now let's try another sentence about your favorite weekend activity, "
    "and remember to use the past tense when you talk about what you did yesterday. "
)


def pieces(text: str, rng: random.Random):
    for w in text.split(" "):
        tok = " " + w
        while len(tok) > 6:
            k = rng.randint(3, 5)
            yield tok[:k]
            tok = tok[k:]
        yield tok


def main(turns: int = 200, out: str | None = None) -> dict:
    rng = random.Random(0)
    mon = SafetyMonitor()
    lat = []
    for _ in range(turns):
        mon.reset()
        for tok in pieces(TEXT * 2, rng):
            t0 = time.perf_counter()
            mon.feed(tok)
            lat.append((time.perf_counter() - t0) * 1e3)
        mon.flush()
    lat.sort()
    q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]  # noqa: E731
    res = {"tokens": len(lat), "rules": len(mon.rules.rules), "window_chars": mon.window,
           "ms_mean": round(statistics.mean(lat), 4), "ms_p50": round(q(0.5), 4), "ms_p95": round(q(0.95), 4),
           "ms_p99": round(q(0.99), 4), "ms_max": round(lat[-1], 4), "budget_ms_per_token": 80}
    print(json.dumps(res, indent=1))
    if out:
        Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    main(out=sys.argv[1] if len(sys.argv) > 1 else None)
