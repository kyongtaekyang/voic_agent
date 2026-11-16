import asyncio

import numpy as np

from core.asr_monitor import LanguageMismatchDetector
from gateway.asr_worker import AsrWorker, resample_linear

RATE = 24000


def _speech(sec):
    t = np.arange(int(RATE * sec)) / RATE
    return (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def _silence(sec):
    return np.zeros(int(RATE * sec), np.float32)


def test_resample_length():
    assert resample_linear(np.zeros(24000, np.float32), 24000, 16000).size == 16000


def test_segments_and_language_mismatch():
    answers = iter(["<ko-KR> 안녕하세요", "<ko-KR> 이거 뭐예요", "<en-US> hello"])
    seen = []

    def fake(audio16):
        seen.append(audio16.size)
        return next(answers)

    async def main():
        events = []

        async def emit(ev):
            events.append(ev)

        w = AsrWorker(fake, LanguageMismatchDetector(mismatch_turns=2), korean_agent_url="http://k/")
        task = asyncio.create_task(w.run(emit))
        for _ in range(3):
            w.feed(_speech(0.6))
            w.feed(_silence(0.7))
        w.feed(_silence(0.1))  # 짧은 잡음 없음
        w.close()
        await asyncio.wait_for(task, 10)
        return events

    events = asyncio.run(main())
    user = [e for e in events if e["type"] == "eva.user_text"]
    assert [e["text"] for e in user] == ["안녕하세요", "이거 뭐예요", "hello"]
    assert user[0]["lang"] == "ko"
    mm = [e for e in events if e["type"] == "eva.language_mismatch"]
    assert len(mm) == 1 and mm[0]["korean_agent_url"] == "http://k/"
    assert all(n >= 16000 * 0.6 for n in seen)  # 16kHz 로 바꿔서 넘김


def test_short_noise_is_ignored():
    calls = []

    async def main():
        w = AsrWorker(lambda a: calls.append(a) or "x", min_speech_ms=300)

        async def emit(ev):
            pass

        task = asyncio.create_task(w.run(emit))
        w.feed(_speech(0.1))
        w.feed(_silence(1.0))
        w.close()
        await asyncio.wait_for(task, 5)

    asyncio.run(main())
    assert calls == []
