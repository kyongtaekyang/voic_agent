"""영어 발화 감지 로직 (모델 없이 _transcribe 를 바꿔서 확인)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

stt = pytest.importorskip("realtime_server.stt_streaming")


class FakeHandler(stt.NemotronTransformersSTTHandler):
    def __init__(self, auto_text: str, mismatch_turns: int = 2):
        self.prompt_language = "ko-KR"
        self._detects_utterance_language = False
        self.mismatch_turns = mismatch_turns
        self.probe_min_samples = int(0.6 * stt.SAMPLE_RATE)
        self.probe_min_cps = 1.5
        self._non_korean_streak = 0
        self.auto_text = auto_text
        self.calls = []

    def _transcribe(self, audio, language=None):
        self.calls.append(language)
        return self.auto_text if language == "auto" else ""


def _out(text):
    return SimpleNamespace(text=text, language_code=None, turn_id="t1")


SPEECH = np.zeros(stt.SAMPLE_RATE, dtype=np.float32)  # 1초
SHORT = np.zeros(stt.SAMPLE_RATE // 4, dtype=np.float32)


@pytest.fixture
def events(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    monkeypatch.setattr(stt, "log_event", lambda kind, **f: path.open("a").write(json.dumps({"type": kind, **f}) + "\n"))
    return path


def _read(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def test_empty_korean_result_triggers_auto_probe_and_mismatch(events, monkeypatch):
    monkeypatch.setenv("KVA_LOG_TRANSCRIPTS", "1")
    h = FakeHandler("What is your favorite animal? <en-US>")
    h._check_language(_out(""), SPEECH)
    assert h.calls == ["auto"] and h._non_korean_streak == 1 and not _read(events)
    h._check_language(_out(""), SPEECH)
    ev = _read(events)
    assert ev[-1]["type"] == "language_mismatch" and ev[-1]["probed"] is True
    assert ev[-1]["language_code"] == "<en-US>" and ev[-1]["text"] == "What is your favorite animal?"


def test_probe_without_tag_uses_hangul_ratio(events, monkeypatch):
    monkeypatch.setenv("KVA_LOG_TRANSCRIPTS", "0")
    h = FakeHandler("Hello, can you help me", mismatch_turns=1)
    h._check_language(_out(""), SPEECH)
    ev = _read(events)[-1]
    assert ev["language_code"] is None
    assert ev["text"] is None and ev["chars"] > 0  # 기본값: 발화 원문을 남기지 않음


def test_korean_resets_streak(events):
    h = FakeHandler("I like soccer <en-US>")
    h._check_language(_out(""), SPEECH)
    h._check_language(_out("오늘은 공룡을 배워요."), SPEECH)
    assert h._non_korean_streak == 0
    h._check_language(_out(""), SPEECH)
    assert not _read(events)


def test_short_empty_segment_is_not_probed(events):
    h = FakeHandler("noise <en-US>")
    h._check_language(_out(""), SHORT)
    assert h.calls == [] and h._non_korean_streak == 0


def test_single_syllable_for_long_segment_is_probed(events):
    h = FakeHandler("I like playing soccer <en-US>", mismatch_turns=1)
    h._check_language(_out("이"), SPEECH * 1)  # 1초에 한 음절
    assert h.calls == ["auto"] and _read(events)[-1]["type"] == "language_mismatch"


def test_normal_korean_density_is_not_probed(events):
    h = FakeHandler("unused <en-US>")
    h._check_language(_out("오늘은 공룡을 배워요."), SPEECH)
    assert h.calls == []


def test_latin_text_uses_ratio_without_probe(events):
    h = FakeHandler("unused", mismatch_turns=1)
    h._check_language(_out("hello teacher"), SPEECH)
    assert h.calls == [] and _read(events)[-1]["probed"] is False


def test_probe_korean_tag_is_not_mismatch(events):
    h = FakeHandler("음 <ko-KR>", mismatch_turns=1)
    h._check_language(_out(""), SPEECH)
    assert h._non_korean_streak == 0 and not _read(events)
