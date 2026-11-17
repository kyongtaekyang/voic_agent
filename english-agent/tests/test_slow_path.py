"""영어 Slow Path 시험: 실제 gateway.recorder.Recorder 로 세션을 만들고, Transcribe 30B 대신 가짜 HTTP 서버를 쓴다. GPU 를 쓰지 않는다."""

import base64
import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

import gateway.recorder as rmod
from gateway.recorder import Recorder
from slow_path import run as sp
from slow_path.wer import wer, words

RATE = 24000


def tone(sec, amp=0.2):
    t = np.arange(int(sec * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(sec):
    return np.zeros(int(sec * RATE), np.float32)


def test_words_and_wer():
    assert words("Don't stop, OK?  <en-US> 'quoted' it's") == ["don't", "stop", "ok", "quoted", "it's"]
    assert wer("Hello there, friend.", "hello there friend")["wer"] == 0.0
    r = wer("I like big red dogs", "I like red dog")
    assert r["errors"] == 2 and r["ref_words"] == 5 and r["wer"] == 0.4
    assert wer("", "abc")["wer"] is None


def test_segments_and_latency():
    user = np.concatenate([tone(1.0), silence(0.3), tone(0.5), silence(2.0)])  # 0.3초 침묵은 이어 붙여 한 발화
    agent = np.concatenate([silence(2.2), tone(0.8), silence(0.5)])
    us = sp.speech_segments(user, RATE)
    ag = sp.speech_segments(agent, RATE)
    assert len(us) == 1 and us[0][0] == pytest.approx(0.0, abs=0.03) and us[0][1] == pytest.approx(1.8, abs=0.03)
    assert len(ag) == 1 and ag[0][0] == pytest.approx(2.2, abs=0.03)
    assert sp.response_latency(us, ag) == [pytest.approx(400, abs=40)]
    # Agent 가 말하는 도중에 사용자 발화가 끝나면(겹침) 응답 지연으로 세지 않는다
    assert sp.response_latency([(0.0, 1.0)], [(0.5, 2.0)]) == []
    assert sp.response_latency([(0.0, 1.0)], [(7.0, 8.0)]) == []  # max_gap_s 초과


def make_session(root, monkeypatch, *, block=True):
    clock = [100.0]
    monkeypatch.setattr(rmod.time, "monotonic", lambda: clock[0])
    rec = Recorder(root, RATE)
    rec.event({"type": "eva.session.started", "sample_rate": RATE})
    rec.user(np.concatenate([tone(1.0), silence(2.0)]))  # 0~1초 사용자 발화
    clock[0] += 1.9
    rec.agent(tone(0.5))  # 1.4~1.9초 Agent 응답 → 지연 약 400ms
    rec.event({"type": "eva.user_text", "text": "Hello can you help me", "final": True, "lang": "en"})
    rec.event({"type": "eva.user_text", "text": "공룡 좋아요", "final": True, "lang": "ko"})
    rec.event({"type": "eva.language_mismatch", "detected_language": "ko", "turns": 2})
    for t in ["Sure", ",", " I", " can", "."]:
        rec.event({"type": "eva.agent_text", "delta": t})
    if block:
        rec.event({"type": "eva.safety_block", "rule": "insult_words", "category": "insult", "token_index": 9,
                   "dropped_ms": 200, "already_played_ms": 600, "redirect_text": "Let's keep it kind."})
    clock[0] += 1.1
    rec.event({"type": "eva.session.ended", "duration_s": 3.0, "user_s": 3.0, "agent_s": 0.5, "text_tokens": 10,
               "safety_blocks": int(block)})
    rec.close()
    return rec.dir


class _Mock(BaseHTTPRequestHandler):
    requests: list = []
    replies: list = []
    status = 200
    finish = "stop"

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Mock.requests.append(body)
        if _Mock.status != 200:
            self.send_response(_Mock.status)
            self.end_headers()
            return
        out = json.dumps({"choices": [{"message": {"content": _Mock.replies.pop(0)}, "finish_reason": _Mock.finish}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def mock_url():
    _Mock.requests, _Mock.replies, _Mock.status, _Mock.finish = [], [], 200, "stop"
    srv = HTTPServer(("127.0.0.1", 0), _Mock)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


CFG = {"transcribe_model": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B", "audit_agent_audio": True, "vad": {}}


def test_transcribe_metrics_and_report(tmp_path, monkeypatch, mock_url):
    d = make_session(tmp_path, monkeypatch)
    _Mock.replies = ["Hello, can you help me? 공룡 좋아요.", "Sure, I can. You are so stupid."]
    res = sp.transcribe_stage(tmp_path, CFG, mock_url, None)
    assert res == {"stage": "transcribe", "done": 1, "failed": 0}

    # 요청: 사용자 → Agent 순서, 16kHz wav, Thinking 끔, temperature 0
    assert len(_Mock.requests) == 2
    for body in _Mock.requests:
        assert body["temperature"] == 0.0 and body["chat_template_kwargs"] == {"enable_thinking": False}
        url = body["messages"][0]["content"][1]["audio_url"]["url"]
        with wave.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as w:
            assert w.getframerate() == 16000 and w.getnchannels() == 1

    meta = json.loads((d / "meta.json").read_text())
    assert meta["status"] == "transcribed" and meta["history"][0]["stage"] == "transcribe"
    m = json.loads((d / "metrics.json").read_text())
    assert m["asr_turns"] == 2 and m["non_english_turns"] == 1 and m["language_mismatch"] == 1
    assert m["asr_wer"]["wer"] == 0.0  # 문장부호만 다름
    assert m["safety"]["blocks"] == 1 and m["safety"]["by_category"] == {"insult": 1}
    assert m["safety"]["already_played_ms"]["max"] == 600 and m["safety"]["dropped_ms_total"] == 200
    audit = m["safety"]["audit"]
    assert audit["violations_in_agent_audio"] >= 1 and any(i["category"] == "insult" for i in audit["items"])
    assert m["agent_caption_vs_audio_wer"]["ref_words"] == 7  # 차단된 문장은 자막에 없어서 차이가 남
    lat = m["response_latency_ms_approx"]
    assert lat["n"] == 1 and lat["p50"] == pytest.approx(400, abs=60)

    k = sp.report(tmp_path)
    assert k["sessions"] == 1 and k["asr_wer_pooled"] == 0.0 and k["safety_blocks"] == 1
    assert k["audit_violations_in_agent_audio"] >= 1
    assert json.loads((tmp_path / "_kpi.json").read_text())["sessions"] == 1


def test_audit_off_sends_only_user_audio(tmp_path, monkeypatch, mock_url):
    make_session(tmp_path, monkeypatch, block=False)
    _Mock.replies = ["Hello can you help me 공룡 좋아요"]
    assert sp.transcribe_stage(tmp_path, CFG | {"audit_agent_audio": False}, mock_url, None)["done"] == 1
    assert len(_Mock.requests) == 1
    m = json.loads(next(tmp_path.glob("*/metrics.json")).read_text())
    assert "audit" not in m["safety"] and m["safety"]["blocks"] == 0


def test_audit_long_unpunctuated_transcript():
    """문장부호 없는 긴 전사(축어 전사에서 흔함): 앞부분의 위반도 찾아야 한다 (SafetyMonitor window 240자보다 긺)."""
    from core.safety_monitor import SafetyMonitor
    text = "you are so stupid " + " ".join(["and then we went to the park"] * 25)
    assert len(text) > 600
    found = sp.safety_audit(text, SafetyMonitor())
    assert [f["category"] for f in found] == ["insult"]
    assert sp.safety_audit("Great job. Let's count to ten.", SafetyMonitor()) == []


def test_truncated_transcript_is_flagged(tmp_path, monkeypatch, mock_url):
    make_session(tmp_path, monkeypatch, block=False)
    _Mock.replies, _Mock.finish = ["Hello can you"], "length"
    assert sp.transcribe_stage(tmp_path, CFG | {"audit_agent_audio": False}, mock_url, None)["done"] == 1
    d = next(tmp_path.glob("*/transcript.json")).parent
    assert json.loads((d / "transcript.json").read_text())["user"]["truncated"] is True
    assert json.loads((d / "metrics.json").read_text())["transcript_truncated"] is True
    assert sp.report(tmp_path)["sessions_truncated"] == 1


def test_failure_keeps_queued(tmp_path, monkeypatch, mock_url):
    d = make_session(tmp_path, monkeypatch)
    _Mock.status = 500
    assert sp.transcribe_stage(tmp_path, CFG, mock_url, None) == {"stage": "transcribe", "done": 0, "failed": 1}
    meta = json.loads((d / "meta.json").read_text())
    assert meta["status"] == "queued" and meta["errors"][0]["stage"] == "transcribe"
    assert not (d / "metrics.json").exists()


def test_recording_in_progress_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(rmod.time, "monotonic", lambda: 1.0)
    rec = Recorder(tmp_path, RATE)  # close() 전: meta.json 없음
    assert list(sp.sessions(tmp_path)) == []
    rec.close()
    assert [m["status"] for _, m in sp.sessions(tmp_path)] == ["queued"]


def test_local_only():
    assert sp.local_only("http://127.0.0.1:8002/") == "http://127.0.0.1:8002"
    with pytest.raises(SystemExit):
        sp.local_only("http://10.0.0.5:8002")


def test_cli_status_and_report_on_empty_queue(tmp_path, capsys):
    assert sp.main(["status", "--queue", str(tmp_path / "none")]) == 0
    assert json.loads(capsys.readouterr().out)["sessions"] == {}
    assert sp.main(["report", "--queue", str(tmp_path / "none")]) == 0
    assert json.loads(capsys.readouterr().out)["sessions"] == 0
