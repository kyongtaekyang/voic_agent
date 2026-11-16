import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from slow_path import run as sp
from slow_path.cer import cer, normalize
from web.recorder import SessionRecorder


def test_normalize_and_cer():
    assert normalize("안녕, 하세요! <ko-KR>") == "안녕하세요"
    assert cer("오늘은 공룡을 배워요", "오늘은 공룡을 배워요.")["cer"] == 0.0
    r = cer("가나다라", "가나라")
    assert r["errors"] == 1 and r["cer"] == 0.25
    assert cer("", "abc")["cer"] is None


def _pcm_b64(n=1600):
    return base64.b64encode(b"\x01\x00" * n).decode()


def _record_session(root):
    rec = SessionRecorder(root)
    rec.server_event({"type": "session.created"})
    rec.client_event({"type": "input_audio_buffer.append", "audio": _pcm_b64()})
    rec.server_event({"type": "input_audio_buffer.speech_started", "audio_start_ms": 100, "item_id": "u1"})
    rec.server_event({"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 1900, "item_id": "u1"})
    rec.server_event({"type": "conversation.item.input_audio_transcription.completed", "item_id": "u1",
                      "transcript": "공룡에 대해 배우고 싶어요"})
    rec.server_event({"type": "response.created"})
    rec.server_event({"type": "response.output_audio_transcript.delta", "delta": "좋아요! "})
    rec.server_event({"type": "response.output_audio.delta", "delta": _pcm_b64(3200)})
    rec.server_event({"type": "response.output_audio_transcript.delta", "delta": "공룡 이야기를 해 볼까요?"})
    rec.server_event({"type": "response.done"})
    rec.client_event({"type": "conversation.item.truncate", "item_id": "a1", "audio_end_ms": 900})
    rec.server_event({"type": "kva.output_guard", "rule": "x"})
    rec.server_event({"type": "kva.status", "mode": "realtime"})  # 남기지 않음
    return rec.close()


def test_recorder_files(tmp_path):
    d = _record_session(tmp_path)
    meta = json.loads((d / "meta.json").read_text())
    assert meta["status"] == "queued" and meta["user_s"] == 0.1 and meta["agent_s"] >= 0.2
    ev = [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines()]
    types = [e["type"] for e in ev]
    assert "kva.status" not in types and "rec.agent_audio_start" in types
    done = [e for e in ev if e["type"] == "response.done"][0]
    assert done["text"] == "좋아요! 공룡 이야기를 해 볼까요?"
    assert all("delta" not in e and "audio" not in e for e in ev)  # 음성 데이터는 events 에 넣지 않음


class _Mock(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Mock.requests.append(body)
        if body["model"].startswith("nvidia/"):
            content = "공룡에 대해서 배우고 싶어요"
        else:
            content = 'x {"summary": "공룡 이야기", "topics": ["공룡"], "strengths": [], "suggestions": [], "concerns": []} y'
        out = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def mock_server():
    srv = HTTPServer(("127.0.0.1", 0), _Mock)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    _Mock.requests = []
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_batch_transcribe_then_summarize(tmp_path, mock_server):
    d = _record_session(tmp_path)
    cfg = sp.load_cfg()
    assert sp.run_stage("transcribe", tmp_path, cfg, mock_server, None) == {"stage": "transcribe", "done": 1, "failed": 0}
    req = _Mock.requests[0]
    assert req["temperature"] == 0.0 and req["chat_template_kwargs"] == {"enable_thinking": False}
    assert [m["role"] for m in req["messages"]] == ["user"]  # 시스템 프롬프트 없음
    parts = req["messages"][0]["content"]
    assert parts[1]["audio_url"]["url"].startswith("data:audio/wav;base64,")
    m = json.loads((d / "metrics.json").read_text())
    assert m["turns"] == 1 and m["user_speech_s"] == 1.8 and m["barge_ins"] == 1
    assert m["realtime_stt_cer"]["cer"] is not None and m["response_latency_ms"]["n"] == 1
    assert m["kva_events"] == {"kva.output_guard": 1}
    assert json.loads((d / "meta.json").read_text())["status"] == "transcribed"

    assert sp.run_stage("summarize", tmp_path, cfg, mock_server, None)["done"] == 1
    s = json.loads((d / "summary.json").read_text())
    assert s["summary"]["topics"] == ["공룡"]
    req = _Mock.requests[1]
    assert req["messages"][0]["role"] == "system" and "선생님: 좋아요!" in req["messages"][1]["content"]
    assert json.loads((d / "meta.json").read_text())["status"] == "summarized"


def test_failure_keeps_status(tmp_path):
    d = _record_session(tmp_path)
    res = sp.run_stage("transcribe", tmp_path, sp.load_cfg(), "http://127.0.0.1:9", None)
    assert res["failed"] == 1
    meta = json.loads((d / "meta.json").read_text())
    assert meta["status"] == "queued" and meta["errors"][0]["stage"] == "transcribe"


def test_local_only():
    assert sp.local_only("http://127.0.0.1:8002/") == "http://127.0.0.1:8002"
    with pytest.raises(SystemExit):
        sp.local_only("http://example.com:8002")
