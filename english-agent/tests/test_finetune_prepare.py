"""finetune/prepare.py 시험: moshi-finetune 데이터 형식 (스테레오 L=튜터 R=학생, 정렬 json, jsonl). GPU 를 쓰지 않는다."""

import json
import wave

import numpy as np

from finetune import prepare as fp
from gateway.recorder import Recorder

RATE = 24000


def write_mono(p, x, rate=RATE):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def tone(sec, rate=RATE, f=220):
    t = np.arange(int(sec * rate)) / rate
    return (0.2 * np.sin(2 * np.pi * f * t)).astype(np.float32)


WORDS = [{"word": "Great", "start": 0.1, "end": 0.4}, {"word": "job!", "start": 0.45, "end": 0.8},
         {"word": "", "start": 1.0, "end": 1.1}, {"word": "bad", "start": 1.5, "end": 1.5},
         {"word": "late", "start": 9.0, "end": 9.5}]


def test_alignments_from_words():
    al = fp.alignments_from_words(WORDS, duration=2.0)
    assert al == [["Great", [0.1, 0.4], "SPEAKER_MAIN"], ["job!", [0.45, 0.8], "SPEAKER_MAIN"]]


def test_from_pairs_and_check(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    write_mono(src / "Kim_tutor.wav", tone(2.0))
    write_mono(src / "Kim_student.wav", tone(1.5, rate=16000, f=330), rate=16000)  # 다른 샘플레이트, 다른 길이
    (src / "w.json").write_text(json.dumps(WORDS))
    rows = [{"id": "Kim", "tutor_wav": "Kim_tutor.wav", "student_wav": "Kim_student.wav", "tutor_words": "w.json",
             "consent": True, "adult": True},
            {"id": "child", "tutor_wav": "Kim_tutor.wav", "student_wav": "Kim_student.wav", "tutor_words": "w.json",
             "consent": True, "adult": False},
            {"id": "nowords", "tutor_wav": "Kim_tutor.wav", "student_wav": "Kim_student.wav", "consent": True, "adult": True}]
    (src / "m.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "data"
    res = fp.from_pairs(src / "m.jsonl", out)
    assert res["items"] == 1 and res["words"] == 2
    assert sorted(s["source"] for s in res["skipped"]) == ["child", "nowords"]

    line = json.loads((out / "pairs.jsonl").read_text().splitlines()[0])
    assert set(line) == {"path", "duration"} and line["duration"] == 2.0
    assert "Kim" not in line["path"]  # 원래 파일 이름이 남지 않음
    with wave.open(str(out / line["path"])) as w:
        assert w.getnchannels() == 2 and w.getframerate() == RATE
        st = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, 2)
    assert np.abs(st[: RATE, 0]).max() > 1000 and np.abs(st[: RATE, 1]).max() > 1000
    assert np.abs(st[int(1.6 * RATE):, 1]).max() == 0  # 학생 음성(1.5초) 뒤는 무음으로 채움
    al = json.loads((out / line["path"]).with_suffix(".json").read_text(encoding="ascii"))["alignments"]
    assert al[0] == ["Great", [0.1, 0.4], "SPEAKER_MAIN"]
    assert json.loads((out / "pairs_sources.json").read_text())[0]["source"] == "Kim"
    assert fp.check(out) == {"items": 1, "ok": True, "problems": []}


def test_check_finds_problems(tmp_path):
    out = tmp_path / "data"
    (out / "data_stereo").mkdir(parents=True)
    write_mono(out / "data_stereo" / "a.wav", tone(1.0))  # 모노
    (out / "data_stereo" / "a.json").write_text(json.dumps({"alignments": [["x", [0.5, 0.6], "SPEAKER_MAIN"],
                                                                           ["y", [0.1, 0.2], "SPEAKER_OTHER"]]}))
    (out / "x.jsonl").write_text(json.dumps({"path": "data_stereo/a.wav", "duration": 3.0}) + "\n")
    res = fp.check(out)
    assert not res["ok"]
    text = " ".join(res["problems"])
    for w in ("채널 1", "duration", "시간 순서", "SPEAKER_MAIN 이외"):
        assert w in text, text


def make_queue_session(root, monkeypatch, meta, words=True):
    import gateway.recorder as rmod
    clock = [10.0]
    monkeypatch.setattr(rmod.time, "monotonic", lambda: clock[0])
    rec = Recorder(root, RATE)
    rec.user(tone(1.0))
    clock[0] += 1.5
    rec.agent(tone(0.5, f=440))
    rec.close(meta)
    tr = {"user": {"text": "hi"}, "agent": {"text": "Great job!"}}
    if words:
        tr["agent"]["words"] = WORDS[:2]
    (rec.dir / "transcript.json").write_text(json.dumps(tr))
    return rec.dir


def test_from_queue_only_consented_candidates(tmp_path, monkeypatch):
    q = tmp_path / "queue"
    ok = make_queue_session(q, monkeypatch, {"status": "transcribed", "finetune": "candidate", "consent": {"finetune": True}})
    make_queue_session(q, monkeypatch, {"status": "transcribed", "finetune": "needs_review", "consent": {"finetune": True}})
    make_queue_session(q, monkeypatch, {"status": "transcribed", "finetune": "candidate"})  # 동의 없음
    nw = make_queue_session(q, monkeypatch, {"status": "transcribed", "finetune": "candidate", "consent": {"finetune": True}},
                            words=False)
    res = fp.from_queue(q, tmp_path / "data")
    assert res["items"] == 1
    reasons = {s["source"]: s["reason"] for s in res["skipped"]}
    assert "타임스탬프" in reasons[nw.name] and len(reasons) == 2
    line = json.loads((tmp_path / "data" / "queue.jsonl").read_text())
    assert line["path"] == f"data_stereo/q_{ok.name}.wav"
    with wave.open(str(tmp_path / "data" / line["path"])) as w:
        st = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, 2)
    # 왼쪽 = Agent(튜터): 1.0~1.5초에만 소리, 오른쪽 = 사용자: 0~1.0초
    assert np.abs(st[: int(0.9 * RATE), 0]).max() == 0 and np.abs(st[int(1.05 * RATE): int(1.45 * RATE), 0]).max() > 1000
    assert np.abs(st[: int(0.9 * RATE), 1]).max() > 1000
    assert fp.check(tmp_path / "data")["ok"]


def test_cli_check_empty(tmp_path, capsys):
    assert fp.main(["check", "--out", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"items": 0, "ok": True, "problems": []}
