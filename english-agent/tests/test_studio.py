"""튜터 녹음 스튜디오 시험 (studio/server.py): 동의 확인, 두 트랙 시간 맞춤, WebSocket 규약, 삭제(동의 철회), 학습 데이터 변환 기준.
GPU, 브라우저, 마이크를 쓰지 않는다. 시계는 가짜로 바꿔서 정렬을 확인한다."""

import asyncio
import json
import wave

import numpy as np
from aiohttp.test_utils import TestClient, TestServer

from finetune import prepare as fp
from studio.server import RATE, Studio, make_app, validate

ADULT = {"tutor_alias": "T01", "student_alias": "S01", "student_group": "adult",
         "consent_recording_tutor": True, "consent_recording_student": True,
         "consent_training_tutor": True, "consent_training_student": True}
CHILD = ADULT | {"student_group": "child", "age_band": "8-10", "guardian_consent": True, "guardian_consent_ref": "GC-1"}


def cfg(tmp_path):
    return {"port": 0, "queue_dir": str(tmp_path / "tutor"), "gap_fill_ms": 200, "finetune_out": str(tmp_path / "data"),
            "allow_child_data": False}


def frame(sec=0.08, amp=0.2):
    n = int(sec * RATE)
    return (amp * np.sin(2 * np.pi * 220 * np.arange(n) / RATE) * 32767).astype("<i2").tobytes()


def wav_len(p):
    with wave.open(str(p)) as w:
        return w.getnframes() / w.getframerate()


# 동의 ---------------------------------------------------------------------------------------------------------------

def test_validate_consent_rules():
    assert validate(ADULT)[1] == []
    assert validate(CHILD)[1] == []
    errs = validate(ADULT | {"consent_recording_student": False})[1]
    assert any("녹음 동의" in e for e in errs)
    errs = validate(CHILD | {"guardian_consent": False})[1]
    assert any("법정대리인" in e for e in errs)
    errs = validate(CHILD | {"guardian_consent_ref": ""})[1]
    assert any("법정대리인" in e for e in errs)
    assert any("연령대" in e for e in validate(CHILD | {"age_band": ""})[1])
    assert any("tutor_alias" in e for e in validate(ADULT | {"tutor_alias": "<script>"})[1])
    meta, _ = validate(ADULT | {"consent_training_student": "yes"})  # true 만 동의로 본다
    assert meta["consent"]["training_student"] is False


# 정렬 ---------------------------------------------------------------------------------------------------------------

def test_alignment_late_join_and_dropout(tmp_path):
    clock = [100.0]
    st = Studio(cfg(tmp_path), clock=lambda: clock[0])
    sid, errs = st.create(ADULT)
    assert not errs
    rm = st.room(sid)
    assert st.start(rm) == []
    for i in range(1, 38):  # 튜터: 0~2.96초, 80 ms 마다 빠짐없이
        clock[0] = 100.0 + i * 0.08
        st.feed(rm, "tutor", frame())
        # 학생: 1.0초에 들어오고(앞 1.0초 무음), 2.0~2.5초 사이 프레임이 빠짐
        t = i * 0.08
        if t >= 1.04 and not (2.0 <= t < 2.5):
            st.feed(rm, "student", frame())
    clock[0] = 103.0
    m = st.stop(rm)
    assert m["status"] == "recorded" and m["duration_s"] == 3.0
    tu, sd = m["tracks"]["tutor"], m["tracks"]["student"]
    assert tu["lead_pad_s"] == 0 and tu["gap_pad_s"] == 0 and abs(tu["drift_s"]) < 0.1
    assert abs(sd["lead_pad_s"] - 1.0) < 0.1 and abs(sd["gap_pad_s"] - 0.5) < 0.1 and abs(sd["drift_s"]) < 0.1
    d = tmp_path / "tutor" / sid
    assert wav_len(d / "tutor.wav") == wav_len(d / "student.wav") == 3.0
    with wave.open(str(d / "student.wav")) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    assert np.abs(x[: int(0.9 * RATE)]).max() == 0          # 늦게 들어온 앞부분은 무음
    assert np.abs(x[int(1.1 * RATE): int(1.9 * RATE)]).max() > 1000
    assert np.abs(x[int(2.05 * RATE): int(2.4 * RATE)]).max() == 0  # 끊긴 구간은 무음 (뒤 음성이 앞으로 당겨지지 않음)
    assert st.start(rm) != []  # 같은 세션을 다시 녹음하지 않는다


def test_start_refused_when_meta_lacks_consent(tmp_path):
    st = Studio(cfg(tmp_path))
    sid, _ = st.create(CHILD)
    meta = st.read_meta(sid)
    meta["consent"]["guardian"] = False  # 양식 이후 파일이 바뀐 경우도 서버가 막는다
    st.save_meta(sid, meta)
    errs = st.start(st.room(sid))
    assert errs and "법정대리인" in errs[0]
    assert not (tmp_path / "tutor" / sid / "tutor.wav").exists()


# HTTP·WebSocket ------------------------------------------------------------------------------------------------------

def run(coro_fn, tmp_path):
    async def main():
        app = make_app(cfg(tmp_path))
        async with TestClient(TestServer(app)) as client:
            return await coro_fn(client, app)
    return asyncio.run(main())


async def next_of(ws, kind, binary=False):
    while True:
        msg = await asyncio.wait_for(ws.receive(), 5)
        if binary and msg.type.name == "BINARY":
            return msg.data
        if msg.type.name == "TEXT":
            ev = json.loads(msg.data)
            if ev["type"] == kind:
                return ev


def test_http_create_refuses_child_without_guardian(tmp_path):
    async def body(client, app):
        r = await client.post("/api/sessions", json=CHILD | {"guardian_consent": False})
        assert r.status == 400 and any("법정대리인" in e for e in (await r.json())["errors"])
        r = await client.post("/api/sessions", json=CHILD)
        j = await r.json()
        assert r.status == 200 and j["links"]["tutor"].endswith("role=tutor")
        lst = await (await client.get("/api/sessions")).json()
        assert lst["allow_child_data"] is False and lst["sessions"][0]["student_group"] == "child"
        assert (await client.get("/")).status == 200 and (await client.get("/static/capture-worklet.js")).status == 200
        assert (await client.get("/api/sessions/../../etc")).status == 404
    run(body, tmp_path)


def test_ws_record_relay_and_roles(tmp_path):
    async def body(client, app):
        sid = (await (await client.post("/api/sessions", json=ADULT)).json())["id"]
        tutor = await client.ws_connect(f"/ws/{sid}/tutor?listen=1")
        student = await client.ws_connect(f"/ws/{sid}/student?listen=1")
        st = await next_of(tutor, "state")
        while st["connected"]["student"] == 0:
            st = await next_of(tutor, "state")
        await student.send_json({"type": "start"})  # 학생은 녹음을 시작할 수 없다
        assert "튜터" in (await next_of(student, "error"))["detail"]
        await tutor.send_json({"type": "start"})
        while not (await next_of(tutor, "state"))["recording"]:
            pass
        await tutor.send_bytes(frame())
        assert await next_of(student, None, binary=True) == frame()  # 상대 음성 전달
        await student.send_bytes(frame())
        await tutor.send_json({"type": "stop"})
        rec = await next_of(student, "recorded")
        assert rec["meta"]["status"] == "recorded" and rec["meta"]["training_consent"] is True
        r = await client.get(f"/api/sessions/{sid}/audio/tutor.wav")
        assert r.status == 200 and (await r.read())[:4] == b"RIFF"
        await tutor.close()
        await student.close()
    run(body, tmp_path)


def test_tutor_disconnect_ends_recording(tmp_path):
    async def body(client, app):
        sid = (await (await client.post("/api/sessions", json=ADULT)).json())["id"]
        tutor = await client.ws_connect(f"/ws/{sid}/tutor")
        await tutor.send_json({"type": "start"})
        while not (await next_of(tutor, "state"))["recording"]:
            pass
        await tutor.close()
        for _ in range(50):
            meta = await (await client.get(f"/api/sessions/{sid}")).json()
            if meta["status"] == "recorded":
                break
            await asyncio.sleep(0.05)
        assert meta["status"] == "recorded"
    run(body, tmp_path)


# 삭제·학습 데이터 기준 -------------------------------------------------------------------------------------------------

def recorded_session(st, clock, form):
    sid, _ = st.create(form)
    rm = st.room(sid)
    st.start(rm)
    for i in range(1, 26):
        clock[0] += 0.08
        st.feed(rm, "tutor", frame())
        st.feed(rm, "student", frame(amp=0.1))
    st.stop(rm)
    words = [{"word": "Hello", "start": 0.1, "end": 0.4}, {"word": "there", "start": 0.5, "end": 0.9}]
    (st.queue / sid / "tutor_words.json").write_text(json.dumps(words))
    return sid


def test_from_studio_gate_and_delete(tmp_path):
    clock = [10.0]
    st = Studio(cfg(tmp_path), clock=lambda: clock[0])
    adult = recorded_session(st, clock, ADULT)
    child = recorded_session(st, clock, CHILD)
    no_train = recorded_session(st, clock, ADULT | {"consent_training_student": False})
    out = tmp_path / "data"

    res = fp.from_studio(st.queue, out)  # allow_child_data: false
    assert res["items"] == 1
    why = {s["source"]: s["reason"] for s in res["skipped"]}
    assert "내부관리규정" in why[child] and "학습 활용 동의" in why[no_train]
    assert fp.check(out)["ok"]
    line = json.loads((out / "studio.jsonl").read_text())
    assert adult not in line["path"]  # 세션 이름 대신 해시

    res = fp.from_studio(st.queue, out, allow_child=True)  # 규정 검토가 끝나 설정을 켠 경우
    assert res["items"] == 2

    # 동의 철회: 세션과 파생 학습 데이터를 함께 지운다
    gone = st.delete(adult)
    assert gone["derived_removed"] == 1 and not (st.queue / adult).exists()
    rows = [json.loads(x) for x in (out / "studio.jsonl").read_text().splitlines()]
    srcs = json.loads((out / "studio_sources.json").read_text())
    assert len(rows) == 1 and [s["source"] for s in srcs] == [child]
    assert len(list((out / "data_stereo").glob("s_*.wav"))) == 1
    assert fp.check(out)["ok"]


def test_from_studio_rejects_drift(tmp_path):
    clock = [10.0]
    st = Studio(cfg(tmp_path), clock=lambda: clock[0])
    sid = recorded_session(st, clock, ADULT)
    meta = st.read_meta(sid)
    meta["tracks"]["student"]["drift_s"] = 0.9
    st.save_meta(sid, meta)
    res = fp.from_studio(st.queue, tmp_path / "data")
    assert res["items"] == 0 and "어긋남" in res["skipped"][0]["reason"]


def test_from_pairs_child_gate(tmp_path):
    assert fp.training_gate({"consent": True, "adult": True}, False) is None
    assert "법정대리인" in fp.training_gate({"consent": True, "child": True}, True)
    assert "내부관리규정" in fp.training_gate({"consent": True, "child": True, "guardian_consent": True}, False)
    assert fp.training_gate({"consent": True, "child": True, "guardian_consent": True}, True) is None
    assert "동의" in fp.training_gate({"consent": False, "adult": True}, True)
