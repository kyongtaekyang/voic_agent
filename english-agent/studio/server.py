"""사람 튜터 녹음 스튜디오 (프롬프트 3절: 튜터 스타일 LoRA 학습 데이터). GPU 를 쓰지 않으며 운영 모드(gpu-mode)와 무관하게 띄운다.

  CUDA_VISIBLE_DEVICES= .venv-core/bin/python -m studio.server      # http://127.0.0.1:7103 (SSH 터널로 접속)

- 브라우저에서 튜터와 학생 음성을 **따로** 녹음한다. moshi-finetune 은 스테레오(왼쪽 튜터, 오른쪽 학생)로 채널이 나뉜 음성이 필요하기 때문이다.
  - 기기 2대: 튜터와 학생이 각자 브라우저로 같은 녹음 세션에 들어온다 (역할마다 WebSocket 하나). 상대 음성을 들려줄 수 있다 (헤드셋 권장).
  - 기기 1대: 한 화면에서 마이크 두 개를 골라 같은 규약으로 소켓 두 개를 연다.
  - 순차 녹음(turns): 마이크 하나, 소켓 하나(/ws/<세션>/turns). 화면의 "튜터 완료"·"학생 완료" 버튼으로 말하는 차례를 바꾼다.
    서버는 받은 프레임을 지금 차례인 트랙에 쓰고 다른 트랙에는 같은 길이의 무음을 쓴다 (두 트랙이 같은 시간축을 유지).
    겹쳐 말하기(끼어들기)는 기록되지 않는다.
- 두 트랙은 서버의 같은 기준 시각(t0)에 맞춘다. 늦게 들어온 쪽은 앞을 무음으로 채우고, 연결이 끊겼다가 다시 들어오면 끊긴 동안을 무음으로 채운다.
  한 연결 안에서는 받은 프레임을 이어 쓴다 (TCP 라서 프레임은 늦게 올 뿐 없어지지 않음, Studio.feed 참고).
  채운 길이(lead_pad_s, gap_pad_s)와 기기 시계 차이(drift_s)를 meta.json 에 남긴다. finetune/prepare.py 가 이 값으로 학습 데이터 여부를 정한다.
- 동의는 서버가 확인한다 (화면 검사만 믿지 않음). 녹음 동의가 없거나, 아동인데 법정대리인 동의·동의서 번호가 없으면 세션을 만들지 않고 녹음 시작도 거부한다.
  학습 활용 동의는 녹음 동의와 따로 받는다. 아동 녹음은 저장되지만, 내부관리규정 검토가 끝나 config finetune.allow_child_data 를 켜기 전에는
  학습 데이터로 바뀌지 않는다 (프롬프트 3절).
- 저장 위치: queue/tutor/<세션>/{tutor.wav, student.wav, meta.json}. 저장소에 올리지 않고 서버 밖으로 보내지 않는다.
  삭제(동의 철회)는 세션 폴더와 finetune/data 의 파생 데이터를 함께 지운다.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import time
import uuid
import wave
from pathlib import Path

import yaml
from aiohttp import WSMsgType, web

from finetune.prepare import remove_derived

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
WORKLET = ROOT / "web" / "static" / "capture-worklet.js"  # 영어 Agent 화면과 같은 마이크 캡처 (24 kHz PCM16)
RATE = 24000
ROLES = ("tutor", "student")
TURNS = "turns"  # 순차 녹음 소켓 (마이크 하나를 차례대로 튜터·학생 트랙에 씀)
MODES = ("turns", "one_device", "two_devices")
GROUPS = ("adult", "child")
AGE_BANDS = ("5-7", "8-10", "11-13", "14-17")
log = logging.getLogger("eva.studio")


def load_cfg() -> dict:
    p = Path(os.environ.get("EVA_CONFIG") or ROOT / "config" / "services.yaml")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    st = dict(cfg.get("studio") or {})
    st.setdefault("port", 7103)
    st.setdefault("queue_dir", "queue/tutor")
    st.setdefault("gap_fill_ms", 200)
    st.setdefault("finetune_out", "finetune/data")
    st["allow_child_data"] = (cfg.get("finetune") or {}).get("allow_child_data", False) is True  # 화면 안내용
    return st


# 동의 확인 --------------------------------------------------------------------------------------------------------

def _b(v) -> bool:
    return v is True


def validate(form: dict) -> tuple[dict, list[str]]:
    """세션 생성 양식을 검사하고 meta 를 만든다. 문제가 있으면 (meta, [이유]) 의 이유 목록이 비어 있지 않다."""
    errs = []
    alias = {r: str(form.get(f"{r}_alias", "")).strip()[:40] for r in ROLES}
    for r, a in alias.items():
        if not a:
            errs.append(f"{r}_alias: 가명을 입력하세요")
        elif not re.fullmatch(r"[\w\- .]+", a):
            errs.append(f"{r}_alias: 글자, 숫자, -, _, 공백만 쓸 수 있습니다")
    group = form.get("student_group")
    if group not in GROUPS:
        errs.append("student_group: adult 또는 child")
    age_band = form.get("age_band") or None
    if group == "child" and age_band not in AGE_BANDS:
        errs.append(f"age_band: 아동은 연령대가 필요합니다 {AGE_BANDS}")
    consent = {
        "recording_tutor": _b(form.get("consent_recording_tutor")),
        "recording_student": _b(form.get("consent_recording_student")),
        "training_tutor": _b(form.get("consent_training_tutor")),
        "training_student": _b(form.get("consent_training_student")),
        "guardian": _b(form.get("guardian_consent")),
        "guardian_ref": str(form.get("guardian_consent_ref", "")).strip()[:60] or None,
    }
    if not consent["recording_tutor"] or not consent["recording_student"]:
        errs.append("녹음 동의: 튜터와 학생 모두 동의해야 녹음할 수 있습니다")
    if group == "child" and not (consent["guardian"] and consent["guardian_ref"]):
        errs.append("법정대리인 동의: 아동은 법정대리인 동의와 동의서 관리 번호가 필요합니다")
    mode = form.get("mode", "two_devices")
    if mode not in MODES:
        errs.append(f"mode: {' / '.join(MODES)}")
    meta = {
        "kind": "tutor_studio",
        "tutor_alias": alias["tutor"], "student_alias": alias["student"],
        "student_group": group, "age_band": age_band if group == "child" else None,
        "consent": consent, "mode": mode,
        "topic": str(form.get("topic", "")).strip()[:120],
        "notes": str(form.get("notes", "")).strip()[:500],
        "retention_until": str(form.get("retention_until", "")).strip()[:20] or None,  # 보관 기간 정책 미정: 입력만 받음
    }
    return meta, errs


def recording_allowed(meta: dict) -> list[str]:
    """녹음 시작 직전에 meta.json 을 다시 확인한다 (양식 이후 파일이 바뀌었을 수도 있음)."""
    c = meta.get("consent") or {}
    errs = []
    if not (c.get("recording_tutor") is True and c.get("recording_student") is True):
        errs.append("녹음 동의 없음")
    if meta.get("student_group") == "child" and not (c.get("guardian") is True and c.get("guardian_ref")):
        errs.append("아동 녹음에 법정대리인 동의 없음")
    if meta.get("student_group") not in GROUPS:
        errs.append("학생 구분 없음")
    return errs


# 녹음 ------------------------------------------------------------------------------------------------------------

class Track:
    def __init__(self, path: Path):
        self.w = wave.open(str(path), "wb")
        self.w.setnchannels(1)
        self.w.setsampwidth(2)
        self.w.setframerate(RATE)
        self.written = 0     # 쓴 샘플 수 (무음 포함)
        self.received = 0    # 브라우저에서 받은 샘플 수
        self.lead_pad = 0    # 처음 프레임 전에 채운 무음 (늦게 들어옴)
        self.gap_pad = 0     # 연결이 끊겼다가 다시 들어온 동안을 채운 무음
        self.resync = False  # 다시 연결됨: 다음 프레임에서 한 번만 서버 시계에 맞춘다
        self.tail_pad = 0    # 종료 때 두 트랙 길이를 맞추려고 채운 무음

    def pad(self, n: int) -> None:
        if n > 0:
            self.w.writeframes(b"\x00\x00" * n)
            self.written += n

    def info(self, elapsed_s: float) -> dict:
        return {"received_s": round(self.received / RATE, 3), "lead_pad_s": round(self.lead_pad / RATE, 3),
                "gap_pad_s": round(self.gap_pad / RATE, 3), "tail_pad_s": round(self.tail_pad / RATE, 3),
                # 기기 오디오 시계가 서버 시계보다 빠르면 양수 (끊김은 무음으로 메우지만 빠른 쪽은 보정하지 않음)
                "drift_s": round((self.written - self.tail_pad) / RATE - elapsed_s, 3)}


class Room:
    def __init__(self, sid: str, d: Path, meta: dict):
        self.sid, self.dir, self.meta = sid, d, meta
        self.sockets: dict[str, list[web.WebSocketResponse]] = {r: [] for r in (*ROLES, TURNS)}
        self.listen: dict[web.WebSocketResponse, bool] = {}
        self.tracks: dict[str, Track] | None = None
        self.t0: float | None = None
        # 순차 녹음: 지금 차례, 차례 기록, 공용 마이크에서 받은 샘플 수, 다시 연결 표시
        self.active = "tutor"
        self.turns: list[dict] = []
        self.mic_received = 0
        self.mic_resync = False

    @property
    def is_turns(self) -> bool:
        return self.meta.get("mode") == TURNS


class Studio:
    def __init__(self, cfg: dict, clock=time.monotonic):
        self.cfg = cfg
        self.queue = (ROOT / cfg["queue_dir"]) if not Path(cfg["queue_dir"]).is_absolute() else Path(cfg["queue_dir"])
        self.out = (ROOT / cfg["finetune_out"]) if not Path(cfg["finetune_out"]).is_absolute() else Path(cfg["finetune_out"])
        self.clock = clock
        self.gap = int(RATE * int(cfg["gap_fill_ms"]) / 1000)
        self.rooms: dict[str, Room] = {}

    # 저장 --------------------------------------------------------------------------------------------------------
    def sdir(self, sid: str) -> Path:
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}_[0-9a-f]{8}", sid):
            raise web.HTTPNotFound()
        return self.queue / sid

    def read_meta(self, sid: str) -> dict:
        p = self.sdir(sid) / "meta.json"
        if not p.is_file():
            raise web.HTTPNotFound()
        return json.loads(p.read_text(encoding="utf-8"))

    def save_meta(self, sid: str, meta: dict) -> None:
        p = self.sdir(sid) / "meta.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)

    def create(self, form: dict) -> tuple[str | None, list[str]]:
        meta, errs = validate(form if isinstance(form, dict) else {})
        if errs:
            return None, errs
        sid = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"
        (self.queue / sid).mkdir(parents=True, exist_ok=False)
        meta.update(id=sid, status="created", created_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.save_meta(sid, meta)
        return sid, []

    def room(self, sid: str) -> Room:
        if sid not in self.rooms:
            self.rooms[sid] = Room(sid, self.sdir(sid), self.read_meta(sid))
        return self.rooms[sid]

    # 녹음 시작·끝 ----------------------------------------------------------------------------------------------------
    def start(self, rm: Room) -> list[str]:
        if rm.tracks is not None:
            return ["이미 녹음 중"]
        rm.meta = self.read_meta(rm.sid)
        if rm.meta.get("status") != "created":
            return [f"이미 녹음한 세션입니다 (status={rm.meta.get('status')}). 새 세션을 만드세요"]
        errs = recording_allowed(rm.meta)
        if errs:
            return errs
        rm.tracks = {r: Track(rm.dir / f"{r}.wav") for r in ROLES}
        rm.t0 = self.clock()
        rm.active, rm.mic_received, rm.mic_resync = "tutor", 0, False
        rm.turns = [{"role": "tutor", "start_s": 0.0}] if rm.is_turns else []
        rm.meta.update(status="recording", started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.save_meta(rm.sid, rm.meta)
        return []

    def feed(self, rm: Room, role: str, pcm: bytes) -> None:
        """브라우저 프레임(PCM16 24 kHz 모노)을 쓴다.

        한 연결 안에서는 받은 순서대로 **이어 쓴다**. WebSocket 은 TCP 라서 프레임이 없어지지 않고 늦게 몰려 올 뿐이므로,
        도착 시각으로 무음을 채우면 몰려 온 음성이 실제보다 뒤로 밀린다. 샘플 수가 곧 기기 시계다.
        서버 시계에 맞추는 것은 두 번뿐이다: 그 역할의 첫 프레임(늦게 들어옴, lead_pad)과 다시 연결된 뒤 첫 프레임(gap_pad).
        """
        if rm.tracks is None or len(pcm) < 2:
            return
        if rm.is_turns or role == TURNS:
            if rm.is_turns and role == TURNS:
                self._feed_turns(rm, pcm)
            return
        tr = rm.tracks[role]
        n = len(pcm) // 2
        if tr.received == 0 or tr.resync:
            lag = int((self.clock() - rm.t0) * RATE) - tr.written - n
            if lag > self.gap:
                if tr.received == 0:
                    tr.lead_pad += lag
                else:
                    tr.gap_pad += lag
                tr.pad(lag)
            tr.resync = False
        tr.w.writeframes(pcm[: n * 2])
        tr.written += n
        tr.received += n

    def _feed_turns(self, rm: Room, pcm: bytes) -> None:
        """순차 녹음: 공용 마이크 프레임을 지금 차례 트랙에 쓰고, 다른 트랙에는 같은 길이의 무음을 쓴다.
        두 트랙의 길이가 늘 같으므로 시간축이 어긋나지 않는다. 서버 시계 맞춤(첫 프레임, 다시 연결)은 두 트랙에 똑같이 한다."""
        n = len(pcm) // 2
        on = rm.tracks[rm.active]
        off = rm.tracks["student" if rm.active == "tutor" else "tutor"]
        if rm.mic_received == 0 or rm.mic_resync:
            lag = int((self.clock() - rm.t0) * RATE) - on.written - n
            if lag > self.gap:
                for t in rm.tracks.values():
                    if rm.mic_received == 0:
                        t.lead_pad += lag
                    else:
                        t.gap_pad += lag
                    t.pad(lag)
            rm.mic_resync = False
        on.w.writeframes(pcm[: n * 2])
        on.written += n
        on.received += n
        off.pad(n)
        rm.mic_received += n

    def turn(self, rm: Room, role: str | None = None) -> list[str]:
        """순차 녹음에서 차례를 바꾼다 (role 이 없으면 상대 차례로). 차례 경계는 트랙 시간(쓴 샘플 수)으로 남긴다."""
        if rm.tracks is None:
            return ["녹음 중이 아닙니다"]
        if not rm.is_turns:
            return ["순차 녹음 세션이 아닙니다"]
        nxt = role or ("student" if rm.active == "tutor" else "tutor")
        if nxt not in ROLES:
            return [f"차례는 {ROLES} 중 하나"]
        if nxt == rm.active:
            return []
        at = round(rm.tracks["tutor"].written / RATE, 3)
        rm.turns[-1]["end_s"] = at
        rm.turns.append({"role": nxt, "start_s": at})
        rm.active = nxt
        return []

    def stop(self, rm: Room) -> dict:
        if rm.tracks is None:
            return rm.meta
        elapsed = self.clock() - rm.t0
        end = max(max(t.written for t in rm.tracks.values()), int(elapsed * RATE))
        for t in rm.tracks.values():
            t.tail_pad = end - t.written
            t.pad(end - t.written)
            t.w.close()
        rm.meta.update(status="recorded", recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       duration_s=round(end / RATE, 3), elapsed_s=round(elapsed, 3),
                       tracks={r: t.info(elapsed) for r, t in rm.tracks.items()})
        if rm.is_turns and rm.turns:
            rm.turns[-1]["end_s"] = round(end / RATE, 3)
            rm.meta["turns"] = [t for t in rm.turns if t["end_s"] > t["start_s"]]
        self.save_meta(rm.sid, rm.meta)
        rm.tracks, rm.t0 = None, None
        return rm.meta

    def state(self, rm: Room) -> dict:
        return {"type": "state", "sid": rm.sid, "status": rm.meta.get("status"), "mode": rm.meta.get("mode"),
                "connected": {r: len(rm.sockets[r]) for r in (*ROLES, TURNS)}, "recording": rm.tracks is not None,
                "active": rm.active if rm.tracks is not None and rm.is_turns else None, "turn_count": len(rm.turns),
                "elapsed_s": round(self.clock() - rm.t0, 1) if rm.t0 is not None else 0.0}

    async def broadcast(self, rm: Room, msg: dict) -> None:
        for r in (*ROLES, TURNS):
            for ws in list(rm.sockets[r]):
                if not ws.closed:
                    await ws.send_json(msg)

    # 삭제 (동의 철회) ---------------------------------------------------------------------------------------------------
    def delete(self, sid: str) -> dict:
        d = self.sdir(sid)
        rm = self.rooms.get(sid)
        if rm is not None and rm.tracks is not None:
            raise web.HTTPConflict(text="녹음 중에는 지울 수 없습니다")
        if not d.is_dir():
            raise web.HTTPNotFound()
        shutil.rmtree(d)
        self.rooms.pop(sid, None)
        derived = remove_derived(self.out, sid)
        return {"deleted": sid, "derived_removed": derived}


# HTTP ------------------------------------------------------------------------------------------------------------

def summary(sid: str, m: dict) -> dict:
    c = m.get("consent") or {}
    return {"id": sid, "status": m.get("status"), "created_at": m.get("created_at"), "tutor_alias": m.get("tutor_alias"),
            "student_alias": m.get("student_alias"), "student_group": m.get("student_group"), "age_band": m.get("age_band"),
            "mode": m.get("mode"), "topic": m.get("topic"), "duration_s": m.get("duration_s"),
            "training_consent": c.get("training_tutor") is True and c.get("training_student") is True,
            "guardian": c.get("guardian"), "retention_until": m.get("retention_until"), "tracks": m.get("tracks"),
            "turns": m.get("turns")}


def make_app(cfg: dict, clock=time.monotonic) -> web.Application:
    st = Studio(cfg, clock)
    app = web.Application(client_max_size=1024 * 64)

    async def index(_):
        return web.FileResponse(STATIC / "index.html")

    async def worklet(_):
        return web.FileResponse(WORKLET, headers={"Content-Type": "text/javascript"})

    async def create(request):
        try:
            form = await request.json()
        except ValueError:
            raise web.HTTPBadRequest(text="JSON 이 아닙니다")
        sid, errs = st.create(form)
        if errs:
            return web.json_response({"ok": False, "errors": errs}, status=400)
        return web.json_response({"ok": True, "id": sid, "links": {
            "tutor": f"/?sid={sid}&role=tutor", "student": f"/?sid={sid}&role=student", "both": f"/?sid={sid}&role=both",
            "turns": f"/?sid={sid}&role=turns"}})

    async def listing(_):
        rows = []
        for d in sorted((p for p in st.queue.iterdir() if p.is_dir()), reverse=True) if st.queue.is_dir() else []:
            mp = d / "meta.json"
            if mp.is_file():
                rows.append(summary(d.name, json.loads(mp.read_text(encoding="utf-8"))))
        return web.json_response({"sessions": rows, "allow_child_data": bool(cfg.get("allow_child_data", False))})

    async def detail(request):
        sid = request.match_info["sid"]
        return web.json_response(summary(sid, st.read_meta(sid)))

    async def remove(request):
        return web.json_response(st.delete(request.match_info["sid"]))

    async def audio(request):
        sid, role = request.match_info["sid"], request.match_info["role"]
        if role not in ROLES:
            raise web.HTTPNotFound()
        p = st.sdir(sid) / f"{role}.wav"
        if not p.is_file() or st.read_meta(sid).get("status") != "recorded":
            raise web.HTTPNotFound()
        return web.FileResponse(p, headers={"Cache-Control": "no-store"})

    async def ws(request):
        sid, role = request.match_info["sid"], request.match_info["role"]
        if role not in (*ROLES, TURNS):
            raise web.HTTPNotFound()
        rm = st.room(sid)
        if (role == TURNS) != rm.is_turns:  # 순차 녹음 세션은 turns 소켓만, 다른 세션은 tutor/student 소켓만
            raise web.HTTPConflict(text="이 세션의 녹음 방식과 맞지 않는 링크입니다")
        sock = web.WebSocketResponse(heartbeat=20, max_msg_size=1 << 20)
        await sock.prepare(request)
        rm.sockets[role].append(sock)
        rm.listen[sock] = request.query.get("listen", "0") == "1" and role != TURNS
        if rm.tracks is not None:  # 녹음 중에 다시 들어옴: 끊긴 동안을 무음으로 채운다
            if role == TURNS and rm.mic_received:
                rm.mic_resync = True
            elif role != TURNS and rm.tracks[role].received:
                rm.tracks[role].resync = True
        other = {"tutor": "student", "student": "tutor", TURNS: None}[role]
        controller = role in ("tutor", TURNS)  # 녹음 시작·종료·차례 바꾸기를 할 수 있는 화면
        await st.broadcast(rm, st.state(rm))
        try:
            async for msg in sock:
                if msg.type == WSMsgType.BINARY:
                    st.feed(rm, role, msg.data)
                    for o in list(rm.sockets[other]) if other else []:  # 상대 음성 들려주기 (기기 2대일 때)
                        if rm.listen.get(o) and not o.closed:
                            await o.send_bytes(msg.data)
                elif msg.type == WSMsgType.TEXT:
                    try:
                        ev = json.loads(msg.data)
                    except ValueError:
                        continue
                    kind = ev.get("type")
                    if kind == "listen":
                        rm.listen[sock] = bool(ev.get("on"))
                    elif kind in ("start", "stop", "turn"):
                        if not controller:  # 녹음 시작·끝은 튜터 화면만
                            await sock.send_json({"type": "error", "detail": "녹음 시작·종료는 튜터 화면에서 합니다"})
                            continue
                        if kind in ("start", "turn"):
                            errs = st.start(rm) if kind == "start" else st.turn(rm, ev.get("role"))
                            if errs:
                                await sock.send_json({"type": "error", "detail": "; ".join(errs)})
                        else:
                            st.stop(rm)
                            await st.broadcast(rm, {"type": "recorded", "meta": summary(sid, rm.meta)})
                        await st.broadcast(rm, st.state(rm))
                    elif kind == "ping":
                        await sock.send_json(st.state(rm))
        finally:
            rm.sockets[role].remove(sock)
            rm.listen.pop(sock, None)
            # 튜터(순차 녹음은 공용) 소켓이 모두 끊기면 녹음을 끝낸다 (녹음이 끝없이 이어지지 않게)
            if rm.tracks is not None and controller and not rm.sockets[role]:
                log.warning("튜터 연결이 끊겨 녹음을 끝냄: %s", sid)
                st.stop(rm)
            await st.broadcast(rm, st.state(rm))
            if not any(rm.sockets.values()) and rm.tracks is None:
                st.rooms.pop(sid, None)
        return sock

    app.router.add_get("/", index)
    app.router.add_get("/static/capture-worklet.js", worklet)
    app.router.add_static("/static", STATIC)
    app.router.add_post("/api/sessions", create)
    app.router.add_get("/api/sessions", listing)
    app.router.add_get("/api/sessions/{sid}", detail)
    app.router.add_delete("/api/sessions/{sid}", remove)
    app.router.add_get("/api/sessions/{sid}/audio/{role}.wav", audio)
    app.router.add_get("/ws/{sid}/{role}", ws)

    async def on_shutdown(app):
        for rm in list(st.rooms.values()):
            st.stop(rm)  # 종료 시 녹음 중인 세션을 닫아 wav 헤더를 마무리
    app.on_shutdown.append(on_shutdown)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")  # 외부에 열지 않는다. SSH 터널로 접속 (studio/README.md)
    ap.add_argument("--port", type=int, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_cfg()
    web.run_app(make_app(cfg), host=a.host, port=a.port or int(cfg["port"]), access_log=None)


if __name__ == "__main__":
    main()
