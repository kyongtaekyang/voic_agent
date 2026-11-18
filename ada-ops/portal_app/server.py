"""음성 Agent 운영 포털: 전체 메뉴와 흐름(학생 서비스 → 녹음 수집 → 야간 처리 → 학습 데이터 → 학습)을 한 화면에 보여 준다.

  CUDA_VISIBLE_DEVICES= english-agent/.venv-core/bin/python ada-ops/portal_app/server.py      # http://127.0.0.1:7100 (SSH 터널)
  ada-ops/portal start|stop|status

- GPU 를 쓰지 않는다. 상태는 기존 운영 스크립트(gpu-mode, gpu-wait, cpu-demo, studio)와 pid·ready 파일, 설정 파일을 읽어서 만든다.
- 할 수 있는 동작은 허용 목록뿐이다: GPU 자동 기동 예약(gpu-wait start/stop), 우리 GPU 서비스 중지(gpu-mode stop),
  CPU 시연(cpu-demo), 튜터 녹음 스튜디오(studio).
  **gpu-mode <모드> 를 바로 부르는 동작은 두지 않는다.** GPU 는 다른 사용자와 같이 쓰므로, 기동은 gpu-wait 가
  다른 프로그램의 최근 최고 사용량을 보고 여유가 있을 때만 한다 (ada-ops/gpu-wait 머리말).
- 학습 데이터 변환·학습은 화면에서 실행하지 않고, 막힌 이유와 실행 명령만 보여 준다.
- 음성 파일은 보여 주지 않는다 (세션 수만). 음성 데이터는 서버 밖으로 보내지 않는다.
- 명령은 셸 없이 고정된 인자 목록으로만 실행한다. POST 는 같은 출처(Origin)와 X-Portal 헤더가 있어야 받는다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import socket
import time
from pathlib import Path

import yaml
from aiohttp import web

OPS = Path(__file__).resolve().parent.parent
ROOT = OPS.parent
KO = ROOT / "korean_agent"
EN = ROOT / "english-agent"
STATIC = Path(__file__).resolve().parent / "static"
PORT = 7100
CACHE_S = 5.0
RESERVABLE = ("realtime", "en-solo")  # gpu-wait 로 예약할 수 있는 모드 (batch·summary 는 nightly-batch 가, train 은 아직 없음)
MARGIN = 3000                         # gpu-wait 기본 여유분과 같게
log = logging.getLogger("portal")

# 화면 링크 (모두 127.0.0.1 에만 열려 있어 SSH 터널로 접속)
SCREENS = {"ko_web": 7201, "en_web": 7101, "studio": 7103}
PROBE_PORTS = {"ko_web": 7201, "ko_realtime": 8766, "ko_gemma": 8001, "transcribe": 8002,
               "en_web": 7101, "en_core": 7102, "moshi": 8998, "studio": 7103}

# 사용자 결정이 필요한 항목 (2026-11-18 기준, 바뀌면 여기를 고친다)
DECISIONS = [
    {"id": "gpu_window", "title": "GPU 사용 시간대", "detail": "baek 의 app_flash.py 가 30~43 GB 를 쓴다. 한국어 실시간 약 15 GB, Moshi BF16 약 24 GB 가 필요하다. 시간대를 나눌지 정해야 한다.", "blocks": ["학생 서비스", "야간 처리", "학습"]},
    {"id": "consent_policy", "title": "녹음 동의·보관 기간 정책", "detail": "두 Agent 의 세션 녹음(record)은 정책이 정해질 때까지 꺼 둔다. 그래서 야간 처리 대기열이 비어 있다.", "blocks": ["녹음 수집", "야간 처리"]},
    {"id": "child_review", "title": "아동 데이터 내부관리규정 검토", "detail": "검토가 끝나야 finetune.allow_child_data 를 켤 수 있다. 그 전에는 아동 녹음을 학습 데이터로 바꾸지 않는다.", "blocks": ["학습 데이터"]},
    {"id": "finetune_source", "title": "튜터 말투 학습 데이터 출처", "detail": "튜터 녹음 스튜디오 녹음만 쓸지, 동의받은 실제 수업 세션도 쓸지 정해야 한다.", "blocks": ["학습 데이터"]},
    {"id": "transcribe_ts", "title": "Transcribe 30B 단어 타임스탬프 형식 확인", "detail": "GPU 에서 한 번 띄워서 미들웨어 출력 형식을 확인해야 tutor_words.json 을 만들 수 있다.", "blocks": ["학습 데이터"]},
    {"id": "moshi_int8", "title": "영어 실시간 Moshi INT8 활성화", "detail": "services/en-moshi.sh 는 GPU 실측(메모리, 프레임 지연 80 ms)과 승인 전까지 실행 권한을 꺼 두었다. 그래서 realtime 모드는 지금 한국어만 올린다.", "blocks": ["학생 서비스"]},
    {"id": "multi_session", "title": "동시 접속 구성", "detail": "두 Agent 모두 지금은 한 번에 한 세션만 받는다. 여러 인스턴스를 띄울지 정해야 한다.", "blocks": ["학생 서비스"]},
    {"id": "remote_access", "title": "외부(가정) 접속 방법", "detail": "지금은 SSH 터널로만 접속한다. 학생이 집에서 쓰려면 인증이 있는 HTTPS 주소가 필요하다.", "blocks": ["학생 서비스", "녹음 수집"]},
    {"id": "ko_child_eval", "title": "한국어 아동 평가 세트 위치", "detail": "아동 음성 인식 평가용 데이터를 어디에 둘지 정해야 한다.", "blocks": ["야간 처리"]},
    {"id": "whisper", "title": "Whisper 사용 여부", "detail": "moshi-finetune 공식 주석 도구는 Whisper 를 쓰지만 사양 밖 모델이다. 지금은 Transcribe 30B 로 대신할 계획이다.", "blocks": ["학습 데이터"]},
]


# 명령 실행 -----------------------------------------------------------------------------------------------------------

async def run_cmd(argv: list[str], timeout: float = 30.0) -> tuple[int, str]:
    """셸 없이 실행한다. GPU 를 쓰지 않도록 CUDA_VISIBLE_DEVICES 를 비운다 (gpu-mode 가 띄우는 서비스는 자기 설정을 따름)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    p = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                             cwd=str(OPS), env=env, start_new_session=True)
    try:
        out, _ = await asyncio.wait_for(p.communicate(), timeout)
    except asyncio.TimeoutError:
        p.kill()
        return 124, f"시간 초과 ({timeout:.0f}초)"
    return p.returncode, out.decode("utf-8", "replace").strip()


# 상태 읽기 -----------------------------------------------------------------------------------------------------------

def parse_modes(text: str) -> dict[str, list[dict]]:
    modes: dict[str, list[dict]] = {}
    for line in text.splitlines():
        f = line.split()
        if not f or f[0].startswith("#") or len(f) < 4:
            continue
        modes.setdefault(f[0], []).append({"order": int(f[1]), "name": f[2], "mib": int(f[3])})
    for v in modes.values():
        v.sort(key=lambda s: s["order"])
    return modes


def parse_gpu_wait(text: str) -> dict:
    m = re.search(r"감시 중 \(pid (\d+)\).*?\brun (\S+)", text)
    last = re.search(r"마지막 판단: (.*)", text)
    peak = re.search(r"다른 프로그램 최고 (\d+) MiB", text)
    return {"active": bool(m), "pid": int(m.group(1)) if m else None, "mode": m.group(2) if m else None,
            "last": last.group(1).strip() if last else None, "others_peak": int(peak.group(1)) if peak else None,
            "raw": text[:2000]}


def pid_alive(pidfile: Path) -> int | None:
    try:
        pid = int(pidfile.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


def load_yaml(p: Path) -> dict:
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def queue_counts(d: Path) -> dict:
    """세션 폴더(meta.json)의 상태별 개수. 폴더가 없으면 0 (녹음이 꺼져 있으면 정상)."""
    out = {"total": 0, "by_status": {}, "exists": d.is_dir()}
    if not d.is_dir():
        return out
    for s in d.iterdir():
        mp = s / "meta.json"
        if not (s.is_dir() and mp.is_file()):
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            m = {"status": "읽기 실패"}
        out["total"] += 1
        st = str(m.get("status") or "unknown")
        out["by_status"][st] = out["by_status"].get(st, 0) + 1
        if m.get("student_group") == "child":
            out["child"] = out.get("child", 0) + 1
    return out


def finetune_counts(d: Path) -> dict:
    if not d.is_dir():
        return {"sets": 0, "items": 0}
    sets = list(d.glob("*.jsonl"))
    items = 0
    for p in sets:
        try:
            items += sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
        except OSError:
            pass
    return {"sets": len(sets), "items": items}


class Portal:
    def __init__(self, runner=run_cmd, ops: Path = OPS, probe=port_open):
        self.run, self.ops, self.probe = runner, ops, probe
        self.lock = asyncio.Lock()
        self.cache: tuple[float, dict] | None = None
        self.need_cache: dict[str, int] = {}

    def svc_file_state(self, name: str) -> str:
        sh = self.ops / "services" / f"{name}.sh"
        if not sh.exists():
            return "not_built"
        if not os.access(sh, os.X_OK):
            return "disabled"
        return "ok"

    async def svc_state(self, name: str) -> dict:
        fs = self.svc_file_state(name)
        if fs != "ok":
            return {"name": name, "state": fs}
        pid = pid_alive(self.ops / "run" / f"{name}.pid")
        if pid is None:
            return {"name": name, "state": "stopped"}
        ready = self.ops / "services" / f"{name}.ready"
        if ready.exists():
            rc, _ = await self.run(["bash", str(ready)], 5)
            return {"name": name, "state": "running" if rc == 0 else "starting", "pid": pid}
        return {"name": name, "state": "running", "pid": pid}

    async def need(self, mode: str) -> int | None:
        if mode not in self.need_cache:
            rc, out = await self.run([str(self.ops / "gpu-mode"), "need", mode], 15)
            if rc != 0 or not out.strip().splitlines()[-1].strip().isdigit():
                return None
            self.need_cache[mode] = int(out.strip().splitlines()[-1])
        return self.need_cache[mode]

    async def collect(self) -> dict:
        ops = self.ops
        modes = parse_modes((ops / "modes.conf").read_text(encoding="utf-8")) if (ops / "modes.conf").exists() else {}
        rc, mem = await self.run([str(ops / "gpu-mode"), "mem"], 15)
        gpu = None
        if rc == 0 and len(mem.split()) >= 4:
            total, used, free, ours = (int(x) for x in mem.split()[:4])
            gpu = {"total": total, "used": used, "free": free, "ours": ours, "others": used - ours}
        _, gw_text = await self.run([str(ops / "gpu-wait"), "status"], 15)
        gw = parse_gpu_wait(gw_text)
        current = (ops / "run" / "mode").read_text().strip() if (ops / "run" / "mode").exists() else ""
        _, demo = await self.run([str(ops / "cpu-demo"), "status"], 15)
        studio_pid = pid_alive(ops / "run" / "studio.pid")
        portal_states = {}
        for m, svcs in modes.items():
            for s in svcs:
                if s["name"] not in portal_states:
                    portal_states[s["name"]] = await self.svc_state(s["name"])
        avail = None
        if gpu:
            peak = gw["others_peak"] if gw["others_peak"] is not None else gpu["others"]
            avail = gpu["total"] - peak - gpu["ours"] - MARGIN
        mode_cards = []
        for m, svcs in modes.items():
            n = await self.need(m)
            gpu_svcs = [s for s in svcs if s["mib"] > 0]
            runnable = [s for s in gpu_svcs if portal_states[s["name"]]["state"] not in ("not_built", "disabled")]
            reason = None
            if m not in RESERVABLE:
                reason = {"batch": "nightly-batch 가 밤에 직접 올린다", "summary": "nightly-batch 가 밤에 직접 올린다",
                          "train": "학습 서비스(services/moshi-finetune.sh)가 아직 없다"}.get(m, "예약 대상 아님")
            elif not runnable:
                reason = "실행할 수 있는 GPU 서비스가 없다"
            mode_cards.append({
                "mode": m, "need": n, "services": [dict(s, **portal_states[s["name"]]) for s in svcs],
                "reservable": reason is None, "reason": reason, "current": m == current,
                "reserved": gw["active"] and gw["mode"] == m,
                "fits_now": (n is not None and avail is not None and n <= avail),
                "partial": [s["name"] for s in gpu_svcs if portal_states[s["name"]]["state"] in ("not_built", "disabled")],
            })
        ports = {k: self.probe(p) for k, p in PROBE_PORTS.items()}
        ko_cfg, en_cfg = load_yaml(KO / "config" / "services.yaml"), load_yaml(EN / "config" / "services.yaml")
        ko_q = KO / ((ko_cfg.get("slow_path") or {}).get("queue_dir") or "queue/korean")
        en_q = EN / (en_cfg.get("queue_dir") or "queue/english")
        tu_q = EN / ((en_cfg.get("studio") or {}).get("queue_dir") or "queue/tutor")
        ft_out = EN / ((en_cfg.get("studio") or {}).get("finetune_out") or "finetune/data")
        status = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": gpu, "gpu_wait": gw, "gpu_avail_for_us": avail, "current_mode": current or None,
            "modes": mode_cards, "ports": ports,
            "cpu_demo": "실행 중" in demo and "아님" not in demo,
            "studio": studio_pid is not None,
            "record": {"korean": (ko_cfg.get("slow_path") or {}).get("record") is True,
                       "english": (en_cfg.get("gateway") or {}).get("record") is True},
            "allow_child_data": (en_cfg.get("finetune") or {}).get("allow_child_data") is True,
            "queues": {"korean": queue_counts(ko_q), "english": queue_counts(en_q), "tutor": queue_counts(tu_q)},
            "finetune": finetune_counts(ft_out),
            "screens": SCREENS, "decisions": DECISIONS,
        }
        status["flow"] = build_flow(status)
        return status

    async def status(self, fresh: bool = False) -> dict:
        now = time.monotonic()
        if not fresh and self.cache and now - self.cache[0] < CACHE_S:
            return self.cache[1]
        st = await self.collect()
        self.cache = (time.monotonic(), st)
        return st

    # 동작 --------------------------------------------------------------------------------------------------------
    async def action(self, req: dict) -> tuple[int, dict]:
        act = req.get("action")
        if self.lock.locked():
            return 409, {"ok": False, "error": "다른 동작을 처리하는 중입니다. 잠시 뒤 다시 하세요"}
        async with self.lock:
            st = await self.status(fresh=True)
            gw = st["gpu_wait"]
            ops = str(self.ops)
            if act == "reserve":
                mode = req.get("mode")
                card = next((c for c in st["modes"] if c["mode"] == mode), None)
                if card is None:
                    return 400, {"ok": False, "error": f"모르는 모드: {mode}"}
                if not card["reservable"]:
                    return 400, {"ok": False, "error": f"{mode} 는 예약할 수 없습니다: {card['reason']}"}
                if st["cpu_demo"]:
                    return 409, {"ok": False, "error": "CPU 시연이 실행 중입니다. 시연이 7201·8001·8766 포트를 쓰므로 GPU 기동이 실패합니다. 시연을 먼저 끄세요"}
                cmds = []
                if gw["active"]:
                    if req.get("replace") is not True:
                        return 409, {"ok": False, "need_confirm": True,
                                     "error": f"이미 {gw['mode']} 예약이 감시 중입니다 (pid {gw['pid']}). 바꾸려면 확인하세요"}
                    cmds.append([f"{ops}/gpu-wait", "stop"])
                cmds.append([f"{ops}/gpu-wait", "start", mode])
            elif act == "unreserve":
                cmds = [[f"{ops}/gpu-wait", "stop"]]
            elif act == "gpu_stop":
                cmds = [[f"{ops}/gpu-mode", "stop"]]
            elif act == "demo_start":
                if gw["active"] or st["current_mode"]:
                    return 409, {"ok": False, "error": "GPU 예약이나 운영 모드가 있으면 시연을 켜지 않습니다. 시연이 포트를 차지하면 GPU 기동이 실패합니다"}
                cmds = [[f"{ops}/cpu-demo", "start"]]
            elif act == "demo_stop":
                cmds = [[f"{ops}/cpu-demo", "stop"]]
            elif act in ("studio_start", "studio_stop"):
                cmds = [[f"{ops}/studio", act.split("_")[1]]]
            else:
                return 400, {"ok": False, "error": f"허용하지 않은 동작: {act}"}
            outs, rc = [], 0
            for argv in cmds:
                rc, out = await self.run(argv, 240)
                outs.append(f"$ {' '.join(Path(argv[0]).name if i == 0 else a for i, a in enumerate(argv))}\n{out}")
                log.info("동작 %s: %s → %s", act, argv, rc)
                if rc != 0:
                    break
            self.cache = None
            return (200 if rc == 0 else 500), {"ok": rc == 0, "output": "\n".join(outs)}


# 흐름 ----------------------------------------------------------------------------------------------------------------

def _svc(st: dict, mode: str, name: str) -> str:
    for c in st["modes"]:
        if c["mode"] == mode:
            for s in c["services"]:
                if s["name"] == name:
                    return s["state"]
    return "not_built"


def build_flow(st: dict) -> list[dict]:
    """다섯 단계와 각 노드의 상태. state: running, stopped, gpu_wait, not_built, policy, decision, cpu_ready, demo."""
    gw, ports, q = st["gpu_wait"], st["ports"], st["queues"]
    reserved_rt = gw["active"] and gw["mode"] == "realtime"

    def live(port_key, ok_detail):
        return ("running", ok_detail) if ports.get(port_key) else None

    # 1) 학생 서비스
    if st["cpu_demo"]:
        ko = ("demo", "CPU 시연 중 (가짜 LLM, 고정 답변)")
        en = ("demo", "CPU 시연 중 (가짜 Moshi, 고정 대본)")
    else:
        ko = live("ko_web", "실행 중: 화면 7201 → Realtime 8766 → Gemma 8001") or (
            ("gpu_wait", "GPU 여유를 기다리는 중 (gpu-wait realtime 예약)") if reserved_rt else ("stopped", "중지. 서비스 운영에서 GPU 자동 기동을 예약할 수 있다"))
        moshi = _svc(st, "realtime", "en-moshi")
        if ports.get("en_web") and ports.get("moshi"):
            en = ("running", "실행 중: 화면 7101 → Gateway 7102 → Moshi 8998")
        elif moshi in ("disabled", "not_built"):
            en = ("decision", "realtime 모드에서 Moshi INT8 이 아직 꺼져 있다 (GPU 실측·승인 대기). en-solo 모드(BF16, 24 GB)로는 예약할 수 있다")
        else:
            en = ("gpu_wait", "GPU 여유를 기다리는 중") if gw["active"] else ("stopped", "중지")
    rec_ko = ("running", "세션 녹음 켜짐") if st["record"]["korean"] else ("policy", "녹음 동의·보관 기간 정책이 정해질 때까지 꺼 둠 (record: false)")
    rec_en = ("running", "세션 녹음 켜짐") if st["record"]["english"] else ("policy", "녹음 동의·보관 기간 정책이 정해질 때까지 꺼 둠 (record: false)")
    tq = q["tutor"]
    studio = ("running", f"실행 중, 세션 {tq['total']}개") if st["studio"] else ("stopped", f"중지, 세션 {tq['total']}개")
    queued = q["korean"]["by_status"].get("queued", 0) + q["english"]["by_status"].get("queued", 0)
    batch = ("gpu_wait", f"처리 대기 세션 {queued}개. Transcribe 30B(약 40 GB)가 필요해 GPU 여유가 없으면 실행하지 못함") if queued else (
        "policy", "처리할 세션 없음 (세션 녹음이 꺼져 있음). 매일 밤 실행 예약(cron)은 아직 걸지 않음")
    ts = ("decision", "Transcribe 30B 단어 타임스탬프 형식 확인 필요 (GPU)")
    child = "허용" if st["allow_child_data"] else "보류 (내부관리규정 검토 전)"
    prep = ("cpu_ready", f"CPU 로 실행 가능. 지금은 단어 타임스탬프가 없어 모든 세션을 건너뜀. 아동 데이터: {child}")
    ft = st["finetune"]
    train_state = _svc(st, "train", "moshi-finetune")
    train = ("not_built", "학습 서비스(moshi-finetune) 아직 없음. 약 40 GB GPU 필요") if train_state == "not_built" else ("stopped", "준비됨")
    apply = ("decision", "학습한 LoRA 를 실시간 Moshi 에 적용할지는 평가 후 결정")

    def node(id_, title, s, link=None, menu=None):
        return {"id": id_, "title": title, "state": s[0], "detail": s[1], "link": link, "menu": menu}

    return [
        {"stage": "학생 서비스", "desc": "학생이 Agent 와 실시간으로 대화한다", "nodes": [
            node("ko", "한국어 Agent", ko, SCREENS["ko_web"], "services"),
            node("en", "영어 Agent (Moshi)", en, SCREENS["en_web"], "services")]},
        {"stage": "녹음 수집", "desc": "동의받은 대화를 서버에 저장한다", "nodes": [
            node("rec_ko", "한국어 세션 녹음", rec_ko, None, "data"),
            node("rec_en", "영어 세션 녹음", rec_en, None, "data"),
            node("studio", "튜터 녹음 스튜디오", studio, SCREENS["studio"], "data")]},
        {"stage": "야간 처리", "desc": "밤에 GPU 로 축어 전사, 지표, 요약을 만든다", "nodes": [
            node("batch", "야간 배치 (Transcribe·요약·지표)", batch, None, "data")]},
        {"stage": "학습 데이터", "desc": "튜터 대화를 학습 형식(스테레오 + 단어 정렬)으로 바꾼다", "nodes": [
            node("ts", "튜터 단어 타임스탬프", ts, None, "train"),
            node("prep", f"학습 데이터 변환 (현재 {ft['items']}개)", prep, None, "train")]},
        {"stage": "학습", "desc": "Moshi 를 튜터 말투로 추가 학습하고 적용한다", "nodes": [
            node("train", "튜터 말투 LoRA 학습", train, None, "train"),
            node("apply", "실시간 Agent 에 적용", apply, None, "train")]},
    ]


# HTTP ---------------------------------------------------------------------------------------------------------------

def make_app(portal: Portal | None = None) -> web.Application:
    pt = portal or Portal()
    app = web.Application(client_max_size=1024 * 16)

    async def index(_):
        return web.FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    async def status(request):
        return web.json_response(await pt.status(fresh=request.query.get("fresh") == "1"))

    async def action(request):
        origin = request.headers.get("Origin")
        if request.headers.get("X-Portal") != "1" or (origin and origin.split("://", 1)[-1] != request.host):
            raise web.HTTPForbidden(text="포털 화면에서만 요청할 수 있습니다")
        try:
            body = await request.json()
        except ValueError:
            raise web.HTTPBadRequest(text="JSON 이 아닙니다")
        code, res = await pt.action(body if isinstance(body, dict) else {})
        return web.json_response(res, status=code)

    app.router.add_get("/", index)
    app.router.add_static("/static", STATIC)
    app.router.add_get("/api/status", status)
    app.router.add_post("/api/action", action)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")  # 외부에 열지 않는다
    ap.add_argument("--port", type=int, default=PORT)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    web.run_app(make_app(), host=a.host, port=a.port, access_log=None)


if __name__ == "__main__":
    main()
