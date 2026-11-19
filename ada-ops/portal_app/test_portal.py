"""운영 포털 시험. 운영 스크립트(gpu-mode, gpu-wait, cpu-demo, studio)는 **실제로 부르지 않는다** (가짜 runner).
실행: cd ada-ops && ../english-agent/.venv-core/bin/python -m pytest -q portal_app/test_portal.py"""

import asyncio
import os
import sys
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server as ps  # noqa: E402

MODES = """# 주석
realtime   10     ko-gemma-vllm       11000
realtime   40     ko-web              0
realtime   50     en-moshi            12000
batch      10     transcribe-vllm     40000
train      10     finetune-job        40000
ghost      10     moshi-finetune      40000
en-solo    10     en-moshi-bf16       24000
"""
GW_ACTIVE = ("감시 중 (pid 1890345): bash /home/yang/pgm/voice_agent/ada-ops/gpu-wait run realtime\n"
             "마지막 판단: 2026-11-18 10:20:07 여유 부족: need=15000 avail=3947\n"
             "기록: 1709개, 최근 2시간 다른 프로그램 최고 42193 MiB")


def make_ops(tmp: Path) -> Path:
    ops = tmp / "ops"
    (ops / "services").mkdir(parents=True)
    (ops / "run").mkdir()
    (ops / "modes.conf").write_text(MODES, encoding="utf-8")
    for n, ex in [("ko-gemma-vllm", True), ("ko-web", True), ("en-moshi", False), ("transcribe-vllm", True), ("en-moshi-bf16", True), ("finetune-job", True)]:
        p = ops / "services" / f"{n}.sh"
        p.write_text("#!/bin/sh\n")
        os.chmod(p, 0o755 if ex else 0o644)
    return ops


class FakeRunner:
    def __init__(self, gw=GW_ACTIVE, demo="실행 중 아님"):
        self.gw, self.demo, self.calls = gw, demo, []

    async def __call__(self, argv, timeout=30):
        self.calls.append(argv)
        name, args = Path(argv[0]).name, argv[1:]
        if name == "gpu-mode" and args == ["mem"]:
            return 0, "49140 42193 6311 0"
        if name == "gpu-mode" and args[0] == "need":
            return 0, {"realtime": "15000", "batch": "40000", "train": "40000", "en-solo": "24000", "ghost": "0"}[args[1]]
        if name == "gpu-wait" and args == ["status"]:
            return 0, self.gw
        if name == "cpu-demo" and args == ["status"]:
            return 0, self.demo
        return 0, "ok"

    def actions(self):
        return [[Path(a[0]).name, *a[1:]] for a in self.calls
                if not (a[1:] in (["mem"], ["status"]) or (len(a) > 1 and a[1] == "need") or a[0] == "bash")]


def portal(tmp, **kw):
    r = FakeRunner(**kw)
    return ps.Portal(runner=r, ops=make_ops(tmp), probe=lambda p: False), r


def test_parsers():
    m = ps.parse_modes(MODES)
    assert [s["name"] for s in m["realtime"]] == ["ko-gemma-vllm", "ko-web", "en-moshi"]
    g = ps.parse_gpu_wait(GW_ACTIVE)
    assert g["active"] and g["pid"] == 1890345 and g["mode"] == "realtime" and g["others_peak"] == 42193
    assert not ps.parse_gpu_wait("감시 중 아님")["active"]


def test_status_and_flow(tmp_path):
    pt, _ = portal(tmp_path)
    st = asyncio.run(pt.status())
    cards = {c["mode"]: c for c in st["modes"]}
    assert cards["realtime"]["reservable"] and cards["realtime"]["reserved"]
    assert cards["realtime"]["partial"] == ["en-moshi"]          # 실행 권한이 없어 올라가지 않음을 알린다
    assert not cards["batch"]["reservable"] and cards["train"]["reservable"] and not cards["ghost"]["reservable"]
    assert cards["en-solo"]["reservable"]
    assert st["gpu_avail_for_us"] == 49140 - 42193 - 0 - 3000 and not cards["realtime"]["fits_now"]
    flow = {n["id"]: n for s in st["flow"] for n in s["nodes"]}
    assert flow["ko"]["state"] == "gpu_wait" and flow["en"]["state"] == "decision" and flow["train"]["state"] in ("stopped", "cpu_ready", "decision")
    assert [s["stage"] for s in st["flow"]] == ["학생 서비스", "녹음 수집", "야간 처리", "학습 데이터", "학습"]


def test_reserve_rules(tmp_path):
    pt, r = portal(tmp_path)
    code, res = asyncio.run(pt.action({"action": "reserve", "mode": "en-solo"}))
    assert code == 409 and res["need_confirm"] and r.actions() == []  # 이미 realtime 예약이 있으면 확인 없이 바꾸지 않는다
    code, res = asyncio.run(pt.action({"action": "reserve", "mode": "en-solo", "replace": True}))
    assert code == 200 and r.actions() == [["gpu-wait", "stop"], ["gpu-wait", "start", "en-solo"]]
    for mode in ("batch", "ghost", "nope"):
        assert asyncio.run(pt.action({"action": "reserve", "mode": mode}))[0] == 400


def test_reserve_train_once_and_rearm(tmp_path):
    """학습은 끝나는 작업: --once 로 예약하고, 감시 중이던 realtime 예약을 run/gpu-wait.after 에 남겨 학습 뒤 다시 걸게 한다."""
    pt, r = portal(tmp_path)
    code, res = asyncio.run(pt.action({"action": "reserve", "mode": "train", "replace": True}))
    assert code == 200 and r.actions() == [["gpu-wait", "stop"], ["gpu-wait", "start", "train", "--once"]]
    assert (pt.ops / "run" / "gpu-wait.after").read_text().strip() == "realtime"


def test_runner_does_not_hide_gpu(monkeypatch):
    """포털은 CUDA_VISIBLE_DEVICES="" 로 뜨지만, gpu-wait 에 그대로 물려주면 GPU 서비스가 GPU 를 못 본다."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    rc, out = asyncio.run(ps.run_cmd(["bash", "-c", 'echo "[${CUDA_VISIBLE_DEVICES-unset}]"']))
    assert rc == 0 and out == "[unset]"


def test_never_starts_gpu_mode_directly(tmp_path):
    pt, r = portal(tmp_path, gw="감시 중 아님")
    for body in ({"action": "reserve", "mode": "realtime"}, {"action": "gpu_stop"}, {"action": "unreserve"},
                 {"action": "gpu_mode", "mode": "realtime"}, {"action": "shell", "cmd": "rm -rf /"}):
        asyncio.run(pt.action(body))
    assert r.actions() == [["gpu-wait", "start", "realtime"], ["gpu-mode", "stop"], ["gpu-wait", "stop"]]


def test_demo_conflicts(tmp_path):
    pt, r = portal(tmp_path)
    code, res = asyncio.run(pt.action({"action": "demo_start"}))
    assert code == 409 and "예약" in res["error"]
    pt2, r2 = portal(tmp_path / "b", gw="감시 중 아님", demo="실행 중 (pid 1)")
    code, res = asyncio.run(pt2.action({"action": "reserve", "mode": "realtime"}))
    assert code == 409 and "시연" in res["error"] and r2.actions() == []


def test_http_csrf(tmp_path):
    pt, r = portal(tmp_path, gw="감시 중 아님")

    async def body():
        async with TestClient(TestServer(ps.make_app(pt))) as c:
            assert (await c.get("/")).status == 200
            st = await (await c.get("/api/status")).json()
            assert st["flow"] and st["decisions"]
            assert (await c.post("/api/action", json={"action": "studio_start"})).status == 403
            r1 = await c.post("/api/action", json={"action": "studio_start"},
                              headers={"X-Portal": "1", "Origin": "http://evil.example"})
            assert r1.status == 403
            r2 = await c.post("/api/action", json={"action": "studio_start"}, headers={"X-Portal": "1"})
            assert r2.status == 200
    asyncio.run(body())
    assert r.actions() == [["studio", "start"]]
