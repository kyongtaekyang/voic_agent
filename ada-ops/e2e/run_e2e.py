"""전체 흐름 시험 (GPU 없음): 녹음 → GPU 예약 → 단어 타임스탬프 → 학습 데이터 → 학습 → 적용 → 포털 표시.

  english-agent/.venv-core/bin/python ada-ops/e2e/run_e2e.py [--keep]

실제 프로그램(CLI·서버·운영 스크립트)을 그대로 쓰고, GPU 가 필요한 두 곳만 대역으로 바꾼다.
  - Transcribe 30B → mock_transcribe.py (응답 형식은 실제 NeMo 미들웨어와 같음)
  - moshi-finetune 학습 → check_train.py (moshi-finetune 의 실제 데이터 로더·Interleaver 로 학습 데이터를 읽은 뒤 체크포인트를 씀)
  - nvidia-smi → 가짜 (49 GB 중 1 GB 사용)
격리: 임시 폴더에 ada-ops 운영 스크립트(gpu-mode, gpu-wait)를 복사해서 run/·logs/ 를 따로 쓰고, 설정(EVA_CONFIG)으로 녹음·학습
데이터·결과를 모두 임시 폴더에 둔다. 포트는 17103(스튜디오), 18002(가짜 Transcribe). 실제 gpu-wait 예약, 실제 녹음, 실제
finetune/data 는 건드리지 않는다 (시작할 때 확인하고, 실제 경로를 가리키면 멈춤).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

OPS = Path(__file__).resolve().parent.parent
VA = OPS.parent
EN = VA / "english-agent"
PY_CORE = EN / ".venv-core" / "bin" / "python"
PY_FT = Path.home() / "opt" / "moshi-finetune" / ".venv" / "bin" / "python"
RATE = 24000
STUDIO_PORT, TR_PORT = 17103, 18002
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'통과' if ok else '실패'}] {name}{' — ' + detail if detail else ''}", flush=True)
    if not ok:
        raise AssertionError(f"{name}: {detail}")


def port_free(p: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", p)) != 0


def wait_port(p: int, timeout: float = 30) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not port_free(p):
            return
        time.sleep(0.2)
    raise TimeoutError(f"port {p}")


# 준비 -------------------------------------------------------------------------------------------------------------

def make_sandbox(T: Path) -> dict:
    ops = T / "ops"
    (ops / "services").mkdir(parents=True)
    (ops / "run").mkdir()
    (ops / "logs").mkdir()
    for n in ("gpu-mode", "gpu-wait"):
        shutil.copy2(OPS / n, ops / n)
    shutil.copy2(OPS / "services" / "finetune-job.sh", ops / "services" / "finetune-job.sh")
    (ops / "modes.conf").write_text("train 10 finetune-job 40000\nrealtime 10 e2e-not-built 1000\n", encoding="utf-8")
    if ops.resolve() == OPS.resolve() or (ops / "run").resolve() == (OPS / "run").resolve():
        raise SystemExit("격리 실패: 임시 ops 가 실제 ada-ops 를 가리킵니다")
    bin_ = T / "bin"
    bin_.mkdir()
    smi = bin_ / "nvidia-smi"
    smi.write_text("""#!/usr/bin/env bash
# e2e 가짜 nvidia-smi: 49140 MiB 중 1000 MiB 사용, 계산 프로세스 없음
case "$*" in
  *query-compute-apps*) ;;
  *name,memory.total*) echo "E2E FAKE GPU, 49140 MiB, 1000 MiB, 48140 MiB" ;;
  *memory.total,memory.used,memory.free*) echo "49140, 1000, 48140" ;;
  *memory.free*) echo "48140" ;;
  *) echo "49140, 1000" ;;
esac
""", encoding="utf-8")
    smi.chmod(0o755)
    cfg = yaml.safe_load((EN / "config" / "services.yaml").read_text(encoding="utf-8"))
    cfg["queue_dir"] = str(T / "queue" / "english")
    cfg["slow_path"].update(queue_dir=str(T / "queue" / "english"), transcribe_url=f"http://127.0.0.1:{TR_PORT}")
    cfg["studio"].update(port=STUDIO_PORT, queue_dir=str(T / "queue" / "tutor"), finetune_out=str(T / "data"))
    cfg["finetune"] = {"allow_child_data": False, "min_items": 1, "data_dir": str(T / "data"), "runs_dir": str(T / "runs"),
                       "job_status": str(T / "runs" / "job_status.json"), "active_lora": str(T / "active_lora.json"),
                       "base_config": "finetune/moshi_lora.yaml", "transcribe_ready_timeout_s": 60}
    cfg_p = T / "services.yaml"
    cfg_p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    env = dict(os.environ)
    # 실제 GPU 는 보이지 않게 한다 (가짜 nvidia-smi 로 GPU 여유를 흉내 냄). 학습 대역도 이 값을 확인하고 멈춘다
    env.update(CUDA_VISIBLE_DEVICES="")
    env.update(PATH=f"{bin_}:{env['PATH']}", EVA_CONFIG=str(cfg_p), VA_ROOT=str(VA), HF_HUB_OFFLINE="1",
               FT_TRANSCRIBE_CMD=f"{PY_CORE} {OPS / 'e2e' / 'mock_transcribe.py'} --port {TR_PORT}",
               FT_TRAIN_CMD=f"{PY_FT} {OPS / 'e2e' / 'check_train.py'}",
               FT_CONFIG_OVERRIDES=json.dumps({"duration_sec": 8, "batch_size": 1, "max_steps": 1}))
    return {"ops": ops, "cfg": cfg_p, "env": env}


def tone(sec: float, f0: float, amp: float = 0.3) -> np.ndarray:
    """음절처럼 켜졌다 꺼지는 배음 소리 (0.3초 소리, 0.15초 쉼)."""
    t = np.arange(int(sec * RATE)) / RATE
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3))
    gate = ((t % 0.45) < 0.3).astype(np.float32)
    return (amp * x * gate / 1.8).astype(np.float32)


# 1. 녹음 ----------------------------------------------------------------------------------------------------------

async def record_session(sb: dict) -> str:
    import aiohttp
    base = f"http://127.0.0.1:{STUDIO_PORT}"
    form = {"tutor_alias": "E2E-T", "student_alias": "E2E-S", "student_group": "adult", "mode": "turns",
            "consent_recording_tutor": True, "consent_recording_student": True,
            "consent_training_tutor": True, "consent_training_student": True, "topic": "e2e"}
    async with aiohttp.ClientSession() as s:
        r = await s.post(f"{base}/api/sessions", json=form)
        sid = (await r.json())["id"]
        ws = await s.ws_connect(f"{base}/ws/{sid}/turns")
        await ws.send_json({"type": "start"})
        while True:
            ev = json.loads((await ws.receive()).data)
            if ev.get("type") == "state" and ev.get("recording"):
                break
        plan = [("tutor", 3.0, 220), ("student", 2.0, 330), ("tutor", 3.0, 220), ("student", 2.0, 330)]
        t_start = time.monotonic()
        sent = 0.0
        for i, (role, sec, f0) in enumerate(plan):
            if i:
                await ws.send_json({"type": "turn"})
            x = (tone(sec, f0) * 32767).astype("<i2").tobytes()
            step = int(0.08 * RATE) * 2
            for k in range(0, len(x), step):
                await ws.send_bytes(x[k:k + step])
                sent += 0.08
                await asyncio.sleep(max(0.0, t_start + sent - time.monotonic()))  # 실제 마이크처럼 실시간 속도로
        await ws.send_json({"type": "stop"})
        while True:
            msg = await ws.receive()
            if msg.type.name == "TEXT" and json.loads(msg.data).get("type") == "recorded":
                break
        await ws.close()
    return sid


# 본 시험 ------------------------------------------------------------------------------------------------------------

def run(T: Path) -> None:
    sb = make_sandbox(T)
    env, ops = sb["env"], sb["ops"]
    for p in (STUDIO_PORT, TR_PORT):
        if not port_free(p):
            raise SystemExit(f"포트 {p} 사용 중")
    procs = []
    try:
        print("1) 튜터 녹음 스튜디오: 순차 녹음 (튜터 → 학생 → 튜터 → 학생, 실시간 10초)", flush=True)
        procs.append(subprocess.Popen([str(PY_CORE), "-m", "studio.server", "--port", str(STUDIO_PORT)], cwd=EN, env=env,
                                      stdout=open(T / "studio.log", "wb"), stderr=subprocess.STDOUT))
        wait_port(STUDIO_PORT)
        sid = asyncio.run(record_session(sb))
        meta = json.loads((T / "queue" / "tutor" / sid / "meta.json").read_text(encoding="utf-8"))
        check("녹음 저장", meta["status"] == "recorded" and abs(meta["duration_s"] - 10) < 0.6, f"{meta['duration_s']}초")
        check("차례 기록", [t["role"] for t in meta["turns"]] == ["tutor", "student", "tutor", "student"],
              str([(t["role"], t["start_s"]) for t in meta["turns"]]))
        check("두 트랙 시계 차이", all(abs(t["drift_s"]) <= 0.5 for t in meta["tracks"].values()),
              str({r: t["drift_s"] for r, t in meta["tracks"].items()}))

        print("2) 학습 작업 계획 (GPU 없이)", flush=True)
        out = subprocess.run([str(PY_CORE), "-m", "finetune.job", "plan"], cwd=EN, env=env, capture_output=True, text=True, check=True)
        plan = json.loads(out.stdout)
        check("타임스탬프 대기 세션", plan["words_pending"]["studio"] == [sid])

        print("3) GPU 예약: gpu-wait start train --once (가짜 nvidia-smi, 기록 2초) → gpu-mode train → finetune-job", flush=True)
        (ops / "run" / "gpu-wait.after").write_text("realtime\n")  # 포털이 학습 예약 때 남기는 이전 예약
        subprocess.run([str(ops / "gpu-wait"), "start", "train", "--once", "--window", "2", "--interval", "1", "--stable", "2"],
                       env=env, check=True, capture_output=True)
        job_p = T / "runs" / "job_status.json"
        t0, seen = time.time(), set()
        while time.time() - t0 < 600:
            st = json.loads(job_p.read_text(encoding="utf-8")) if job_p.is_file() else {}
            if st.get("phase"):
                seen.add(st["phase"])
            if st.get("result") in ("trained", "prepared", "failed"):
                break
            time.sleep(1)
        print(f"   작업 단계: {sorted(seen)}  결과: {st.get('result')}  {st.get('detail', '')}", flush=True)
        check("학습 작업 완료", st.get("result") == "trained", st.get("detail", ""))
        check("작업 단계 모두 실행", list(st.get("phases", {})) == ["words", "prepare", "train"], str(list(st.get("phases", {}))))

        d = T / "queue" / "tutor" / sid
        words = json.loads((d / "tutor_words.json").read_text(encoding="utf-8"))
        raw = json.loads((d / "tutor_words.raw.json").read_text(encoding="utf-8"))
        tutor_turns = [t for t in meta["turns"] if t["role"] == "tutor"]
        inside = all(any(t["start_s"] - 0.05 <= w["start"] and w["end"] <= t["end_s"] + 0.05 for t in tutor_turns) for w in words)
        check("튜터 단어 타임스탬프", len(words) >= 8 and raw["error"] is None, f"{len(words)}단어, 원본 형식 키 {sorted(raw)}")
        check("단어가 튜터 차례 안에만 있음", inside)

        prep = st["phases"]["prepare"]
        check("학습 데이터 변환·검사", prep["studio"]["items"] == 1 and prep["check"]["ok"], json.dumps(prep["check"], ensure_ascii=False))
        tr = st["phases"]["train"]
        log = Path(tr["log"]).read_text(encoding="utf-8")
        check("moshi-finetune 실제 데이터 로더 통과", "data check: codes (1, 17," in log, log.strip().splitlines()[0][:120])
        check("LoRA 체크포인트", Path(tr["checkpoint"]["lora"]).is_file() and Path(tr["checkpoint"]["config"]).is_file())
        cfg_used = yaml.safe_load(Path(tr["config"]).read_text(encoding="utf-8"))
        check("학습 설정: 새 run_dir, 외부 기록 없음", cfg_used["run_dir"].startswith(str(T / "runs")) and "wandb" not in cfg_used)

        print("4) 작업 끝 → GPU 반환과 이전 예약 복구", flush=True)
        t0 = time.time()
        while time.time() - t0 < 30 and (ops / "run" / "mode").exists():
            time.sleep(0.5)
        check("gpu-mode stop (모드 해제)", not (ops / "run" / "mode").exists())
        t0 = time.time()
        status = ""
        while time.time() - t0 < 20:
            status = subprocess.run([str(ops / "gpu-wait"), "status"], env=env, capture_output=True, text=True).stdout
            if "run realtime" in status:
                break
            time.sleep(0.5)
        check("이전 예약(realtime) 다시 걸림", "run realtime" in status, status.splitlines()[0] if status else "")
        subprocess.run([str(ops / "gpu-wait"), "stop"], env=env, capture_output=True)

        print("5) 적용 (사람이 실행) → en-solo Moshi 명령에 LoRA 인자", flush=True)
        out = subprocess.run([str(PY_CORE), "-m", "finetune.promote", "set", "latest", "--by", "e2e"], cwd=EN, env=env,
                             capture_output=True, text=True)
        check("promote set", out.returncode == 0, out.stderr.strip()[-200:])
        out = subprocess.run(["bash", str(OPS / "services" / "en-moshi-bf16.sh")], env=env | {"EVA_MOSHI_PRINT_CMD": "1"},
                             capture_output=True, text=True)
        cmd = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr
        check("moshi.server --lora-weight", "--lora-weight" in cmd and str(T / "runs") in cmd and "--config-path" in cmd, cmd[-160:])

        print("6) 포털 흐름 표시", flush=True)
        sys.path.insert(0, str(OPS / "portal_app"))
        os.environ.update(EVA_CONFIG=str(sb["cfg"]))
        import server as portal  # noqa: E402

        async def fake_run(argv, timeout=30):
            # 임시 ops 에는 gpu-mode·gpu-wait 만 있다. 나머지(cpu-demo, studio 등)는 '꺼짐'으로 답한다
            if Path(argv[0]).name == "bash" or not Path(argv[0]).exists():
                return 1, ""
            return await portal.run_cmd(argv, timeout)
        os.environ["PATH"] = env["PATH"]
        p = portal.Portal(runner=fake_run, ops=ops, probe=lambda _p: False)
        stt = asyncio.run(p.status(fresh=True))
        flow = {n["id"]: n for s in stt["flow"] for n in s["nodes"]}
        check("포털: 타임스탬프 완료", flow["ts"]["state"] == "cpu_ready", flow["ts"]["detail"])
        check("포털: 학습 완료", flow["train"]["state"] == "cpu_ready", flow["train"]["detail"])
        check("포털: 적용 중", flow["apply"]["state"] == "running", flow["apply"]["detail"])

        print("7) 동의 철회: 스튜디오 삭제 → 녹음과 파생 학습 데이터 삭제", flush=True)
        import urllib.request
        req = urllib.request.Request(f"http://127.0.0.1:{STUDIO_PORT}/api/sessions/{sid}", method="DELETE")
        res = json.loads(urllib.request.urlopen(req).read())
        left = [ln for ln in (T / "data" / "studio.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
        check("삭제 반영", res["derived_removed"] == 1 and not left and not d.exists())
    finally:
        # 학습 작업이 끝나면(실패해도) 3초 뒤 분리된 프로세스가 gpu-mode stop 과 이전 예약 복구를 한다.
        # 그보다 먼저 정리하면 복구된 임시 gpu-wait 가 남으므로, 작업이 시작됐다면 복구가 끝날 때까지 기다린 뒤 멈춘다
        if (T / "runs" / "job_status.json").exists():
            t0 = time.time()
            rel = ops / "logs" / "finetune-release.log"
            while time.time() - t0 < 20 and ((ops / "run" / "mode").exists() or not rel.exists()):
                time.sleep(0.5)
            time.sleep(2)
        subprocess.run([str(ops / "gpu-wait"), "stop"], env=env, capture_output=True)
        subprocess.run([str(ops / "gpu-mode"), "stop"], env=env, capture_output=True)
        for pr in procs:
            pr.terminate()
            pr.wait(10)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="임시 폴더를 지우지 않음 (디버깅)")
    a = ap.parse_args()
    T = Path(tempfile.mkdtemp(prefix="va_e2e_"))
    print(f"전체 흐름 시험 (임시 폴더 {T})", flush=True)
    t0 = time.time()
    try:
        run(T)
        ok = True
    except Exception as e:  # noqa: BLE001  어떤 예외든 로그를 보여 주고 실패로 끝낸다
        import traceback
        traceback.print_exc()
        ok = False
        print(f"\n중단: {e}", flush=True)
        for lp in sorted(T.rglob("*.log")):
            print(f"--- {lp.relative_to(T)} (끝 15줄)")
            print("\n".join(lp.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]))
    passed = sum(1 for _, o, _ in RESULTS if o)
    print(f"\n결과: {passed}/{len(RESULTS)} 통과, {time.time() - t0:.0f}초 {'(성공)' if ok else '(실패)'}")
    if ok and not a.keep:
        shutil.rmtree(T, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
