"""튜터 말투 LoRA 학습 작업: 단어 타임스탬프 → 학습 데이터 변환·검사 → moshi-finetune 학습 → 보고 (프롬프트 3절, 8단계).

학습 모드 서비스(ada-ops/services/finetune-job.sh)로 실행된다. GPU 시간은 gpu-wait 예약(--once)으로만 얻는다.

  python -m finetune.job run            # 전체 (서비스가 부름)
  python -m finetune.job plan           # GPU 없이 무엇을 할지만 출력
  python -m finetune.job status         # 마지막 작업 상태

단계
1. words   : 타임스탬프가 필요한 세션이 있을 때만 Transcribe 30B 서버(services/transcribe-vllm.sh)를 이 작업의 자식으로 띄우고,
             slow_path.words 로 tutor_words.json / agent.words 를 만든 뒤 서버를 내린다 (같은 프로세스 그룹이라 gpu-mode 가 우리 것으로 셈)
2. prepare : finetune.prepare from-studio, from-queue, check (CPU)
3. train   : 학습 데이터가 finetune.min_items 이상일 때만. moshi_lora.yaml 을 바탕으로 이번 실행 설정을 runs/<run>/config.yaml 에 쓰고
             ~/opt/moshi-finetune/.venv 의 torchrun 으로 학습한다. moshi-finetune 은 이어서 학습하지 못하고 run_dir 가 있으면 멈추므로
             실행마다 새 run_dir 를 쓴다.
4. report  : runs/<run>/report.json (단계 결과, 마지막 체크포인트, 손실). **적용(promote)은 사람이 따로 한다** (finetune/promote.py).
끝나면 GPU 를 돌려준다: gpu-mode 가 띄운 경우(FT_RELEASE=1) 분리된 프로세스로 `gpu-mode stop` 을 부르고,
run/gpu-wait.after 에 적힌 예약(예: realtime)을 다시 건다.

시험용 환경 변수 (ada-ops/e2e): FT_TRANSCRIBE_CMD, FT_TRAIN_CMD, FT_MIN_ITEMS, FT_CONFIG_OVERRIDES(JSON), FT_OPS
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

from finetune import prepare as fp
from slow_path import words as sw

ROOT = Path(__file__).resolve().parent.parent          # english-agent
VA_ROOT = Path(os.environ.get("VA_ROOT") or ROOT.parent)  # voice_agent
STEP_RE = re.compile(r"\bstep[:=]?\s*(\d+)", re.I)
LOSS_RE = re.compile(r"\bloss[:=]?\s*([0-9.]+)", re.I)


def ft_cfg(cfg_all: dict) -> dict:
    f = dict(cfg_all.get("finetune") or {})
    f.setdefault("min_items", 10)
    f.setdefault("data_dir", (cfg_all.get("studio") or {}).get("finetune_out") or "finetune/data")
    f.setdefault("runs_dir", "finetune/runs")
    f.setdefault("job_status", "finetune/runs/job_status.json")
    f.setdefault("active_lora", "finetune/active_lora.json")
    f.setdefault("base_config", "finetune/moshi_lora.yaml")
    f.setdefault("moshi_finetune_dir", str(Path.home() / "opt" / "moshi-finetune"))
    f.setdefault("transcribe_cmd", str(VA_ROOT / "ada-ops" / "services" / "transcribe-vllm.sh"))
    f.setdefault("transcribe_ready_timeout_s", 1500)
    return f


class Job:
    def __init__(self, cfg_all: dict | None = None):
        self.all = cfg_all if cfg_all is not None else sw.load_all()
        self.f = ft_cfg(self.all)
        self.sp, self.tq, self.eq, self.allow_child = sw.paths(self.all)
        self.url = self.sp.get("transcribe_url", "http://127.0.0.1:8002")
        self.data = sw.rel(self.f["data_dir"])
        self.runs = sw.rel(self.f["runs_dir"])
        self.status_p = sw.rel(self.f["job_status"])
        self.min_items = int(os.environ.get("FT_MIN_ITEMS") or self.f["min_items"])
        self.state: dict = {}

    # 상태 ----------------------------------------------------------------------------------------------------------
    def save(self, **kw) -> None:
        self.state.update(kw, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.status_p.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.status_p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.status_p)

    def plan(self) -> dict:
        return {"words_pending": {"studio": sw.studio_pending(self.tq, self.allow_child), "queue": sw.queue_pending(self.eq)},
                "min_items": self.min_items, "data_dir": str(self.data), "runs_dir": str(self.runs)}

    # 1) 단어 타임스탬프 ------------------------------------------------------------------------------------------------
    def phase_words(self) -> dict:
        pend = self.plan()["words_pending"]
        if not pend["studio"] and not pend["queue"]:
            return {"skipped": "타임스탬프가 필요한 세션 없음"}
        cmd = shlex.split(os.environ.get("FT_TRANSCRIBE_CMD") or self.f["transcribe_cmd"])
        log_p = self.run_dir / "transcribe.log"
        self.save(phase="words", detail=f"Transcribe 서버 기동 (세션 {len(pend['studio']) + len(pend['queue'])}개)")
        with log_p.open("ab") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            self.wait_ready(proc, float(self.f["transcribe_ready_timeout_s"]))
            self.save(detail="단어 타임스탬프 생성 중")
            res = {"studio": sw.studio_stage(self.tq, self.sp, self.url, self.allow_child),
                   "queue": sw.queue_stage(self.eq, self.sp, self.url)}
        finally:
            stop_proc(proc)
        return res

    def wait_ready(self, proc: subprocess.Popen, timeout: float) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if proc.poll() is not None:
                raise RuntimeError(f"Transcribe 서버가 종료됐습니다 (code {proc.returncode}). 로그: {self.run_dir / 'transcribe.log'}")
            try:
                with requests.Session() as s:
                    s.trust_env = False
                    if s.get(f"{sw.local_only(self.url)}/v1/models", timeout=3).status_code == 200:
                        return
            except requests.RequestException:
                pass
            time.sleep(2)
        raise RuntimeError(f"Transcribe 서버가 {timeout:.0f}초 안에 준비되지 않았습니다")

    # 2) 학습 데이터 ---------------------------------------------------------------------------------------------------
    def phase_prepare(self) -> dict:
        self.save(phase="prepare", detail="학습 데이터 변환·검사")
        st = fp.from_studio(self.tq, self.data, self.allow_child)
        qu = fp.from_queue(self.eq, self.data)
        chk = fp.check(self.data)
        return {"studio": st, "queue": qu, "check": chk}

    def manifests(self) -> list[Path]:
        out = []
        for name in ("studio", "queue", "pairs"):
            p = self.data / f"{name}.jsonl"
            if p.is_file() and any(ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()):
                out.append(p)
        return out

    # 3) 학습 -------------------------------------------------------------------------------------------------------
    def render_config(self, manifests: list[Path]) -> Path:
        base = yaml.safe_load(sw.rel(self.f["base_config"]).read_text(encoding="utf-8"))
        base["data"]["train_data"] = ",".join(str(p) for p in manifests)  # moshi-finetune: "경로[:가중치],..." (가중치 생략 = 1)
        base["run_dir"] = str(self.run_dir / "train")                     # 실행마다 새 폴더 (moshi-finetune 은 있으면 멈춤)
        base.pop("wandb", None)                                            # 학습 기록을 외부로 보내지 않음
        over = json.loads(os.environ.get("FT_CONFIG_OVERRIDES") or "{}")
        for k, v in over.items():
            base[k] = v
        p = self.run_dir / "config.yaml"
        p.write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return p

    def phase_train(self, items: int) -> dict:
        if items < self.min_items:
            return {"skipped": f"학습 데이터 {items}개 < 최소 {self.min_items}개 (finetune.min_items)"}
        mans = self.manifests()
        cfg = self.render_config(mans)
        mf = Path(self.f["moshi_finetune_dir"])
        cmd = (shlex.split(os.environ["FT_TRAIN_CMD"]) + [str(cfg)] if os.environ.get("FT_TRAIN_CMD")
               else [str(mf / ".venv" / "bin" / "torchrun"), "--nproc-per-node", "1", "-m", "train", str(cfg)])
        max_steps = int(yaml.safe_load(cfg.read_text(encoding="utf-8")).get("max_steps") or 0)
        self.save(phase="train", detail="학습 시작", step=0, max_steps=max_steps)
        log_p = self.run_dir / "train.log"
        last_step, last_loss = 0, None
        env = dict(os.environ, WANDB_MODE="disabled", HF_HUB_OFFLINE=os.environ.get("HF_HUB_OFFLINE", "1"))
        with log_p.open("w", encoding="utf-8") as lf:
            proc = subprocess.Popen(cmd, cwd=str(mf), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    text=True, errors="replace", env=env)
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                m = STEP_RE.search(line)
                if m:
                    last_step = int(m.group(1))
                    lm = LOSS_RE.search(line)
                    last_loss = float(lm.group(1)) if lm else last_loss
                    self.save(detail=f"학습 중 {last_step}/{max_steps}", step=last_step, loss=last_loss)
            rc = proc.wait()
        ckpt = latest_checkpoint(self.run_dir / "train")
        res = {"exit_code": rc, "config": str(cfg), "log": str(log_p), "manifests": [str(p) for p in mans],
               "last_step": last_step, "last_loss": last_loss, "checkpoint": ckpt}
        if rc != 0:
            raise RuntimeError(f"학습이 실패했습니다 (code {rc}). 로그: {log_p}")
        if not ckpt:
            raise RuntimeError(f"학습은 끝났지만 LoRA 체크포인트(consolidated/lora.safetensors)가 없습니다: {self.run_dir / 'train'}")
        return res

    # 전체 ----------------------------------------------------------------------------------------------------------
    def run(self) -> int:
        run_id = time.strftime("lora_%Y%m%d-%H%M%S")
        self.run_dir = self.runs / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.state = {"run": run_id, "run_dir": str(self.run_dir), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "result": "running", "phases": {}}
        self.save(phase="start", detail="시작")
        code = 0
        try:
            self.state["phases"]["words"] = self.phase_words()
            prep = self.phase_prepare()
            self.state["phases"]["prepare"] = prep
            if not prep["check"]["ok"]:
                raise RuntimeError("학습 데이터 검사 실패: " + "; ".join(prep["check"]["problems"][:5]))
            self.state["phases"]["train"] = self.phase_train(prep["check"]["items"])
            trained = "checkpoint" in self.state["phases"]["train"]
            self.save(phase="done", result="trained" if trained else "prepared",
                      detail="학습 완료. 적용은 사람이 확인한 뒤 finetune.promote 로 합니다" if trained
                      else self.state["phases"]["train"].get("skipped", "학습 안 함"))
        except Exception as e:  # noqa: BLE001  상태 파일에 남기고 GPU 를 돌려준다
            code = 1
            self.save(phase="failed", result="failed", detail=str(e)[:500])
        finally:
            (self.run_dir / "report.json").write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")
            release_gpu()
        return code


def latest_checkpoint(train_dir: Path) -> dict | None:
    cks = sorted((train_dir / "checkpoints").glob("checkpoint_*/consolidated/lora.safetensors")) if train_dir.is_dir() else []
    if not cks:
        return None
    lora = cks[-1]
    conf = lora.parent / "config.json"
    return {"lora": str(lora), "config": str(conf) if conf.is_file() else None, "step": int(lora.parent.parent.name.split("_")[-1])}


def stop_proc(proc: subprocess.Popen, wait_s: float = 60) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(wait_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def release_gpu() -> None:
    """gpu-mode 가 띄운 작업이면(FT_RELEASE=1) 끝난 뒤 모드를 내리고 이전 예약을 다시 건다.
    gpu-mode stop 은 이 작업의 프로세스 그룹을 멈추므로, 새 세션으로 분리해서 이 프로세스가 끝난 뒤에 돌게 한다."""
    if os.environ.get("FT_RELEASE") != "1":
        return
    ops = Path(os.environ.get("FT_OPS") or VA_ROOT / "ada-ops")
    after = ops / "run" / "gpu-wait.after"
    rearm = ""
    if after.is_file():
        mode = after.read_text().strip()
        if re.fullmatch(r"[\w-]+", mode or ""):
            rearm = f'; "{ops}/gpu-wait" start {mode}'
        after.unlink(missing_ok=True)
    script = f'sleep 3; "{ops}/gpu-mode" stop{rearm}'
    with (ops / "logs" / "finetune-release.log").open("ab") as lf:
        subprocess.Popen(["bash", "-c", script], stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "plan", "status"])
    a = ap.parse_args(argv)
    job = Job()
    if a.cmd == "plan":
        print(json.dumps(job.plan(), ensure_ascii=False, indent=1))
        return 0
    if a.cmd == "status":
        print(job.status_p.read_text(encoding="utf-8") if job.status_p.is_file() else "{}")
        return 0
    return job.run()


if __name__ == "__main__":
    sys.exit(main())
