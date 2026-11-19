"""Moshi GPU 실측: 메모리와 80 ms 프레임 처리 시간 (프롬프트 2단계 KPI, en-moshi INT8 활성화 판단용).

  # GPU 창이 열렸을 때 (우리 서비스를 모두 내린 상태에서). english-agent/.venv 사용
  .venv/bin/python -m bench.moshi_gpu_measure --repo kyutai/moshiko-pytorch-q8 --frames 500 --out docs/reports/moshi_q8_gpu.json
  .venv/bin/python -m bench.moshi_gpu_measure --repo kyutai/moshiko-pytorch-bf16 --frames 500 [--lora-weight ... --config-path ...]
  # 코드 확인용 (CPU, 몇 프레임만): --device cpu --frames 3

moshi.server 와 같은 경로로 한 프레임씩 처리한다: mimi.encode(80 ms 무음) → lm_gen.step → mimi.decode.
처음 warmup 프레임(CUDA 커널 준비)은 통계에서 뺀다. 결과: 프레임 시간 p50/p95/p99/최대, 80 ms 초과 비율,
torch 최대 할당 메모리, 측정 전후 nvidia-smi 사용량 차이(프로세스 전체, CUDA 문맥 포함).
결과에 따라 ada-ops/modes.conf 의 en-moshi 필요 메모리를 고치고, 사용자 승인 뒤 services/en-moshi.sh 에 실행 권한을 준다.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


def smi_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        pid, mem = (x.strip() for x in line.split(","))
        if pid == str(os.getpid()):
            return int(mem)
    return 0


def q(xs: list[float], p: float) -> float:
    return round(float(np.percentile(xs, p * 100)), 2) if xs else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="kyutai/moshiko-pytorch-q8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--lora-weight", default=None)
    ap.add_argument("--config-path", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from moshi.models import LMGen, loaders

    from core import moshi_q8
    moshi_q8.install()  # q8 모델의 weight_scb 를 float32 로 다시 불러온다 (moshi 0.2.13 문제)
    dev = torch.device(a.device)
    t_load = time.perf_counter()
    ci = loaders.CheckpointInfo.from_hf_repo(a.repo, lora_weights=a.lora_weight, config_path=a.config_path)
    mimi = ci.get_mimi(device=dev)
    lm = ci.get_moshi(device=dev, dtype=torch.bfloat16)
    load_s = time.perf_counter() - t_load
    lm_gen = LMGen(lm)
    frame = int(mimi.sample_rate / mimi.frame_rate)
    mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for i in range(a.warmup + a.frames):
            x = torch.from_numpy((rng.standard_normal(frame) * 1e-3).astype(np.float32)).to(dev)[None, None]
            sync()
            t0 = time.perf_counter()
            codes = mimi.encode(x)
            for c in range(codes.shape[-1]):
                tokens = lm_gen.step(codes[:, :, c: c + 1])
                if tokens is not None:
                    mimi.decode(tokens[:, 1:])
            sync()
            if i >= a.warmup:
                times.append((time.perf_counter() - t0) * 1000)
    budget = 1000 / mimi.frame_rate
    res = {
        "repo": a.repo, "device": a.device, "lora": a.lora_weight, "frames": len(times), "warmup": a.warmup,
        "load_s": round(load_s, 1), "frame_budget_ms": budget,
        "frame_ms": {"p50": q(times, .5), "p95": q(times, .95), "p99": q(times, .99), "max": round(max(times), 2) if times else None},
        "over_budget_ratio": round(sum(t > budget for t in times) / max(1, len(times)), 4),
        "torch_max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20) if dev.type == "cuda" else None,
        "nvidia_smi_process_mib": smi_used_mib() if dev.type == "cuda" else None,
        "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else None,
        "torch": torch.__version__, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
