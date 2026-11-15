"""Gemma 4 vLLM 서버 실행 (ada-ops/services/ko-gemma-vllm.sh 가 .venv-vllm 의 python 으로 부른다).

config/services.yaml 의 llm 절과 ports.gemma_vllm 으로 명령을 만들고 vllm 으로 실행을 넘긴다.
--gpu-memory-utilization 은 GPU 전체 메모리에 대한 비율이다. ada-ops/gpu_layout.md 와 같이 고친다.
확인만 할 때: python deploy/run_gemma_vllm.py --print
"""

import os
import shlex
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def build() -> list[str]:
    cfg = yaml.safe_load((ROOT / "config" / "services.yaml").read_text(encoding="utf-8"))
    llm = cfg["llm"]
    vllm = str(Path(sys.executable).parent / "vllm")
    argv = [vllm, "serve", llm["model"],
            "--served-model-name", llm["served_name"],
            "--host", "127.0.0.1", "--port", str(cfg["ports"]["gemma_vllm"]),
            "--gpu-memory-utilization", str(llm["gpu_memory_utilization"]),
            "--max-model-len", str(llm["max_model_len"])]
    if llm.get("quantization") and llm["quantization"] not in ("none", "bf16"):
        argv += ["--quantization", str(llm["quantization"])]
    argv += [str(a) for a in llm.get("vllm_extra_args") or []]
    return argv


if __name__ == "__main__":
    argv = build()
    if "--print" in sys.argv:
        print(shlex.join(argv))
        sys.exit(0)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 모델은 이미 캐시에 있다. 실행 중 외부 다운로드를 하지 않는다
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")  # 사용 통계 전송 끔 (외부 전송 금지)
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.execv(argv[0], argv)
