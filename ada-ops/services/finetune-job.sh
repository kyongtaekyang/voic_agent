#!/usr/bin/env bash
# 학습 모드(train): 튜터 말투 LoRA 학습 작업 (english-agent/finetune/job.py). 서버가 아니라 끝나는 작업이다.
#   단어 타임스탬프(Transcribe 30B, 자식 프로세스) → 학습 데이터 변환·검사 → moshi-finetune 학습 → 보고
# gpu-wait start train --once 로 예약해서 띄운다 (포털이 그렇게 부름). 끝나면 job.py 가 분리된 프로세스로 gpu-mode stop 을 부르고
# run/gpu-wait.after 의 이전 예약을 다시 건다. 학습 중에는 양보(yield)하지 않는다 (moshi-finetune 은 이어서 학습하지 못함).
# GPU 를 쓰는 자식(vLLM, torchrun)은 이 프로세스 그룹 안에서 돌므로 gpu-mode 가 우리 사용량으로 센다.
# VA_ROOT 로 저장소 위치를 바꿀 수 있다 (ada-ops/e2e 시험용). CUDA_VISIBLE_DEVICES 를 비우지 않는다 (GPU 작업).
set -euo pipefail
VA_ROOT="${VA_ROOT:-/home/yang/pgm/voice_agent}"
OPS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
export VA_ROOT FT_OPS="${FT_OPS:-$OPS}" FT_RELEASE="${FT_RELEASE:-1}"
cd "$VA_ROOT/english-agent"
exec "$VA_ROOT/english-agent/.venv-core/bin/python" -m finetune.job run
