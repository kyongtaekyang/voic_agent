#!/usr/bin/env bash
# 영어 Agent Core (Gateway, ws://127.0.0.1:7102/session). Moshi(:8998) 와 화면(:7101) 사이에서 안전 감시·ASR·녹음을 한다.
# GPU 를 쓰지 않는다 (ASR 은 CPU). 설정: english-agent/config/services.yaml (EVA_CONFIG 로 바꿀 수 있음)
set -euo pipefail
EVA=/home/yang/pgm/voice_agent/english-agent
cd "$EVA"
export CUDA_VISIBLE_DEVICES=""
exec "$EVA/.venv-core/bin/python" -m gateway.server   # .venv-core: moshi 와 패키지 버전이 충돌해서 분리 (bench/setup_venvs.sh)
