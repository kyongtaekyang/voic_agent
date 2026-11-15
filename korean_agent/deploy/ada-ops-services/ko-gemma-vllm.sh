#!/usr/bin/env bash
# 한국어 Agent: Gemma 4 vLLM (127.0.0.1:8001). gpu-mode realtime 이 시작한다. 설정은 korean_agent/config/services.yaml
set -euo pipefail
KVA=/home/yang/pgm/voice_agent/korean_agent
cd "$KVA"
exec "$KVA/.venv-vllm/bin/python" deploy/run_gemma_vllm.py
