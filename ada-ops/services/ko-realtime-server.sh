#!/usr/bin/env bash
# 한국어 Agent: Realtime Voice Server (ws://127.0.0.1:8766/v1/realtime). STT(CPU) + Gemma 연결 + Qwen3-TTS(ggml, GPU).
set -euo pipefail
KVA=/home/yang/pgm/voice_agent/korean-voice-agent
cd "$KVA"
SP="$KVA/.venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1
exec "$KVA/.venv/bin/python" -m realtime_server.main
