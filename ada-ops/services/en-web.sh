#!/usr/bin/env bash
# 영어 Agent 화면 (http://127.0.0.1:7101). 접속은 SSH 터널로 한다 (english-agent/README.md).
set -euo pipefail
EVA=/home/yang/pgm/voice_agent/english-agent
cd "$EVA"
export CUDA_VISIBLE_DEVICES=""
exec "$EVA/.venv-core/bin/python" -m web.server
