#!/usr/bin/env bash
# 한국어 Agent 화면 (http://127.0.0.1:7201). 접속은 SSH 터널로 한다 (korean-voice-agent/README.md).
set -euo pipefail
KVA=/home/yang/pgm/voice_agent/korean-voice-agent
cd "$KVA"
exec "$KVA/.venv/bin/python" -m web.server
