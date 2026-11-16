#!/usr/bin/env bash
# 영어 Agent: Moshi PyTorch BF16 서버 (http://127.0.0.1:8998, WebSocket /api/chat). 영어 단독 프로필(en-solo)용.
# 1단계 기능 검증용이다. GPU 약 24GB 이상이 필요하므로 한국어 Agent 를 끈 상태에서만 쓴다 (프롬프트 2.1절).
# 모델 선택: EVA_MOSHI_REPO=kyutai/moshiko-pytorch-bf16 (남성, 기본) | kyutai/moshika-pytorch-bf16 (여성)
# 접속은 SSH 터널로 한다 (english-agent/README.md). --host 를 0.0.0.0 으로 바꾸지 않는다.
set -euo pipefail
EVA=/home/yang/pgm/voice_agent/english-agent
cd "$EVA"
exec "$EVA/.venv/bin/python" -m moshi.server --host 127.0.0.1 --port 8998 \
  --hf-repo "${EVA_MOSHI_REPO:-kyutai/moshiko-pytorch-bf16}"
