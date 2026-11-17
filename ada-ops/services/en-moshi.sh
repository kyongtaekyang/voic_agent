#!/usr/bin/env bash
# 영어 Agent: Moshi PyTorch INT8 서버 (http://127.0.0.1:8998, WebSocket /api/chat). realtime 모드(한국어와 GPU 공유)용.
# 프롬프트 2.1절의 운영 백엔드는 Rust/Candle INT8(config-q8.json)이지만, 서버에 cargo·nvcc(CUDA 툴킷)가 없어 빌드할 수 없다.
# 그래서 Kyutai 가 배포한 PyTorch INT8 가중치(kyutai/moshiko-pytorch-q8, config.json "quantize": true, bitsandbytes int8)를
# 임시 대체로 쓴다. 텍스트 토큰·프로토콜은 en-solo 의 BF16 서버와 같다 (같은 moshi.server).
#
# 이 파일은 일부러 실행 권한 없이 둔다 (chmod -x). gpu-mode·gpu-wait 는 실행 권한이 있는 서비스만 계산·기동하므로,
# 권한을 주는 순간 realtime 의 need 가 15000 → 약 27000 MiB 로 늘고 gpu-wait 판단이 바로 바뀐다.
# GPU 시간대에 메모리·프레임 지연(80ms 예산)을 실측하고 사용자 승인을 받은 뒤 chmod +x 한다.
# 모델 선택: EVA_MOSHI_Q8_REPO=kyutai/moshiko-pytorch-q8 (남성, 기본) | kyutai/moshika-pytorch-q8 (여성)
# 접속은 SSH 터널로 한다 (english-agent/README.md). --host 를 0.0.0.0 으로 바꾸지 않는다.
set -euo pipefail
EVA=/home/yang/pgm/voice_agent/english-agent
cd "$EVA"
# core.moshi_q8: moshi 0.2.13 이 int8 스케일(weight_scb)을 bfloat16 으로 바꿔 첫 프레임에서 멈추는 문제를 고친 뒤 moshi.server 를 실행한다.
exec "$EVA/.venv/bin/python" -m core.moshi_q8 --host 127.0.0.1 --port 8998 \
  --hf-repo "${EVA_MOSHI_Q8_REPO:-kyutai/moshiko-pytorch-q8}"
