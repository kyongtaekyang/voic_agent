#!/usr/bin/env bash
# 영어 Agent: Moshi PyTorch BF16 서버 (http://127.0.0.1:8998, WebSocket /api/chat). 영어 단독 프로필(en-solo)용.
# 1단계 기능 검증용이다. GPU 약 24GB 이상이 필요하므로 한국어 Agent 를 끈 상태에서만 쓴다 (프롬프트 2.1절).
# 모델 선택: EVA_MOSHI_REPO=kyutai/moshiko-pytorch-bf16 (남성, 기본) | kyutai/moshika-pytorch-bf16 (여성)
# 튜터 말투 LoRA: finetune/active_lora.json 이 있으면 --lora-weight/--config-path 를 붙인다 (python -m finetune.promote 로 적용·해제).
#   LoRA 는 이 BF16 서버에만 적용된다. realtime 의 INT8(en-moshi.sh)은 int8 가중치에 LoRA 를 합칠 수 없다.
# 접속은 SSH 터널로 한다 (english-agent/README.md). --host 를 0.0.0.0 으로 바꾸지 않는다.
# EVA_MOSHI_PRINT_CMD=1 이면 실행하지 않고 명령만 출력한다 (ada-ops/e2e 시험용).
set -euo pipefail
VA_ROOT="${VA_ROOT:-/home/yang/pgm/voice_agent}"
EVA="$VA_ROOT/english-agent"
cd "$EVA"
LORA_ARGS="$("$EVA/.venv-core/bin/python" -m finetune.promote show-args)"
CMD=("$EVA/.venv/bin/python" -m moshi.server --host 127.0.0.1 --port 8998 --hf-repo "${EVA_MOSHI_REPO:-kyutai/moshiko-pytorch-bf16}")
# shellcheck disable=SC2206  # show-args 는 공백 없는 경로만 낸다 (finetune/runs/lora_<시각>/...)
[[ -n "$LORA_ARGS" ]] && { CMD+=($LORA_ARGS); echo "[en-moshi-bf16] 튜터 말투 LoRA 적용: $LORA_ARGS"; }
if [[ "${EVA_MOSHI_PRINT_CMD:-0}" == 1 ]]; then echo "${CMD[*]}"; exit 0; fi
exec "${CMD[@]}"
