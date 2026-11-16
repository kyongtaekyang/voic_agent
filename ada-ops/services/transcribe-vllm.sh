#!/usr/bin/env bash
# 후처리 모드: Nemotron 3.5 Transcribe 30B (vLLM, 127.0.0.1:8002). gpu-mode batch 가 시작한다.
# **미검증 (2026-11-16):** 가중치와 NeMo Speech vLLM 플러그인(nemo_speechlm)이 아직 설치되지 않았다.
# 옵션은 docs/Nemotron_3.5_Transcribe.md(모델 카드 기준, vLLM 0.32.0)를 따랐고, 설치된 vLLM 은 0.33.0 이다.
# Ada 48GB 에서는 BF16(약 68GB)을 올릴 수 없으므로 FP8 로 띄운다. 처음 띄울 때 --help 와 모델 카드로 옵션을 다시 확인하고
# 결과를 korean_agent/docs/version_check.md 에 기록할 것.
set -euo pipefail
KVA=/home/yang/pgm/voice_agent/korean_agent
MODEL="${KVA_TRANSCRIBE_MODEL:-/home/yang/pgm/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B}"
PY="$KVA/.venv-vllm/bin/python"
[[ -d "$MODEL" ]] || { echo "가중치 없음: $MODEL (KVA_TRANSCRIBE_MODEL 로 지정)"; exit 1; }
"$PY" -c "import nemo.collections.speechlm2" 2>/dev/null || { echo "NeMo Speech 플러그인 미설치 (.venv-vllm)"; exit 1; }
export VLLM_PLUGINS=nemo_speechlm
export VLLM_MAX_AUDIO_DECODE_DURATION_S=3600
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180
export VLLM_USE_V2_MODEL_RUNNER=0
exec "$KVA/.venv-vllm/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port 8002 \
  --served-model-name nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
  --trust-remote-code --dtype bfloat16 --quantization fp8 --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --max-num-batched-tokens 4096 --max-num-seqs 8 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 --no-async-scheduling \
  --limit-mm-per-prompt '{"audio":{"count":1,"length":120000000}}' \
  --reasoning-parser nemotron_v3 --trust-request-mm-kwargs \
  --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware
