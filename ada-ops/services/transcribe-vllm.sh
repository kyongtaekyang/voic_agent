#!/usr/bin/env bash
# 후처리 모드: Nemotron 3.5 Transcribe 30B (vLLM, 127.0.0.1:8002). gpu-mode batch(야간 배치)와 학습 작업(finetune-job)이 띄운다.
# 2026-11-19: 모델 카드 Environment Setup 절차대로 전용 환경을 만들었다 (ada-ops/setup-gpu-envs transcribe).
#   - ~/opt/nemo-speech/.venv : NeMo Speech main + vLLM 0.32.0 (uv.lock 고정). korean_agent/.venv-vllm(vLLM 0.33)과 별개
#   - 가중치: NVIDIA 가 낸 FP8 체크포인트 (Ada 48GB 에서 BF16 약 68GB 는 올릴 수 없음). 온라인 --quantization fp8 대신 공식 FP8 사용
#   - --middleware ctc_timestamp_middleware: 단어 타임스탬프 (english-agent/slow_path/words.py 가 씀)
# **GPU 에서 아직 띄워 보지 않았다.** 처음 띄울 때 메모리(--gpu-memory-utilization 은 비율이라 다른 사용자 사용량에 따라 실패할 수 있음)와
# 옵션을 확인하고 korean_agent/docs/version_check.md 에 기록할 것. KVA_TRANSCRIBE_MODEL, NEMO_SPEECH_VENV 로 바꿀 수 있다.
set -euo pipefail
MODEL="${KVA_TRANSCRIBE_MODEL:-/home/yang/pgm/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8}"
VENV="${NEMO_SPEECH_VENV:-/home/yang/opt/nemo-speech/.venv}"
[[ -d "$MODEL" ]] || { echo "가중치 없음: $MODEL (ada-ops/setup-gpu-envs transcribe)"; exit 1; }
"$VENV/bin/python" -c "import nemo.collections.speechlm2.vllm.salm.ctc_serving" 2>/dev/null || { echo "NeMo Speech 환경 없음: $VENV (ada-ops/setup-gpu-envs transcribe)"; exit 1; }
export VLLM_PLUGINS=nemo_speechlm
export VLLM_MAX_AUDIO_DECODE_DURATION_S=3600
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180
export VLLM_USE_V2_MODEL_RUNNER=0     # 단어 타임스탬프는 V1 모델 러너가 필요 (모델 카드)
export HF_HUB_OFFLINE=1
exec "$VENV/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port 8002 \
  --served-model-name nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
  --trust-remote-code --gpu-memory-utilization "${TRANSCRIBE_GPU_UTIL:-0.80}" \
  --max-model-len 32768 --max-num-batched-tokens 4096 --max-num-seqs 8 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 --no-async-scheduling \
  --limit-mm-per-prompt '{"audio":{"count":1,"length":120000000}}' \
  --reasoning-parser nemotron_v3 --trust-request-mm-kwargs \
  --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware
