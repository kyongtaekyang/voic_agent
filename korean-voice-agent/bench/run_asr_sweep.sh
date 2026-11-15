#!/usr/bin/env bash
# Stage 1 ASR CPU RTF sweep. Uses Korean_interface venv read-only (never pip install into it).
cd "$(dirname "$0")/.."
PY=/home/yang/pgm/Korean_interface/.venv-linux/bin/python
M=/home/yang/pgm/Korean_interface/models/stt/nemotron-3.5-asr-streaming-0.6b
for T in 4 8; do
  echo "=== threads=$T start $(date -Is) | $(uptime)"
  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 $PY bench/asr_cpu_rtf.py --model $M --threads $T \
    --lookahead 0 3 6 13 $(for f in bench/data/ko_ref/*.wav; do printf -- "--wav %s " $f; done) --out docs/reports/asr_cpu_rtf_t$T.json 2>&1 \
    | grep -vE 'torch_dtype|Loading weights|max_length|warnings.warn'
  echo "=== threads=$T end $(date -Is) | $(uptime)"
done
echo SWEEP_DONE
