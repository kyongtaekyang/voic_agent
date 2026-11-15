#!/usr/bin/env bash
# Separate venv for Gemma 4 vLLM server (keeps the speech-to-speech venv pins untouched).
set -e
cd "$(dirname "$0")/.."
python3.11 -m venv .venv-vllm
.venv-vllm/bin/python -m pip install -q --upgrade pip
.venv-vllm/bin/python -m pip install 'vllm==0.33.0'
.venv-vllm/bin/python -m pip freeze > docs/reports/vllm_freeze.txt
echo VLLM_DONE
