#!/usr/bin/env bash
# Project venv for korean_agent (python3.11). Pinned to speech-to-speech 1.1.0.
set -e
cd "$(dirname "$0")/.."
python3.11 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install 'speech-to-speech[supertonic]==1.1.0'
.venv/bin/python -m pip install 'transformers==5.15.1'  # qwen-tts-hf 0.1.1.post1 fails on 5.18+ (docs/version_check.md 2.1)
.venv/bin/python -m pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12  # qwentts-cpp ggml CUDA lib is built for CUDA 12; add both lib dirs to LD_LIBRARY_PATH
.venv/bin/python -m pip freeze > docs/reports/venv_freeze.txt
echo VENV_DONE
