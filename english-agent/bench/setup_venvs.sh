#!/usr/bin/env bash
# 영어 Agent 가상환경 두 개 (python3.11). moshi 0.2.13 은 huggingface-hub<1.0, safetensors<0.8 을 요구하고
# transformers 5.15.1(Nemotron ASR) 은 huggingface-hub 1.x 를 요구해서 한 환경에 둘 수 없다.
#   .venv      : Moshi 서버 전용 (ada-ops/services/en-moshi-bf16.sh). GPU torch.
#   .venv-core : Gateway·화면·ASR·테스트 (ada-ops/services/en-core.sh, en-web.sh). CPU torch. moshi 를 import 하지 않는다.
set -e
cd "$(dirname "$0")/.."
[ -d .venv ] || python3.11 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q 'moshi==0.2.13' 'huggingface-hub<1.0' 'safetensors<0.8' pyyaml
.venv/bin/python -m pip uninstall -q -y transformers tokenizers librosa 2>/dev/null || true
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > logs/venv_freeze.txt

[ -d .venv-core ] || python3.11 -m venv .venv-core
.venv-core/bin/python -m pip install -q --upgrade pip
.venv-core/bin/python -m pip install -q --index-url https://download.pytorch.org/whl/cpu 'torch==2.9.1'
.venv-core/bin/python -m pip install -q 'transformers==5.15.1' librosa 'sphn==0.2.1' aiohttp pyyaml numpy pytest
.venv-core/bin/python -m pip check
.venv-core/bin/python -m pip freeze > logs/venv_core_freeze.txt
echo VENVS_DONE
