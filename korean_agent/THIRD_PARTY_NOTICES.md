# 외부 구성요소와 라이선스

2026-11-13 기준으로 서버에 설치된 버전과 로컬 모델 카드에서 확인한 값이다. "확인 필요"는 서버 안에서 라이선스 문서를 찾지 못한 항목으로, 배포 전에 원 저장소에서 확인해야 한다.
모델 가중치는 저장소에 포함하지 않고, 서버의 Hugging Face 캐시 또는 `~/pgm/Korean_interface/models/` 사본을 읽는다.

## 모델

| 구성요소 | 용도 | 라이선스 | 근거 |
|---|---|---|---|
| google/gemma-4-E4B-it | 대화 LLM (vLLM) | Apache-2.0 (Gemma 4 라이선스 안내: https://ai.google.dev/gemma/docs/gemma_4_license) | 캐시된 모델 카드 |
| nvidia Nemotron 3.5 ASR Streaming 0.6B | 음성 인식 (CPU) | OpenMDW 1.1 (`license: other`, https://openmdw.ai/license/1-1/) | 로컬 사본 README.md |
| Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice | 음성 합성 (주 경로) | Apache-2.0 | 로컬 사본 README.md |
| Supertonic-3 | 음성 합성 (CPU 대체 경로, 시험용) | OpenRAIL (사용 제한 조항 포함, LICENSE 파일 동봉) | 로컬 사본 README.md, LICENSE |
| pipecat-ai/smart-turn-v3 (smart-turn-v3.2-cpu.onnx) | 턴 종료 판정 | 확인 필요 (캐시에 모델 카드 없음) | - |
| Silero VAD | 음성 구간 검출 | 확인 필요 (speech-to-speech 가 불러오는 경로와 버전 확인) | speech_to_speech/VAD/vad_iterator.py 주석 |

Supertonic-3 은 OpenRAIL 계열이라 용도 제한 조항이 있다. 운영 경로에 쓰려면 법무 검토가 필요하다.

## 소프트웨어 (`.venv`, Python 3.11)

| 패키지 | 버전 | 라이선스 |
|---|---|---|
| speech-to-speech | 1.1.0 | Apache-2.0 |
| transformers | 5.15.1 (고정) | Apache-2.0 |
| torch | 2.15.0+cu132 | BSD-3-Clause 계열 (복합) |
| faster-qwen3-tts | 0.5.4 | MIT |
| qwen-tts-hf | 0.1.1.post1 | Apache-2.0 |
| qwentts-cpp-python | 0.5.0 | MIT |
| supertonic | 1.3.1 | MIT |
| onnxruntime | 1.31.1 | MIT |
| fastapi | 0.162.5 | MIT |
| uvicorn | 0.54.0 | BSD-3-Clause |
| websockets | 17.4 | BSD-3-Clause |
| numpy | 2.4.6 | BSD-3-Clause 외 |
| soundfile | 0.14.0 | BSD-3-Clause |
| PyYAML | 6.0.3 | MIT |

## 소프트웨어 (`.venv-vllm`)

| 패키지 | 버전 | 라이선스 |
|---|---|---|
| vllm | 0.33.0 | Apache-2.0 |
| transformers | 5.20.2 | Apache-2.0 |
| torch | 2.14.0+cu130 | BSD-3-Clause 계열 (복합) |

## 외부 전송 관련 설정

- vLLM 사용 통계 전송을 끈다 (`VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`, `deploy/run_gemma_vllm.py`).
- 서비스 실행 중에는 `HF_HUB_OFFLINE=1` 로 모델 저장소에 접속하지 않는다.
- 모든 서비스는 127.0.0.1 에만 열린다. 화면은 SSH 터널로 접속한다 (README.md).
