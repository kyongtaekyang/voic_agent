# 포트 배정 (영어·한국어 Agent 공통)

포트를 바꾸려면 이 파일을 먼저 고치고, 두 프로그램의 설정을 함께 갱신합니다.

| 서비스 | 규약 포트 | 2026-11-12 확인 결과 |
|---|---|---|
| 영어 Agent 화면 / Core | 7101 / 7102 | 비어 있음 |
| Moshi 서버 | 8998 | 비어 있음 |
| 한국어 Agent 화면 | 7201 | 비어 있음 |
| Qwen Audio Agent Gateway | 설치 후 기록 | - |
| 한국어 Realtime Voice Server | 8765 | **충돌**: yang 의 `.venv/bin/python main.py` (3일째 실행 중) |
| Gemma 4 vLLM | 8001 | 비어 있음 |
| Nemotron Transcribe 30B vLLM | 8000 | **충돌**: yang 의 `uvicorn app.main:app --port 8000` (7일째 실행 중) |

그 밖에 사용 중: 7860 (baek `app_flash.py`), 8020·28473 (Korean_interface PoC).

## 결정 (2026-11-12): 기존 프로그램은 그대로 두고 이 프로젝트 포트를 바꾼다

| 서비스 | 확정 포트 |
|---|---|
| 한국어 Realtime Voice Server | **8766** (`ws://127.0.0.1:8766/v1/realtime`) |
| Nemotron Transcribe 30B vLLM (후처리 모드) | **8002** |

나머지는 규약 포트 그대로 (7101 / 7102 / 8998 / 7201 / 8001).

## 2026-11-16 영어 Agent 확인

7101, 7102, 8998 은 여전히 비어 있다 (`ss -ltnp`). Moshi PyTorch 서버(moshi 0.2.13)의 기본 포트는 8998 로 규약과 같다
(`moshi/server.py` 의 `--port` 기본값). Rust 백엔드 `config-q8.json` 의 `port` 도 8998 이다.

2026-11-17: realtime 의 en-moshi(PyTorch INT8 임시 대체, `services/en-moshi.sh`)도 같은 8998 을 쓴다.
en-solo(BF16)와 realtime(INT8)은 서로 다른 모드라서 동시에 뜨지 않는다.
