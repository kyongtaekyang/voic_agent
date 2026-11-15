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
