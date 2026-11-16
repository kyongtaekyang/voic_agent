# 영어 음성 Agent (english-agent)

영어 전용 실시간 음성 튜터. 한국어 Agent(`../korean_agent/`)와는 별개의 프로그램·화면이며, Ada 서버 GPU 한 장을 함께 쓴다 (`../ada-ops/`).
명세: `../english_voice_agent_prompt.md`, 참고: `../Moshi_오픈소스_실시간_음성AI_사용설명서.md`.

- 대화 엔진: Kyutai Moshi (Mimi 코덱, 24 kHz, Full-Duplex)
- 화면/Core: 127.0.0.1:7101 / 7102, Moshi 서버: 127.0.0.1:8998
- 음성과 대화 기록은 서버 밖으로 보내지 않는다.
- Safety Monitor 검증이 끝나기 전에는 아동 사용자에게 공개하지 않는다 (1차 대상: 성인 학습자, 데모).

## 상태 (2026-11-16)

GPU 여유가 부족해서(약 10 GB) GPU 없이 할 수 있는 작업만 사전 구축했다. 단계 완료 판정 전이다. 보고서: `docs/reports/cpu_prebuild_20261116.md`.

- 0단계: 공용 `ada-ops/gpu-mode` 공유, 영어 단독 프로필 `en-solo` 추가, Moshi BF16 가중치 내려받기 시작
- 2단계 조사: 텍스트 토큰은 공식 WebSocket 규격(0x02)으로 제공됨 (`docs/version_check.md`)
- 3단계: `core/safety_monitor.py` 규칙 기반 검사기, 토큰당 p99 0.1 ms
- 4단계: `core/asr_monitor.py` 언어 감지 규칙, Nemotron `language=auto` 로 실제 한국어 녹음 7/7 감지, CPU RTF 0.43
- GPU 모델 로딩(1단계 이후)은 GPU 사용 시간 합의 후 진행한다.

## 개발

```bash
cd english-agent
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q tests      # 단위 테스트 (GPU 사용 안 함)
CUDA_VISIBLE_DEVICES= .venv/bin/python bench/safety_latency.py # Safety Monitor 처리 시간
```

가상환경은 `python3.11 -m venv .venv && .venv/bin/pip install moshi pyyaml pytest` 로 만든다 (한국어 Agent 의 `.venv` 와 분리).

