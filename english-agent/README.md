# 영어 음성 Agent (english-agent)

영어 전용 실시간 음성 튜터. 한국어 Agent(`../korean_agent/`)와는 별개의 프로그램·화면이며, Ada 서버 GPU 한 장을 함께 쓴다 (`../ada-ops/`).
명세: `../english_voice_agent_prompt.md`, 참고: `../Moshi_오픈소스_실시간_음성AI_사용설명서.md`.

- 대화 엔진: Kyutai Moshi (Mimi 코덱, 24 kHz, Full-Duplex)
- 화면/Core: 127.0.0.1:7101 / 7102, Moshi 서버: 127.0.0.1:8998
- 음성과 대화 기록은 서버 밖으로 보내지 않는다.
- Safety Monitor 검증이 끝나기 전에는 아동 사용자에게 공개하지 않는다 (1차 대상: 성인 학습자, 데모).

## 상태 (2026-11-16)

GPU 여유가 부족해서(약 10 GB) GPU 없이 할 수 있는 작업만 사전 구축했다. 단계 완료 판정 전이다. 보고서: `docs/reports/cpu_prebuild_20261116.md`.

- 0단계: 공용 `ada-ops/gpu-mode` 공유, 영어 단독 프로필 `en-solo` 추가, Moshi BF16 가중치 받음 (15 GB)
- 2단계 조사: 텍스트 토큰은 공식 WebSocket 규격(0x02)으로 제공됨 (`docs/version_check.md`)
- 3단계: `core/safety_monitor.py` 규칙 기반 검사기, 토큰당 p99 0.1 ms. Gateway 는 말이 끊기면 `idle()` 을 호출해야 함 (`docs/event_schema.md`)
- 4단계: `core/asr_monitor.py` 언어 감지 규칙, Nemotron `language=auto` 로 직원 녹음 8개 중 7개를 비영어로 감지 (나머지 clip05 는 한국어가 아닐 가능성, 미확인), CPU RTF 0.43
- 5단계: 화면(:7101)·Core(:7102)·녹음 큐, 가짜 Moshi 로 66개 테스트와 전체 경로 점검 통과 (`docs/reports/stage5_prebuild_20261116.md`)
- 6단계: `slow_path/` 야간 배치 (Transcribe 30B 전사, ASR 자막 WER, 안전 차단 지표, agent.wav 사후 안전 검사), `ada-ops/nightly-batch` 가 한국어와 함께 처리. 가짜 Transcribe 로만 시험 (`docs/reports/slow_path_prebuild_20261117.md`)
- 6단계 보강: 끼어들기·사용자 발화 비율 지표, 파인튜닝 후보 분류와 개인정보 규칙 검사 (slow_path/turns.py, ndidate.py, python -m slow_path.run classify). 8단계 준비: inetune/ 데이터 변환·검사, 학습 설정 초안 (docs/reports/slowpath_finetune_prebuild_20261118.md)
- GPU 모델 로딩(1단계 이후)은 GPU 사용 시간 합의 후 진행한다.

## 개발

가상환경은 두 개다 (moshi 와 transformers 의 패키지 버전이 충돌한다). 만들기: `bash bench/setup_venvs.sh`

- `.venv`: Moshi 서버 전용 (`ada-ops/services/en-moshi-bf16.sh`)
- `.venv-core`: Gateway(`gateway/`), 화면(`web/`), ASR, 테스트

```bash
cd english-agent
CUDA_VISIBLE_DEVICES= .venv-core/bin/python -m pytest -q tests        # 테스트 (GPU 사용 안 함)
CUDA_VISIBLE_DEVICES= .venv-core/bin/python bench/safety_latency.py  # Safety Monitor 처리 시간
CUDA_VISIBLE_DEVICES= .venv-core/bin/python -m slow_path.run status  # Slow Path 대기열 (전사는 후처리 모드에서 nightly-batch 가 실행)
```

## 실행

정식 실행은 `ada-ops/gpu-mode en-solo`(영어 단독) 또는 `realtime`으로 한다. 이 명령은 Moshi, Core(:7102), 화면(:7101)을 순서대로 띄운다.
운영 모드가 `en-solo`나 `realtime`이 아니면 화면에 "under maintenance"가 표시된다.

**접속 (SSH 터널):** 브라우저 마이크는 HTTPS 또는 localhost에서만 허용되므로 터널로 접속한다. 화면 서버는 127.0.0.1에만 열려 있다.

```bash
ssh -p <SSH 포트> -L 7101:127.0.0.1:7101 <사용자>@<ADA 서버>   # 접속 정보는 저장소에 적지 않는다
# 브라우저에서 http://localhost:7101  (개발자 화면: config 의 web.developer_panel: true 로 바꾼 뒤 http://localhost:7101/?dev=1)
```

**GPU 없이 화면 시험 (가짜 Moshi):** 사인파와 정해진 대본을 돌려주는 `bench/mock_moshi.py`를 쓴다. 사용자 자막과 언어 감지는 실제 ASR로 동작한다.

```bash
cd english-agent && export CUDA_VISIBLE_DEVICES=
sed -e 's#ws://127.0.0.1:8998/api/chat#ws://127.0.0.1:8999/api/chat#' -e 's/require_mode: true/require_mode: false/' config/services.yaml > /tmp/eva_mock.yaml
.venv-core/bin/python bench/mock_moshi.py --port 8999 &
EVA_CONFIG=/tmp/eva_mock.yaml .venv-core/bin/python -m gateway.server &
EVA_CONFIG=/tmp/eva_mock.yaml .venv-core/bin/python -m web.server &
# 끝나면 세 프로세스를 kill 하고 queue/english/ 의 시험 녹음을 지운다
```
