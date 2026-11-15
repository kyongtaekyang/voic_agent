# 한국어 음성 Agent (korean-voice-agent)

5~13세 아동도 쓸 수 있는 한국어 전용 음성 학습 도우미. 영어 Agent(Moshi)와는 별개의 프로그램·화면이며, Ada 서버 GPU 한 장을 함께 쓴다 (`../ada-ops/`).
명세: `../korean_voice_agent_prompt.md`. 음성과 대화 기록은 서버 밖으로 보내지 않는다.

## 구성

| 서비스 | 주소 | 실행 | 설명 |
|---|---|---|---|
| Gemma 4 vLLM | 127.0.0.1:8001 | `ada-ops/services/ko-gemma-vllm.sh` | `deploy/run_gemma_vllm.py`, `.venv-vllm` |
| Realtime Voice Server | ws://127.0.0.1:8766/v1/realtime | `ada-ops/services/ko-realtime-server.sh` | `realtime_server/main.py`: speech-to-speech 1.1.0 + Nemotron ASR(CPU) + 출력 검사 + 정규화 + Qwen3-TTS(GPU) |
| 화면 | http://127.0.0.1:7201 | `ada-ops/services/ko-web.sh` | `web/server.py`: 사용자 화면, 개발자 화면(`?dev=1`), 화면용 WebSocket 중계 |

설정은 모두 `config/services.yaml` 에 있다. 시스템 프롬프트는 `config/prompts/korean_tutor_system.md`, 출력 검사 규칙은 `config/safety_rules.yaml`.

## 실행 (GPU 공유 규약을 따른다)

```bash
cd ~/pgm/voice_agent/ada-ops
./gpu-mode status              # 현재 모드, GPU 사용 프로세스
./gpu-mode realtime --dry-run  # 필요한 메모리와 여유 확인 (시작하지 않음)
./gpu-mode realtime            # ko-gemma-vllm → ko-realtime-server → ko-web 순서로 시작
./gpu-mode stop                # 이 스크립트가 시작한 서비스만 중지
```

**GPU 사용 시간대를 합의한 뒤에만 실행한다.** 같은 GPU 의 다른 사용자 프로세스는 메모리 사용량이 30~43 GB 사이에서 변하므로, 여유 검사를 통과해도 나중에 그 프로세스가 메모리 부족으로 중단될 수 있다.
GPU 여유가 부족하면 `gpu-mode` 는 시작하지 않는다. 다른 사용자의 GPU 프로세스는 절대 멈추지 않는다. `--force` 는 GPU 사용 시간을 조율한 뒤에만 쓴다.
운영 모드가 realtime 이 아니면 화면은 "현재 점검 중"을 표시한다.

## 화면 접속 (사내)

브라우저 마이크는 HTTPS 또는 localhost 에서만 허용되므로, 서비스는 127.0.0.1 에만 열고 SSH 터널로 접속한다.

```bash
ssh -p <SSH포트> -N -L 7201:127.0.0.1:7201 <계정>@<ADA 서버 주소>   # 접속 정보는 사내 문서 참고
# 브라우저: http://localhost:7201/        (개발자 화면: http://localhost:7201/?dev=1)
```

여러 사람이 접속해야 하면 사내 HTTPS 역방향 프록시를 따로 두는 방안을 검토한다 (미정).
Realtime 서버는 동시에 한 세션만 처리한다. 다른 사람이 쓰는 중이면 "다른 사람이 대화하고 있습니다"가 표시된다.

## 시험

```bash
.venv/bin/python -m pytest -q tests                  # 단위 시험 (모델 불필요)
bench/cpu_smoke.sh bench/data/ko_ref/clip06_1e8d231b.wav               # GPU 없이 E2E: 가짜 LLM + Supertonic(CPU)
SMOKE_VIA_WEB=1 bench/cpu_smoke.sh bench/data/ko_ref/clip06_1e8d231b.wav  # 화면 중계(7201)까지 포함
```

스모크 시험은 `CUDA_VISIBLE_DEVICES=""` 로 실행하며, 포트 8001/8766/7201 이 비어 있을 때만 동작한다.

## 문서

- `docs/reports/` 단계별 보고서
- `docs/event_schema.md` 화면 ↔ 서버 이벤트 규격
- `docs/version_check.md` 버전과 호환성 기록
- `THIRD_PARTY_NOTICES.md` 외부 구성요소 라이선스
