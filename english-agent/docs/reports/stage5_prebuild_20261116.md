# 영어 Agent 5단계 사전 구축 보고: 화면·Gateway·녹음 큐 (2026-11-16)

GPU가 없어서 실제 Moshi 대신 **가짜 Moshi 서버**(`bench/mock_moshi.py`, 실제와 같은 바이너리 규격)로 시험했습니다. **단계 완료 판정 전입니다.** 5단계 완료 기준(화면에서 대화·자막·상태 표시)은 실제 Moshi와 브라우저 마이크로 다시 확인해야 합니다.

## 1. 만든 것

| 구성 | 파일 | 포트 |
|---|---|---|
| 화면 서버 | `web/server.py`, `web/static/` (영어 화면, 개발자 화면은 설정으로 켬) | 7101 |
| Core(Gateway) | `gateway/server.py`, `gateway/session.py` | 7102 |
| ASR 병행 감시 | `gateway/asr_worker.py` (에너지 기반 발화 분리 + Nemotron CPU, `language=auto`) | - |
| 녹음 큐 | `gateway/recorder.py` → `queue/english/<세션>/user.wav, agent.wav, events.jsonl, meta.json` | - |
| 운영 스크립트 | `ada-ops/services/en-core.sh`, `en-web.sh` (+ `.ready`). `en-solo` 모드에 추가 | - |
| 점검 도구 | `bench/mock_moshi.py`, `bench/smoke_e2e.py` | 8999(가짜 Moshi) |

이벤트 규격은 `docs/event_schema.md`에 있습니다.

**설계 결정:** 브라우저와 Core는 PCM16 24kHz로 주고받고, Opus 변환은 Core가 합니다(Moshi 공식 화면은 브라우저에서 Opus를 다룸). Core가 원음 PCM을 직접 다루므로 출력 지연 버퍼, 안전 차단, ASR, 녹음을 한 곳에서 처리할 수 있습니다. 대신 브라우저와 서버 사이의 전송량이 늘어납니다(양방향 각 384 kbps). SSH 터널 또는 사내망 기준으로는 문제가 되지 않을 것으로 봅니다.

## 2. 시험 결과 (CPU)

**단위·통합 테스트 66건 통과** (`.venv-core`, 7.9초). Gateway 통합 테스트 5건은 타이밍에 의존하므로 5회 반복했고, 5회 모두 통과했습니다.

| 시험 | 확인한 내용 |
|---|---|
| 대화 흐름 | 가짜 Moshi 음성·자막 수신, 상태 전환, 세션 녹음 4개 파일 생성 |
| 안전 차단 (발화 끝 단어) | "You are so stupid" 뒤 침묵 → `idle()`로 판정 → `eva.safety_block`(insult). **위반 단어는 자막에 나오지 않음**. 침묵 뒤 출력 재개, 다음 문장 정상 표시 |
| 출력 지연 버퍼 | `output_delay_ms: 300`에서 위반 판정 시 아직 내보내지 않은 음성을 버림(`dropped_ms > 0`) |
| 사용 중·점검 중 | 두 번째 연결은 `busy`, 운영 모드가 `batch`이면 `maintenance` |
| Moshi 연결 실패 | `eva.unavailable`(moshi_unreachable) |
| 화면 서버 | `/`, `/api/status`, `/ws` 중계 |

**시험 중 발견해서 고친 문제:** 처음에는 `word` 규칙 단어(예: "stupid")가 판정 전에 자막으로 먼저 나갔습니다. 판정은 다음 토큰에서 하는데, 자막은 바로 보냈기 때문입니다. 자막을 토큰 하나 늦게 내보내도록 고쳤습니다(약 80ms 지연).

**전체 경로 점검** (`bench/smoke_e2e.py`): 가짜 Moshi + Core + 화면 서버를 실제 포트로 띄우고, 합성 음성 4개(영어, 한국어, 한국어, 영어)를 화면 서버 `/ws`로 실시간 속도로 보냈습니다.

| 시각 | 이벤트 |
|---|---|
| 3.25초 | `user_text` "Hello, can you help me with my homework today?" (en, 인식 442ms) |
| 8.00초 | `user_text` "오늘은 공룡에 대해서 배우고 싶어요." (ko, 385ms) |
| 12.54초 | `user_text` "선생님 이거 영어로 뭐라고 해요?" (ko, 373ms) → **`language_mismatch`** (한국어 2회 연속) |
| 16.23초 | `user_text` "What is your favorite animal?" (en, 376ms) |

보낸 음성 19.6초, 받은 Agent 음성 19.5초였고, 녹음 파일이 정상 생성되었습니다. 사용자 자막은 발화가 끝나고 약 0.9초 뒤에 표시됩니다(무음 판정 0.5초 + 인식 약 0.4초).

## 3. 환경 변경: 가상환경 분리

transformers 5.15.1(Nemotron ASR)은 huggingface-hub 1.x와 safetensors 0.8 이상을 요구하고, moshi 0.2.13은 huggingface-hub<1.0과 safetensors<0.8을 요구합니다. 그래서 한 환경에 둘 수 없습니다. Gateway는 moshi를 import하지 않으므로 다음처럼 나눴습니다(`bench/setup_venvs.sh`). 두 환경 모두 `pip check`를 통과했습니다.

- `.venv`: Moshi 서버 전용 (GPU torch 2.9.1)
- `.venv-core`: Gateway, 화면, ASR, 테스트 (CPU torch 2.9.1+cpu, transformers 5.15.1, librosa)

## 4. 남은 일

1. **실제 Moshi로 확인 (GPU 필요):** 브라우저 왕복 대화, 끼어들기, 에코 제거 상태에서 Moshi가 자기 목소리에 반응하지 않는지 확인합니다.
2. **안내 음성 파일:** `assets/redirect_en.wav`가 아직 없습니다. 지금은 차단 시 안내 문구만 표시합니다.
3. **사용자 부분 자막:** 지금은 발화가 끝난 뒤 확정 자막만 나옵니다. lookahead 스트리밍 인식으로 바꾸면 말하는 중에도 자막을 표시할 수 있습니다.
4. **지연 측정:** KPI(사용자 발화 끝 → 첫 응답 음성 p50 300ms)는 실제 Moshi로 측정해야 합니다.
5. **브라우저 확인:** 화면은 아직 실제 브라우저와 마이크로 확인하지 않았습니다. 가짜 Moshi로도 화면 동작(마이크, 재생, 자막, 언어 안내)은 확인할 수 있습니다. 방법은 README에 적었습니다.
