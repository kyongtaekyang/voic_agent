# 6단계 사전 구축 보고: Slow Path (2026-11-16)

GPU와 Transcribe 30B 가중치 없이 만들 수 있는 부분만 만들고, 가짜 서버로 시험했습니다. **단계 완료 판정 전입니다.** 6단계 완료 기준(야간 배치로 전사·요약 자동 생성)과 Gemma 4 오디오 직접 입력 비교 실험은 GPU가 필요합니다.

## 1. 만든 것

| 순서 (프롬프트 2.2절) | 구현 |
|---|---|
| 1. 세션 녹음과 로그를 `queue/korean/`에 쌓음 | `web/recorder.py`. 화면 서버 중계 경로에 연결(`web/server.py`). 세션마다 `user.wav`, `agent.wav`(16kHz), `events.jsonl`(턴 경계, 실시간 STT 결과, 응답 문장, 끼어들기, kva 이벤트), `meta.json` |
| 2. Transcribe 30B로 축어 전사 (시스템 프롬프트 없음, temperature 0, Thinking 끔) | `slow_path/run.py transcribe`. 제품명·교재 어휘는 `slow_path.context_terms`로 지시문에 붙임 |
| 3. 실시간 STT와 비교해 CER 계산 | `slow_path/cer.py` (공백·문장부호 제거 후 편집 거리). `metrics.json`에 턴 수, 사용자 발화 비율, 끼어들기 수, 응답 지연(턴 끝 → 첫 응답 음성) p50·p95도 함께 저장 |
| 4. Transcribe를 내리고 Gemma 4를 올려 요약·학습 피드백 | `slow_path/run.py summarize`, 지시문 `config/prompts/session_summary.md` (JSON: 요약, 주제, 잘한 점, 제안, 사람이 확인할 점) |
| 순서를 모드 전환 스크립트에 포함 | `ada-ops/nightly-batch`: `gpu-mode batch` → 전사 → `gpu-mode summary`(새 모드: Gemma만) → 요약 → `gpu-mode stop` 또는 `--then realtime`. 처리할 세션이 없으면 GPU 모델을 올리지 않음 |

- 세션 상태는 `queued → transcribed → summarized`입니다. 실패한 세션은 상태를 그대로 두고 `meta.json`의 `errors`에 남기므로 다음 배치에서 다시 시도합니다.
- 전사·요약 요청은 `127.0.0.1`로만 보냅니다. 다른 주소를 주면 실행을 거부하고, 프록시 환경 변수도 무시합니다.

## 2. 시험 결과

한국어 Agent 테스트 **110건 통과** (기존 103건 + 신규 7건, CPU).

| 시험 | 확인한 내용 |
|---|---|
| CER | 문장부호·공백·언어 태그 무시, 편집 거리 |
| 녹음기 | 4개 파일 생성, 응답 문장 조립, 음성 데이터는 events에 넣지 않음, kva.status 제외 |
| 화면 서버 중계 + 녹음 | 가짜 Realtime 서버 ↔ `web.server` ↔ 테스트 클라이언트로 실제 중계 경로에서 세션이 기록됨 |
| 배치 전체 | 가짜 OpenAI 호환 서버로 전사 → 지표 → 요약. 전사 요청에 시스템 프롬프트 없음, temperature 0, `enable_thinking: false`, 오디오 data URI 확인 |
| 실패 처리 | 서버 연결 실패 시 상태 유지, 오류 기록 |
| 외부 주소 차단 | `http://example.com` 거부 |

**시험 중 발견해서 고친 문제:** 브라우저 연결이 끊길 때 중계 작업이 취소되면, `finally` 블록의 `await`에서 멈춰서 녹음이 닫히지 않았습니다(`meta.json` 누락). 녹음 닫기를 `finally`의 맨 앞으로 옮겼습니다.

`nightly-batch --dry-run`은 대기열이 비어 있어서 "처리할 세션 없음"으로 정상 종료했습니다.

## 3. 결정이 필요한 것

1. **녹음 기본값은 꺼 두었습니다(`slow_path.record: false`).** 아동 음성을 서버에 저장하므로 법정대리인 동의와 보관 기간 정책이 먼저 정해져야 합니다. 보관 기간 자동 삭제(`retention_days`)는 정책이 정해진 뒤에 구현합니다. 영어 Agent도 같은 이유로 `gateway.record: false`로 바꾸었습니다.
2. **Transcribe 30B 설치:** 가중치(BF16 약 68GB)와 NeMo Speech vLLM 플러그인이 아직 없습니다. `ada-ops/services/transcribe-vllm.sh`는 문서 기준으로 작성했고, 가중치나 플러그인이 없으면 이유를 출력하고 종료합니다. Ada 48GB에서는 FP8(`--quantization fp8`)로 올려야 하는데, 이 조합이 실제로 동작하는지는 GPU 사용 시간에 확인해야 합니다. 문서는 vLLM 0.32.0 기준이고, 설치된 버전은 0.33.0입니다.

## 4. 한계

- 축어 전사 지시문은 아직 기본 문구("Transcribe the following audio.")입니다. 축어·화자 구분 형식은 모델 카드의 템플릿을 확인한 뒤 확정합니다.
- CER은 세션 전체를 이어 붙여서 비교합니다. 발화별 CER을 구하려면 Transcribe의 단어 타임스탬프와 턴 경계를 맞추는 작업이 필요합니다.
- 응답 지연은 화면 서버가 첫 응답 음성을 받은 시각 기준이며, 브라우저 재생 지연은 포함하지 않습니다.
- 요약에는 실시간 STT 대화와 축어 전사(참고)를 함께 넣습니다. 세션이 길면 Gemma의 `max_model_len`(4096)을 넘을 수 있으므로, 실측 후 잘라 넣는 규칙을 정해야 합니다.
