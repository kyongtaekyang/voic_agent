# 영어 Agent 6단계 사전 구축: 후처리 모드 Slow Path (2026-11-17)

**상태:** CPU 사전 구축이며 단계 완료 판정 전입니다. Transcribe 30B 가중치와 NeMo Speech vLLM 플러그인이 아직 없고, GPU 여유가 약 7 GB(baek `app_flash.py` 30.8 GB, Korean_interface·EngVoice 약 10.5 GB)라서 실제 전사는 하지 못했습니다. 모든 시험은 가짜 Transcribe 서버로 했습니다.

## 1. 구현 내용

| 구성 | 파일 | 내용 |
|---|---|---|
| 배치 실행 | `slow_path/run.py` | `transcribe`, `report`, `status`. `queue/english/<세션>/` 의 queued 세션을 전사하고 지표를 계산한 뒤 status 를 transcribed 로 바꿉니다. 실패하면 status 를 그대로 두고 `meta.json` 의 `errors` 에 남겨서 다음 배치에서 다시 시도합니다 |
| WER | `slow_path/wer.py` | 단어 단위. 소문자, 단어 안의 아포스트로피만 남기고 문장부호 제거. 한국어 `cer.normalize` 는 공백까지 지워서 쓰지 않았습니다 |
| 설정 | `config/services.yaml` 의 `slow_path:` | 대기열, Transcribe 주소(127.0.0.1:8002), 지시문, 사후 안전 검사 여부, 에너지 기준 |
| 야간 배치 | `ada-ops/nightly-batch` | 한국어와 영어 대기열을 함께 처리 (2절) |
| 시험 | `tests/test_slow_path.py` | 10건. 실제 `gateway/recorder.Recorder` 로 세션을 만들고 가짜 HTTP 서버로 Transcribe 를 대신합니다 |

**결과 파일 (세션 폴더)**

- `transcript.json`: `user`(user.wav 전사), `agent`(agent.wav 전사, `audit_agent_audio: true` 일 때)
- `metrics.json`: 아래 표
- `queue/english/_kpi.json`: `report` 가 전사된 세션 전체를 모은 값 (WER 합산, 차단 수, 차단 전에 나간 음성 최대값, 사후 검사 위반 수)

| 지표 | 계산 방법 | KPI 연결 |
|---|---|---|
| `asr_wer` | Transcribe 30B 의 user.wav 전사(기준) 대비 ASR Streaming 자막(`eva.user_text` 를 이어 붙임) | 인식 품질 |
| `non_english_turns`, `language_mismatch` | 자막의 `lang` 이 en 이 아닌 턴 수, 언어 안내 횟수 | 비영어 발화는 WER 을 키우므로 함께 표시 |
| `safety.blocks`, `by_category`, `already_played_ms`, `dropped_ms_total` | `eva.safety_block` 이벤트 집계 | 차단 성공률, 차단 전에 나간 음성 길이 |
| `safety.audit` | agent.wav 를 전사해서 문장마다 Safety Monitor 로 다시 검사. 실시간 감시를 통과해서 실제로 소리로 나간 위반을 찾습니다 | 안전성 (누락 탐지) |
| `agent_caption_vs_audio_wer` | agent.wav 전사 대비 Agent 자막. 차단한 구간은 자막에서 빠지므로 0 이 아닐 수 있습니다 | 텍스트 토큰과 음성의 일치 정도 |
| `response_latency_ms_approx` | 에너지 기준으로 user.wav 발화 끝 → agent.wav 다음 음성 시작 | **근사치** (아래 한계 참고) |

## 2. 야간 배치 변경 (`ada-ops/nightly-batch`)

1. 시작할 때 한국어(queued, transcribed)와 영어(queued) 대기열을 모두 확인합니다. 셋 다 0 이면 GPU 모델을 올리지 않고 끝냅니다. 이전에는 한국어 대기열만 확인했으므로, 영어 세션만 있는 밤에는 아무것도 처리하지 않았을 것입니다.
2. Transcribe 30B 가 올라와 있는 동안 한국어와 영어를 차례로 전사하고, 영어는 `report` 로 `_kpi.json` 을 갱신합니다.
3. 전사가 끝난 뒤 한국어 대기열을 다시 확인하고, **요약할 한국어 세션이 있을 때만** Gemma 4 를 올립니다. 영어만 있는 밤에 Gemma 를 불필요하게 올리지 않게 하려는 것입니다.
4. 시험용 환경 변수 `NB_KO_QUEUE`, `NB_EN_QUEUE` 로 대기열 폴더를 바꿀 수 있습니다.

**확인한 내용**

| 시험 | 결과 |
|---|---|
| 실제 `gpu-mode`, `--dry-run`, 임시 대기열 4가지 | 빈 대기열이면 바로 종료. 나머지는 gpu-mode 가 "GPU 여유 부족(필요 40000 MiB, 여유 6979 MiB)"으로 거부하고 종료 처리(5단계 stop)까지 실행. 다른 사용자 프로세스와 `run/mode` 는 그대로 |
| 가짜 gpu-mode, `--dry-run` | 영어만: batch → (요약 건너뜀) → stop. 한국어만 + `--then realtime`: batch → summary → realtime. 둘 다 + 한국어 transcribed: batch → summary → stop |
| 가짜 gpu-mode, 실제 실행, Transcribe 없음 | 영어 전사 실패가 `meta.json` 의 errors 에 남고 status 는 queued 유지, `_kpi.json` 생성, 종료 코드 1, stop 실행 |
| 전체 테스트 | 영어 76건 통과 (기존 66 + 새로 10), 한국어 111건 통과 |

## 3. 한계와 확인할 일

- **응답 지연은 근사치입니다.** user.wav 는 받은 순서대로 이어 쓰고(벽시계 보정 없음), agent.wav 는 벽시계 기준으로 무음을 채우므로 시간이 조금씩 어긋날 수 있습니다. 전이중이라 맞장구도 응답으로 셉니다. 에코 제거가 불완전하면 Agent 음성이 user.wav 에 섞여 사용자 발화로 잡힐 수 있습니다. KPI(p50 300ms) 판정에는 실제 Moshi 로 Gateway 에서 측정한 값을 씁니다.
- **ASR WER 은 비영어 발화를 포함합니다.** 자막과 전사를 턴 단위로 맞추지 않고 세션 전체를 비교하므로, 한국어 발화가 많으면 WER 이 커집니다. `non_english_turns` 를 함께 봐야 합니다.
- **Transcribe 입력은 16 kHz 로 바꿔서 보냅니다** (녹음은 24 kHz, 모델 규격은 16 kHz 모노). vLLM 이 다시 샘플링하는지는 실제 서버로 확인해야 합니다.
- **축어 전사 지시문은 미확정입니다.** 지금은 "Transcribe the following audio." 를 씁니다. 모델 카드 템플릿을 확인한 뒤 한국어 Slow Path 와 함께 고칩니다.
- **긴 세션은 전사가 잘릴 수 있습니다.** `transcribe-vllm.sh` 가 음성 디코딩 제한을 3600초로 늘렸지만(기본 600초), 그것은 입력 음성 길이 제한일 뿐입니다. 출력은 `transcribe_max_tokens: 8192`, 문맥은 `--max-model-len 32768` 로 묶여 있고, 문서에 따르면 발화가 촘촘한 1시간 녹음은 출력 토큰이 약 20,000개 필요합니다. 잘려도 오류가 나지 않으므로, `finish_reason` 이 `length` 이면 `transcript.json` 에 `truncated: true`, `metrics.json` 에 `transcript_truncated`, `_kpi.json` 에 `sessions_truncated` 로 표시합니다. 실제 한계 길이는 서버로 확인해야 합니다. **한국어 `transcribe_one` 도 같은 8192 제한이 있고, 잘림 표시는 아직 없습니다.**
- **사후 안전 검사는 단어를 하나씩 넣습니다.** `SafetyMonitor.check_text()` 는 문장 전체를 한 번에 넣는데, 검사기는 끝부분 240자만 다시 보므로 문장부호 없는 긴 전사(742자)에서 앞쪽 "stupid" 를 놓쳤습니다(직접 확인). 그래서 Gateway 처럼 단어마다 `feed()` 하도록 했고, 이 경우를 시험에 넣었습니다. `check_text()` 자체는 테스트에서만 쓰므로 그대로 두었습니다.
- **사후 안전 검사는 세션당 Transcribe 호출을 두 번** 합니다. 대기열이 길어서 밤 안에 끝나지 않으면 `audit_agent_audio: false` 로 끕니다.
- 녹음(`gateway.record`)은 동의·보관 기간 정책이 정해질 때까지 꺼져 있으므로, 실제 운영에서는 대기열이 비어 있습니다.
