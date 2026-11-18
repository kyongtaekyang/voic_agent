# 영어 Agent 6단계 보강과 8단계 데이터 준비 (2026-11-18)

**상태:** CPU 사전 구축이며 단계 완료 판정 전입니다. GPU 여유가 부족해서(다른 프로그램 최근 2시간 최고 41,525 MiB / 49,140 MiB, `gpu-wait`가 계속 대기 중) 모든 시험은 CPU와 합성 세션으로 했습니다. 실제 녹음 세션은 없습니다(녹음이 꺼져 있음).

## 1. Slow Path 지표 보강 (프롬프트 2.5절)

2.5절이 요구하는 지표 중 **끼어들기 횟수**와 **사용자 발화 비율**이 빠져 있었으므로 추가했습니다. Moshi는 전이중이라 Gateway가 끼어들기 이벤트를 따로 내지 않습니다. 그래서 `user.wav`와 `agent.wav`의 에너지 발화 구간이 겹치는 모양으로 셉니다(`slow_path/turns.py`, `metrics.json`의 `turn_taking`).

| 지표 | 정의 | KPI 연결 |
|---|---|---|
| `barge_ins` | Agent가 말하는 도중에 사용자 발화가 시작된 횟수 | 끼어들기 |
| `barge_in_yield_rate` | 그중 Agent가 `yield_ms`(1,000 ms) 안에 멈춘 비율 | 끼어들기 처리 성공률 |
| `agent_cuts_user`, `_rate` | 사용자가 말하는 도중에 Agent가 시작하고, 사용자 발화가 `cut_min_ms`(500 ms) 이상 더 이어진 횟수와 사용자 턴 대비 비율 | 사용자 발화를 끊는 비율 |
| `user_speech_ratio` | 사용자 발화 시간 / (사용자 + Agent 발화 시간) | 사용자 발화 비율 |

`_kpi.json`에는 `barge_ins`, `barge_in_yield_rate`, `agent_cuts_user_rate`, `user_speech_ratio_median`을 더했습니다.

**한계:** 에너지 기준 근사치입니다. 에코 제거가 불완전해서 Agent 음성이 `user.wav`에 섞이면 끼어들기가 많게 나옵니다. 맞장구도 '말을 끊음'에 섞입니다. 기준값(`yield_ms`, `cut_min_ms`)은 실제 세션을 보고 조정해야 합니다.

## 2. 파인튜닝 후보 분류와 개인정보 검사 (프롬프트 2.5절 4번)

전사가 끝난 세션을 세 가지로 나눕니다(`slow_path/candidate.py`). 결과는 세션 폴더의 `candidate.json`과 `meta.json`의 `finetune`에 쓰고, 목록은 `queue/english/_finetune_candidates.json`(세션 이름과 이유만)에 씁니다.

| 결과 | 조건 |
|---|---|
| `candidate` | 아래 기준을 모두 통과하고 개인정보 규칙에도 걸리지 않음 |
| `needs_review` | 기준은 통과했지만 개인정보 규칙에 걸림. 사람이 확인하고 가린 뒤 `meta.json`에 `finetune_review.approved: true`를 적으면 다시 분류할 때 `candidate`(`pii_reviewed: true`)가 됨. `false`면 `rejected` |
| `rejected` | 기준 미달 (이유를 `reasons`에 기록) |

- **기준(초안, `config/services.yaml`의 `slow_path.finetune_candidate`):** 파인튜닝 동의 기록(`meta.consent.finetune == true`), 전사 잘림 없음, 세션 오류 없음, 실시간 안전 차단 0건, Agent 음성 사후 검사 위반 0건, 사용자 발화 20초 이상과 3턴 이상, 비영어 턴 10% 이하, 사용자 발화 비율 0.2~0.7, Agent가 말을 끊은 비율 0.2 이하.
- **동의 기준:** 프롬프트 3절에 따라 동의 기록이 있는 세션만 후보가 됩니다. 지금 Gateway는 동의 기록을 쓰지 않으므로 실제 세션은 모두 `rejected`가 됩니다. 의도한 동작입니다.
- **개인정보 검사:** 모델을 쓰지 않는 규칙 검사입니다(프롬프트 7절). `config/finetune_pii.yaml`에 자기 이름, 주소, 연락처, 학교, 생일, 비밀번호를 말하는 문장 패턴을 두었고, `safety_rules.yaml`의 전화번호·이메일 형태 규칙도 함께 씁니다. 숫자를 단어로 읽은 전화번호("five five five one two ...")도 잡되, 1씩 커지거나 작아지는 숫자 세기 연습은 제외합니다. 결과 파일에는 걸린 값을 가려서 씁니다(예: `my name is j***`).
- **한계:** 규칙 검사이므로 놓치는 경우가 있습니다. 사람 확인을 대신하지 못하며, 규칙 목록은 법무·개인정보 담당자가 검토해야 합니다. 특히 정규화(`normalize`)가 한글을 지우므로, 한국어로 말한 개인정보("제 이름은 …")는 찾지 못합니다(숫자는 남음). 비영어 턴 10% 기준이 노출을 줄이지만 막지는 못합니다.
- `python -m slow_path.run classify`로 GPU 없이 다시 분류할 수 있습니다. 기준을 바꾸거나 동의·검토 기록을 붙인 뒤에 씁니다. 검토 기록(`finetune_review`)을 따르므로 사람이 승인한 결과를 되돌리지 않습니다.
- 야간 전사 중 분류가 실패하면 `meta.json`의 `errors`에 `classify` 단계로 남기고, 세션은 `transcribed`로 둡니다. `queued`로 남으면 다음 밤에 Transcribe 30B로 다시 전사하게 되어 GPU 시간을 낭비하기 때문입니다. 분류는 `classify`로 다시 합니다.

## 3. LoRA 학습 데이터 준비 (프롬프트 3절, 8단계)

`finetune/README.md`에 자세히 적었습니다. 요약은 다음과 같습니다.

- 공식 `moshi-finetune`(2acc879) 코드를 직접 읽고 데이터 형식을 확인했습니다(`docs/version_check.md` 7절).
- `finetune/prepare.py`를 만들었습니다. `from-pairs`는 외부 성인 녹음(튜터/학생 음성 쌍, `consent`와 `adult`가 모두 true인 줄만)을 변환합니다. `from-queue`는 `candidate` 세션을 변환하고, `check`는 형식을 검사합니다.
- `finetune/moshi_lora.yaml`은 공식 권장값을 쓴 학습 설정 초안이며, 아직 실행하지 않았습니다. wandb는 쓰지 않습니다.
- **문서와 다른 점:** 공식 정렬 도구는 Whisper를 쓰지만 사양 밖 모델이므로 쓰지 않았습니다. 대신 Transcribe 30B 단어 타임스탬프를 쓰도록 했으나, 미들웨어 출력 형식은 미확인입니다. 확인 뒤 Slow Path가 `transcript.json`의 `agent.words`를 저장하도록 바꿔야 `from-queue`가 실제로 데이터를 만듭니다.
- `from-queue`는 세션 길이와 `user.wav` 길이가 0.5초 넘게 다른 세션을 건너뜁니다. 녹음기가 `user.wav`를 벽시계에 맞추지 않으므로, 마이크 프레임이 빠지면 두 채널의 시간이 어긋나서 잘못된 주고받기를 학습하게 됩니다.
- **주의:** 운영 세션의 튜터 채널은 Moshi 자신의 출력입니다. 운영 세션으로 학습하면 Moshi의 기존 말투를 다시 배우게 되므로, 튜터 말투 학습에는 사람 튜터 녹음(`from-pairs`)이 맞습니다.

## 4. 한국어 Slow Path 잘림 표시

전날 보고서에서 남은 일로 적었던 항목을 처리했습니다. 한국어 `transcribe_one`과 요약 호출도 `finish_reason == "length"`이면 `truncated: true`로 표시하고, `metrics.json`에 `transcript_truncated`를 씁니다(커밋 1fef350).

## 5. 시험

| 범위 | 결과 |
|---|---|
| 영어 전체 (`.venv-core`, GPU 없이) | 114건 통과, 1건 건너뜀 (기존 76건에 새로 38건 추가: 지표·분류·개인정보·검토 기록·분류 실패 33건, 데이터 준비 5건) |
| 한국어 전체 | 113건 통과 (기존 111건에 잘림 표시 2건 추가) |

## 6. 하지 않은 것

- `deploy/docker-compose.yml`(4절 저장소 구조)은 만들지 않았습니다. 서버에 docker가 없어서 동작을 확인할 수 없기 때문입니다. 확인할 수 없는 설정 파일을 두기보다 막힌 항목으로 보고합니다.
- 실제 GPU 측정, 브라우저 마이크 시험, 실제 Transcribe 전사는 GPU 시간이 정해진 뒤에 합니다.

## 7. 정해 주셔야 할 것

1. 파인튜닝 학습 데이터 출처 (적법하게 확보한 성인 대화 녹음. 아동 음성은 동의 전 금지)
2. 녹음 동의 절차와 보관 기간 (정해져야 `gateway.record`를 켜고 `consent.finetune`을 기록할 수 있음)
3. baek과 GPU 사용 시간대 (실시간 모드 약 15 GB, 학습 모드 약 40 GB)
4. 여러 세션 동시 처리 구조 (지금은 Agent마다 한 세션)
5. 한국어 아동 평가 세트 위치, Whisper 파인튜닝 여부
