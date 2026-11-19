# 영어 Agent 튜터 스타일 LoRA 파인튜닝 준비 (프롬프트 3절, 8단계)

**상태:** CPU 사전 구축입니다. 학습은 아직 하지 않았습니다. 학습 데이터 출처, 학습 모드 GPU 시간, 평가 세트가 정해지지 않았습니다.

## 1. 파일

| 파일 | 내용 |
|---|---|
| `prepare.py` | 학습 데이터 만들기 (`from-studio`, `from-pairs`, `from-queue`)와 형식 검사 (`check`), 동의 철회 시 파생 데이터 삭제 (`remove_derived`) |
| `moshi_lora.yaml` | moshi-finetune 학습 설정 초안 (공식 예시 기준, 바꾼 값에 이유 표기) |
| `data/`, `runs/` | 학습 데이터와 체크포인트. 서버에만 두고 저장소에 올리지 않습니다 (`.gitignore`) |

공식 학습 코드는 저장소 밖 `~/opt/moshi-finetune`(kyutai-labs/moshi-finetune, 커밋 2acc879, Apache 2.0)에 받아 두었습니다. 학습용 가상환경은 아직 만들지 않았습니다. 이 코드는 `torch==2.6`과 `moshi`(git), `whisper_timestamped`를 요구하므로 기존 `.venv`(torch 2.9.1)와 따로 만들어야 합니다.

## 2. 데이터 형식 (공식 코드로 확인)

`README.md`, `annotate.py`, `finetune/data/interleaver.py`, `train.py`를 직접 읽고 확인했습니다.

- **음성:** 스테레오 wav입니다. **왼쪽(채널 0)은 Moshi가 배울 음성(튜터)**, 오른쪽(채널 1)은 사용자(학생) 음성입니다. 로더가 Mimi 샘플레이트(24 kHz)로 다시 샘플링하지만, `prepare.py`는 처음부터 24 kHz로 씁니다.
- **정렬:** wav 옆에 같은 이름의 `.json`을 두고 `{"alignments": [[단어, [시작초, 끝초], "SPEAKER_MAIN"], ...]}` 형식으로 씁니다. `train.py`가 `keep_main_only=True`로 Interleaver를 만들기 때문에 `SPEAKER_MAIN`(튜터) 단어만 학습에 쓰입니다. 공식 `annotate.py`도 채널 0만 전사합니다.
- **목록:** jsonl 한 줄에 `{"path": "data_stereo/x.wav", "duration": 초}`를 씁니다.
- `interleaver.py`는 정렬 json을 `open()`으로 인코딩 지정 없이 엽니다. 그래서 `prepare.py`는 json을 ASCII로 씁니다.

## 3. 문서와 다른 점: 단어 타임스탬프를 만드는 모델

공식 `annotate.py`는 **Whisper**(`whisper_timestamped`, README는 stereo에 medium 권장)로 단어 타임스탬프를 만듭니다. Whisper는 세 md 파일에 없는 모델이므로 프롬프트 7절에 따라 임의로 쓰지 않았습니다.

대신 **Transcribe 30B의 단어 타임스탬프**를 쓰도록 만들었습니다. `docs/Nemotron_3.5_Transcribe.md`에 따르면 vLLM을 `ctc_timestamp_middleware`와 함께 구동하면 CTC 헤드가 단어별 타임스탬프를 붙입니다. 이 경로에는 다음 한계가 있습니다.

- (2026-11-19 갱신) 응답 형식은 NeMo Speech 소스(main d55f2a4, `ctc_serving.py`·`ctc_timestamps.py`)와 모델 카드로 확인했습니다. 요청에 `"mm_processor_kwargs": {"capture_ctc_timestamps": true}`를 넣고 `stream=false`로 보내면, 응답 최상위 `ctc_timestamps.words`에 `{"word","start","end","speaker"}`(초 단위)가 붙습니다. `slow_path/words.py`가 이것을 중립 형식 `[{"word","start","end"}]`로 바꿔서 스튜디오 세션에는 `tutor_words.json`, 운영 세션에는 `transcript.json`의 `agent.words`로 저장합니다. 다만 실제 GPU에서 한 번도 돌리지 않았으므로, 첫 학습 작업에서 `tutor_words.raw.json`(원본 응답)을 꼭 확인해야 합니다.
- Whisper 정렬과 품질이 같은지는 확인하지 않았습니다. 학습 결과가 나쁘면 정렬 품질을 먼저 의심해야 합니다.
- Whisper를 쓰는 편이 낫다고 판단되면, 근거를 갖춰서 먼저 제안하겠습니다.

## 4. 사용법

```bash
cd ~/pgm/voice_agent/english-agent
# (1) 외부에서 확보한 성인 녹음: 매니페스트 jsonl 한 줄에
#     {"id": "...", "tutor_wav": "...", "student_wav": "...", "tutor_words": "단어.json", "consent": true, "adult": true}
.venv-core/bin/python -m finetune.prepare from-pairs /경로/manifest.jsonl
# (2) 운영 세션 중 Slow Path 가 candidate 로 분류한 세션 (동의 기록 + 개인정보 확인 끝난 것만)
.venv-core/bin/python -m finetune.prepare from-queue
# 형식 검사
.venv-core/bin/python -m finetune.prepare check
```

- `from-pairs`는 `consent`와 `adult`가 모두 `true`인 줄만 만듭니다. 출력 파일 이름은 해시로 바꾸므로 원래 파일 이름(사람 이름 등)이 데이터 폴더에 남지 않습니다. 원본 대응표는 `data/pairs_sources.json`에 둡니다. 개인정보 삭제 요청이 오면 이 표로 찾아서 지웁니다.
- `from-queue`는 `meta.json`의 `finetune`이 `candidate`이고 `consent.finetune`이 `true`인 세션만 읽습니다. `needs_review`(개인정보 규칙에 걸린 세션)는 사람이 확인하고 가린 뒤 `meta.json`에 `finetune_review: {"approved": true, "by": ..., "at": ...}`를 적고 `python -m slow_path.run classify`를 다시 돌립니다. `finetune` 값을 손으로 바꾸면 다음 분류가 되돌리므로 그렇게 하지 않습니다.
- `from-queue`는 세션 길이(벽시계)와 `user.wav` 길이의 차이가 `--max-skew-s`(기본 0.5초)를 넘는 세션을 건너뜁니다. 녹음기(`gateway/recorder.py`)는 `user.wav`를 받은 순서대로 이어 쓰고 `agent.wav`만 벽시계에 맞춥니다. 그래서 마이크 프레임이 빠지면 두 채널의 시간이 어긋나고, 전이중 학습에서는 그 어긋난 주고받기를 그대로 배우게 됩니다. 이 검사는 `from-pairs`에는 해당하지 않습니다.

## 5. 정해야 할 것

1. **학습 데이터 출처:** 프롬프트 3절은 "적법하게 확보한 성인 대화 녹음으로 시작"이라고 정합니다. 어떤 녹음을 쓸지(사내 튜터 녹음, 공개 데이터셋 등) 정해 주셔야 합니다. 아동 음성은 법정대리인 동의와 내부관리규정 검토가 끝나기 전에는 쓰지 않습니다.
2. **운영 세션을 학습에 쓸지:** 운영 세션의 튜터 채널은 Moshi 자신의 출력입니다. 이 세션으로 학습하면 사람 튜터의 말투가 아니라 Moshi의 기존 말투를 다시 배우게 됩니다. `from-queue`는 사양(2.5절)에 따라 만들어 두었지만, 실제 튜터 말투 학습에는 `from-pairs`(사람 튜터 녹음)가 맞습니다.
3. **학습 모드 GPU 시간:** 공식 예시 기준 최대 39.6 GB가 필요합니다. 영어·한국어 Agent를 모두 멈추고, 다른 사용자(baek 등)와 시간을 맞춰야 합니다.
4. **평가 세트:** 원본과 파인튜닝 모델을 같은 세트로 비교해야 합니다 (프롬프트 3절).

## 6. 녹음 스튜디오와 아동 데이터 기준 (2026-11-18)

- 사람 튜터 녹음은 `studio/`(튜터 녹음 스튜디오, `studio/README.md`)에서 합니다. `python -m finetune.prepare from-studio` 가 `queue/tutor/` 세션을 변환합니다. 튜터 단어 타임스탬프(`tutor_words.json`)가 생기기 전에는 모든 세션을 건너뜁니다.
- 학습 데이터 기준(`training_gate`): 학습 활용 동의가 있고, 성인이거나, 아동이면 법정대리인 동의가 있고 `config/services.yaml` 의 `finetune.allow_child_data` 가 `true` 일 때만 변환합니다. 이 값은 내부관리규정 검토가 끝날 때까지 `false` 로 둡니다 (프롬프트 3절). `from-pairs` 매니페스트도 같은 기준을 쓰며, 아동 줄은 `"child": true, "guardian_consent": true` 로 적습니다.
- 스튜디오에서 세션을 지우면(동의 철회) `*_sources.json` 대응표로 파생 학습 데이터도 지웁니다.

## 7. 학습 작업과 적용 (2026-11-19)

GPU가 필요한 단계를 하나의 작업으로 묶었습니다. 포털(:7100) '서비스 운영'에서 **학습(train)** 을 예약하면, GPU 여유가 약 40 GB 생길 때 `gpu-wait`가 `gpu-mode train`을 올리고 `ada-ops/services/finetune-job.sh`가 아래 순서로 실행합니다.

1. **단어 타임스탬프** (`slow_path.words`): 타임스탬프가 없는 세션이 있을 때만 Transcribe 30B FP8(`services/transcribe-vllm.sh`)을 자식 프로세스로 띄우고, 끝나면 내립니다.
2. **데이터 변환·검사** (`finetune.prepare from-studio`, `from-queue`, `check`)
3. **LoRA 학습** (`~/opt/moshi-finetune`, `torchrun -m train`). 학습 데이터가 `finetune.min_items`(초안 10)보다 적으면 학습하지 않고 `prepared`로 끝냅니다.
4. 작업이 끝나면(실패해도) GPU 모드를 내리고, 학습 예약 전에 걸려 있던 예약(예: realtime)을 다시 겁니다.

- 실행마다 `finetune/runs/lora_<시각>/`에 설정(`config.yaml`), 로그, 체크포인트, `report.json`을 남깁니다. 진행 상황은 `finetune/runs/job_status.json`이며 포털 '학습' 메뉴에 보입니다.
- 학습은 `--once` 예약이므로 **학습 중에는 다른 사용자에게 GPU를 양보하지 않습니다.** 다른 사용자가 최고 42 GB를 쓰는 동안에는 시작하지 않습니다.
- 미리 보기: `.venv-core/bin/python -m finetune.job plan` (GPU 없이 무엇을 할지 출력)

**적용은 사람이 평가한 뒤 합니다.** 학습 작업은 결과를 자동으로 적용하지 않습니다.

```bash
.venv-core/bin/python -m finetune.promote list                 # 학습 결과 목록
.venv-core/bin/python -m finetune.promote set latest --by <이름>  # 또는 set lora_<시각>
.venv-core/bin/python -m finetune.promote clear                # 원래 모델로
```

- 적용 상태는 `finetune/active_lora.json`이며, 다음에 **en-solo(BF16) Moshi가 뜰 때부터** `--lora-weight`, `--config-path`로 반영됩니다. realtime 모드의 INT8 Moshi에는 적용되지 않습니다.

**전체 흐름 시험:** `ada-ops/test-all` (단위 시험 + `ada-ops/e2e/run_e2e.py`). GPU 없이 실제 스튜디오, gpu-wait, gpu-mode, 학습 작업, promote, 포털을 그대로 실행하고, Transcribe와 학습만 대역으로 바꿉니다. 학습 대역(`e2e/check_train.py`)은 moshi-finetune의 실제 데이터 로더와 Interleaver로 학습 데이터를 읽어서 형식을 검사합니다.
