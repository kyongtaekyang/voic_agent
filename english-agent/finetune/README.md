# 영어 Agent 튜터 스타일 LoRA 파인튜닝 준비 (프롬프트 3절, 8단계)

**상태:** CPU 사전 구축입니다. 학습은 아직 하지 않았습니다. 학습 데이터 출처, 학습 모드 GPU 시간, 평가 세트가 정해지지 않았습니다.

## 1. 파일

| 파일 | 내용 |
|---|---|
| `prepare.py` | 학습 데이터 만들기 (`from-queue`, `from-pairs`)와 형식 검사 (`check`) |
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

- 미들웨어가 내놓는 응답 형식을 아직 확인하지 못했습니다 (가중치와 NeMo 플러그인이 없음). 그래서 `prepare.py`는 중립 형식 `[{"word", "start", "end"}, ...]`를 입력으로 받습니다. 실제 형식을 확인한 뒤 Slow Path가 `transcript.json`의 `agent.words`에 이 형식으로 저장하도록 바꿔야 합니다. 지금은 `agent.words`가 없으므로 `from-queue`는 모든 세션을 "타임스탬프 없음"으로 건너뜁니다.
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
