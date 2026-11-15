# Korean Realtime Server 구성 및 CPU 스모크 결과 (2026-11-13)

GPU 를 쓰지 않고(`CUDA_VISIBLE_DEVICES=""`) 확인한 결과이다. Gemma 4 실측과 Qwen3-TTS GPU 구동은 GPU 사용 조율 이후에 진행한다.

## 1. 구성

`realtime_server/main.py` 가 speech-to-speech 1.1.0 을 패키지 수정 없이 실행한다. 바꾸는 지점은 세 곳이다.

| 지점 | 내용 | 이유 |
|---|---|---|
| `backend_registry.STT_BACKENDS["nemotron-streaming"]` | `realtime_server.stt_streaming.NemotronTransformersSTTHandler` 로 교체, `required_extra=None` | NeMo 없이 transformers 로 Nemotron 3.5 ASR 구동, 한국어 고정 및 language_mismatch 이벤트 |
| `speech_to_speech.LLM.lm_output_processor.LMOutputProcessor` | `KoreanLMOutputProcessor` 로 교체 | 문장 단위 출력 검사(금칙어, 개인정보 요청) 후 숫자·기호 정규화. `s2s_pipeline` 은 함수 안에서 이 모듈로부터 import 하므로 `s2s_pipeline` 속성이 아니라 이 모듈 속성을 바꿔야 적용된다 |
| `ChatCompletionsApiModelHandler._build_optional_kwargs` | 요청마다 `temperature`, `max_tokens` 추가 | 1.1.0 의 chat-completions 백엔드는 텍스트 응답에 두 값을 보내지 않고 CLI 옵션도 없다(`--llm_gen_*` 는 이 백엔드에서 무시됨). 아동 대상 짧은 답변을 보장하기 위해 넣는다 |

나머지 옵션은 `config/services.yaml` 에서 만든다. 백엔드별 옵션은 `--stt/--llm_backend/--tts` 선택 뒤에만 파서에 추가되므로 `serve --help` 에는 보이지 않는다. `tests/test_main_argv.py` 가 `parse_arguments` 로 직접 확인한다("Ignoring options" 경고 없음).

## 2. 단위 시험

`.venv/bin/python -m pytest -q tests` : 79 passed (정규화, 출력 검사, argv 파싱, 패치 적용).

## 3. CPU E2E 스모크 (`bench/cpu_smoke.sh`)

- 구성: 가짜 OpenAI 호환 LLM(`bench/mock_llm_server.py`, 127.0.0.1:8001, 고정 답변) + `realtime_server.main`(8766, STT Nemotron CPU, TTS 는 `--tts supertonic --supertonic_tts_lang ko` 로 CPU 대체) + `bench/smoke_client.py`(서버 안 참조 음성 WAV 1개를 실시간 속도로 전송).
- 가짜 LLM 답변: "사과가 3개 있어요. 지금은 10:30이에요. 너의 전화번호를 알려 줄래요? 이 문장은 나오면 안 돼요."
- 결과
  - 서버 기동 44초, 핸들러 생성 오류 없음. 이벤트 흐름 정상(speech_started → transcription.delta 21회 → completed → response.created → output_audio.delta 53회 → response.done).
  - 출력 transcript: "사과가 세 개 있어요. 지금은 열 시 삼십 분이에요. 그 이야기는 하지 않을게요. 우리 하던 공부를 계속해 볼까요?" (정규화 2건, `pii_request.ask_phone` 차단 후 대체 문장, 이후 문장 삭제)
  - `logs/events.jsonl`: normalized 2건, guard_blocked 1건 기록.
  - LLM 요청 본문: `temperature 0.6`, `max_tokens 160`, `chat_template_kwargs.enable_thinking=false` 확인.
  - 로그 지연(긴 발화 약 50초): stt 2.34 s(CPU), llm 1.02 s(가짜 서버, 의미 없음).
- 알려진 한계: 차단된 원문이 LLM 대화 기록에 남을 수 있다(다음 단계에서 기록 정리 필요).

## 4. 정규화 수락 시험 (`bench/normalizer_acceptance.py`, CPU)

6번 문장 "사과 3개와 배 10개를 샀어요. 10:30에 만나요." → "사과 세 개와 배 열 개를 샀어요. 열 시 삼십 분에 만나요."

| 음성 | TTS 입력 | Nemotron 인식 | CER |
|---|---|---|---|
| qwen3-0.6b ggml (1단계) | 원문 | 사과세계와 배 열개를 샀어요. 십시 삼십분에 맞나요? | 0.476 (원문 기준) |
| supertonic (1단계) | 원문 | 사과 세계와 배 열개를 샀어요. 삼십에 만나요. | 0.381 (원문 기준) |
| supertonic | 정규화 | 사과 세계와 배열개를 샀어요. 열시 삼십분에 맞나요? | **0.095** |
| qwen3-0.6b ggml CPU | 정규화 | 사과세계와 배 열개를 샀어요. 열시 삼십분에 맞나요? | **0.095** |

정규화 후 시각을 두 엔진 모두 바르게 읽었다. 남은 오차("세 개"→"세계", "만나요"→"맞나요")는 발음이 같거나 비슷한 데서 생긴 인식 쪽 오차로 보인다. 원자료: `docs/reports/normalizer_acceptance.json`.

## 5. vLLM 환경

`.venv-vllm`: vllm 0.33.0, torch 2.14.0+cu130, transformers 5.20.2. 캐시된 `google/gemma-4-E4B-it` 설정을 `model_type=gemma4`, `Gemma4ForConditionalGeneration` 으로 인식한다(GPU 미사용 확인).

## 6. GPU 를 쓰지 않은 이유

현재 GPU 사용량은 33.7/49.1 GB 이고, 같은 GPU 를 쓰는 다른 사용자의 프로세스는 30~43 GB 사이에서 변한다. vLLM 은 시작할 때 `gpu_memory_utilization` 만큼 메모리를 미리 잡고 유지하므로, 그 프로세스가 최대치에 이르면 메모리 부족으로 그 프로세스가 중단될 수 있다. 사용 시간대와 용량을 조율한 뒤 측정한다.
