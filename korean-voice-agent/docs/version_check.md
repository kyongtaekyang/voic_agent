# 버전 확인 기록 (프롬프트 8절)

문서에 나온 CLI 옵션, 환경 변수, 패키지 이름을 실제 설치 버전으로 다시 확인한 결과입니다. 확인 날짜는 2026-11-13입니다.

## 1. Nemotron 3.5 ASR Streaming 0.6B

| 항목 | 문서·프롬프트 | 실제 확인 결과 | 근거 |
|---|---|---|---|
| 모델 ID | `nvidia/nemotron-3.5-asr-streaming-0.6b` | 같음. gated 아님. 라이선스 OpenMDW-1.1 | HF API, 모델 카드 |
| 가중치 위치 | - | HF 캐시(`~/.cache/huggingface/hub`)에는 `refs/main`만 있고 가중치가 없음. `~/pgm/Korean_interface/models/stt/nemotron-3.5-asr-streaming-0.6b/`에 전체 사본이 있어서 이 경로를 사용함 | `find` |
| 한국어 지원 | 모델 카드로 확인하라고 지시 | `ko-KR`, "Transcription-ready(19개 로캘)" 등급. 모델 카드의 FLEURS 한국어 CER은 7.12~8.31% | 모델 카드 Supported Languages, 평가표 |
| 언어 코드 | - | `language="ko-KR"`. `"auto"`로 두면 문장 끝에 `<ko-KR>` 같은 언어 태그를 출력함 | 모델 카드 사용 예시 |
| 발화 종료(EOU) 판정 | 제공 여부를 확인하라고 지시 | **이 모델에는 발화 종료 판정 기능이 없음.** 모델 카드는 별도 모델인 Parakeet Realtime EOU 120M(`nvidia/parakeet_realtime_eou_120m-v1`)을 안내함 | 모델 카드 "Explore more" |
| 스트리밍 API | 모델 카드 예시 | transformers 5.13.0에서 `AutoModelForRNNT`, `set_num_lookahead_tokens`, `streaming_latency_ms`, `num_samples_first_audio_chunk`, `num_mel_frames_first_audio_chunk`가 모두 동작함 | `bench/asr_cpu_rtf.py` 실행 |
| lookahead 값 | - | `supported_num_lookahead_tokens = [0, 3, 6, 13]`, 기본값 3. 청크 지연은 각각 80 / 320 / 560 / 1120 ms | `config.json`, 실행 결과 |
| 모델 카드 예시 코드의 경계 조건 | - | **lookahead 0에서 모델 카드 예시 코드를 그대로 쓰면 실패함.** 두 번째 청크의 시작 위치가 음수가 되어 빈 오디오 조각이 만들어지고, `stft`가 `RuntimeError`를 냄. 벤치마크 스크립트에서는 모자란 앞부분을 0으로 채워서 해결함. 실시간 서버에서도 같은 처리가 필요함 | 실행 오류와 수정 |
| speech-to-speech 연동 | - | speech-to-speech 1.1.0의 `STT/nemo_asr_handler.py`가 `nemotron-3.5` 모델명을 인식하고 언어 태그를 처리함. 단, NeMo 경로(`speech-to-speech[nemo]`, `nemo_toolkit[asr]`)를 사용함 | 패키지 소스 |

측정에 사용한 환경은 `~/pgm/Korean_interface/.venv-linux`(Python 3.11, transformers 5.13.0, torch 2.11.0+cu128)입니다. 이 가상환경은 다른 서비스(포트 8020)가 사용 중이므로 읽기 전용으로만 사용했고, 패키지를 설치하지 않았습니다.

## 2. Qwen Audio Agent와 speech-to-speech

| 항목 | 문서 | 실제 확인 결과 |
|---|---|---|
| 로컬 음성 패키지 | `pip install 'speech-to-speech[paraformer]'` | PyPI 이름은 `speech-to-speech`, 최신 버전은 1.1.0, Python 3.11 이상, 저장소는 `huggingface/speech-to-speech`. `paraformer` extra는 존재하지만 중국어 STT(FunASR)용이므로 한국어 Agent에는 필요 없음. 이 프로젝트는 `speech-to-speech[supertonic]==1.1.0`으로 설치함 |
| Realtime 엔드포인트 | `ws://127.0.0.1:8765/v1/realtime` | 패키지의 `api/openai_realtime`이 같은 경로(`/v1/realtime`)의 OpenAI Realtime 호환 WebSocket 서버를 제공하고, 기본 주소도 `ws://127.0.0.1:8765/v1/realtime`임. **이 서버에서는 8765 포트가 이미 사용 중이므로 `ada-ops/ports.md`의 결정에 따라 8766을 사용함** |
| 포함된 STT | - | faster-whisper, whisper, NeMo(Nemotron 3.5 포함), Parakeet TDT, Paraformer, SenseVoice, Qwen3-ASR, OpenAI 호환 |
| 포함된 LLM | - | OpenAI 호환 Chat Completions / Responses 클라이언트. vLLM의 Gemma 4 서버에 연결할 수 있음 |
| 포함된 VAD·턴 감지 | - | Silero 계열 VAD, FireRedVAD, **Smart Turn**(턴 종료 판정 모델) |
| 포함된 TTS | 문서에 명시 없음 | ChatTTS, Drama-3, Facebook MMS, Kitten, Kokoro, MiniMax, OmniVoice, OpenAI 호환, Phonon, Pocket, Qwen3-TTS, Supertonic (12종). 한국어 비교는 `docs/reports/stage1_report.md` 참조 |
| Qwen Audio Agent 본체 | npm `qwen-audio-agent` | 1단계에서는 설치하지 않음. 4단계에서 확인 예정 |

### 2.1 가상환경 의존성 충돌

- `speech-to-speech[supertonic]==1.1.0`을 설치하면 pip가 `transformers 5.22.2`와 `torch 2.15.0+cu132`를 선택함.
- 이 조합에서는 Qwen3-TTS(`faster-qwen3-tts 0.5.4`가 사용하는 `qwen-tts-hf 0.1.1.post1`)가 모델을 불러오지 못함. 오류는 `AttributeError: property 'upsample' ... has no setter`임.
- speech-to-speech가 macOS용 의존성에 고정한 `transformers 5.18.0`에서도 `'MimiConfig' object has no attribute 'rope_theta'` 오류로 실패함.
- `faster-qwen3-tts`가 요구하는 하한 버전인 **`transformers 5.15.1`에서 정상 동작함.** 다만 speech-to-speech 1.1.0의 메타데이터는 `transformers>=5.18.0`을 요구하므로 pip가 충돌 경고를 냄.
- transformers 5.15.1에서 speech-to-speech의 주요 모듈(`qwen3_tts_handler`, `supertonic_tts_handler`, `nemo_asr_handler`, `api.openai_realtime.server`)은 import되고, `speech-to-speech --help`(명령: `serve`, `talk`, `local`)도 동작함. 실제 파이프라인 실행은 4단계에서 확인함.

### 2.2 Qwen3-TTS 백엔드

| 항목 | 확인 결과 |
|---|---|
| 패키지 기본값 | 모델 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, 백엔드 `ggml`, 양자화 `BF16`, 어텐션 `eager`, 장치 `cuda`, 화자 `Aiden`, 언어 `auto` (`arguments_classes/qwen3_tts_arguments.py`) |
| 한국어 화자 | CustomVoice 화자 9명 가운데 `sohee` |
| `ggml` 백엔드의 모델 지정 | 로컬 경로를 받지 않고 HF 모델 ID만 받음(`qwentts_cpp.models.resolve_gguf_paths`). 처음 실행할 때 GGUF 파일을 내려받음 |
| `ggml` 백엔드의 CUDA 버전 | `qwentts-cpp-python 0.5.0`의 `libggml-cuda.so`는 CUDA 12용(`libcudart.so.12`, `libcublas.so.12`). 가상환경의 torch는 CUDA 13(`2.15.0+cu132`)이므로, `nvidia-cuda-runtime-cu12`와 `nvidia-cublas-cu12`를 설치하고 `LD_LIBRARY_PATH`에 `.venv/lib/python3.11/site-packages/nvidia/{cublas,cuda_runtime}/lib`를 추가해야 동작함 |
| `torch` 백엔드 | 로컬 경로(`Korean_interface/models/tts/qwen3-tts-0.6b-custom`)로 동작함. HF 캐시의 같은 모델은 가중치가 없음 |

## 3. 하드웨어

| 항목 | 확인 결과 |
|---|---|
| GPU | NVIDIA RTX 6000 Ada Generation × 1, 49,140 MiB, 드라이버 595.99.02, CUDA 13.2 |
| CPU | 32코어 |
| NVFP4 | Ada 세대이므로 사용하지 않음 |


## vLLM 전용 venv (`.venv-vllm`, 2026-11-13)

| 항목 | 버전 |
|---|---|
| vllm | 0.33.0 |
| torch | 2.14.0+cu130 |
| transformers | 5.20.2 |
| Gemma 4 E4B 설정 인식 | `model_type=gemma4`, `Gemma4ForConditionalGeneration` (CPU 로 확인) |

`.venv`(speech-to-speech, transformers 5.15.1 고정)와 분리한다. 두 환경은 HTTP(127.0.0.1:8001)로만 연결된다.
speech-to-speech 1.1.0 의 chat-completions 백엔드는 텍스트 응답에 temperature/max_tokens 를 보내지 않으므로 `realtime_server/main.py` 가 요청마다 넣는다.
