# 1단계 보고서: TTS 조사, ASR Streaming 한국어·CPU 확인

작성일은 2026-11-13입니다. 원자료는 같은 폴더의 `asr_cpu_rtf_t4.json`, `asr_cpu_rtf_t8.json`, `tts_qwen3_*.json`, `tts_supertonic_t4.json`, `tts_supertonic_t8.json`, `tts_roundtrip.json`에 있습니다. `asr_cpu_rtf_t8_aborted_20261112.log`는 이전에 중단된 실행의 로그이므로 결과로 보지 않습니다. 버전 확인 내용은 `docs/version_check.md`에 정리했습니다.

## 0. 설계 문서와 다른 점

1. 프롬프트는 `docs/`에 md 파일 3개가 있다고 가정하지만, 실제로는 Nemotron, Qwen Audio Agent 2개만 있습니다. Moshi 문서는 없지만 한국어 Agent는 Moshi를 사용하지 않으므로 문제가 없습니다.
2. Qwen Audio Agent 문서와 speech-to-speech 패키지는 Realtime 서버 주소로 `:8765`를 사용합니다. 이 서버에서는 8765가 이미 사용 중이므로, `ada-ops/ports.md`의 결정대로 **8766**을 사용합니다.
3. **speech-to-speech 1.1.0에 이미 Realtime 서버가 들어 있습니다.** 이 패키지는 OpenAI Realtime 호환 WebSocket 서버(`/v1/realtime`), Nemotron 3.5 STT 처리기, OpenAI 호환 LLM 클라이언트(vLLM 연결용), VAD와 Smart Turn(턴 종료 판정 모델), 끼어들기 취소, TTS 12종을 포함합니다. 프롬프트 4절은 `realtime_server/`의 모듈을 모두 직접 구현하도록 되어 있지만, 이 패키지를 기반으로 쓰고 한국어 전용 모듈(출력 검사, 텍스트 정규화, 어미 기반 보조 판정)만 추가하는 편이 개발량과 위험이 작습니다. **4단계 전에 방향 결정이 필요합니다.**
4. 패키지의 Nemotron 처리기는 transformers가 아니라 NeMo(`speech-to-speech[nemo]`)를 사용합니다. 1단계 측정은 모델 카드의 transformers 경로로 했으므로, 4단계에서 NeMo 경로의 지연을 다시 확인해야 합니다.

## 1. Nemotron 3.5 ASR Streaming 0.6B: 한국어와 CPU 실행

### 1.1 기능 확인

- **한국어:** `ko-KR`을 지원하고, "Transcription-ready" 등급입니다. 모델 카드의 FLEURS 한국어 CER은 7.1~8.3%입니다.
- **발화 종료 판정:** 이 모델에는 없습니다. 따라서 턴 종료는 VAD 또는 Smart Turn과 어미 기반 보조 판정으로 처리해야 합니다(프롬프트 1.5절). NVIDIA가 안내하는 별도 EOU 모델(Parakeet Realtime EOU 120M)은 세 md 파일에 없는 모델이므로 도입하지 않았습니다.
- **모델 카드 예시 코드의 결함:** lookahead 0에서 예시 코드를 그대로 쓰면 `stft` 오류로 실패합니다. 벤치마크에서는 앞부분을 0으로 채워서 해결했고, 실시간 서버에도 같은 처리가 필요합니다.

### 1.2 CPU 실시간 처리 속도

- 평가 음성은 `Korean_interface/data/voices`의 음성 복제용 참조 녹음 가운데 중복을 제거한 8개입니다. 총 길이는 약 217초이고, 성인 화자의 한국어 낭독입니다. 화자 이름은 파일명에 쓰지 않았고, 원본 파일은 복사하지 않고 심볼릭 링크로 연결했습니다.
- 녹음에 딸린 `reference.txt`가 실제 낭독 내용과 달라서, **이번에는 CER을 계산하지 않았습니다.** 출력을 확인한 2개 클립에서는 인식 결과가 자연스러운 한국어 문장이었지만, 정확도는 2단계 평가 세트로 측정해야 판단할 수 있습니다.
- 청크를 가능한 한 빨리 넣는 방식으로 측정했습니다. RTF는 총 처리 시간을 총 음성 길이로 나눈 값입니다.
- 측정 중 서버의 부하 평균(load average)은 2~14였습니다. 4스레드 측정은 가상환경 설치와 겹쳤고, 8스레드 측정은 Qwen3-TTS torch 측정(부하 평균 약 11.5)과 겹쳤습니다. 따라서 두 결과 모두 이 부하의 영향을 받았고, 실제 값은 이보다 조금 나을 수 있습니다.

| 조건 | 청크 지연 | RTF (4스레드) | RTF (8스레드) | 청크당 처리 (8스레드) | CPU 실시간 가능 |
|---|---|---|---|---|---|
| 오프라인(파일 전체) | - | 0.089 | 0.066 | - | (참고값) |
| lookahead 0 | 80 ms | 1.473 | 1.223 | 94 ms | **불가** |
| lookahead 3 (기본값) | 320 ms | 0.668 | 0.437 | 139 ms | 가능 |
| lookahead 6 | 560 ms | 0.444 | 0.298 | 166 ms | 가능 |
| lookahead 13 | 1120 ms | 0.282 | 0.192 | 215 ms | 가능 |

**판단:** 프롬프트의 기준(RTF 1.0 미만이면 CPU 유지)에 따라 **CPU 실행을 유지합니다.** 응답 지연 목표(p50 1.0초)를 고려하면 lookahead 3, 8스레드가 적합합니다. 이 조건에서는 발화가 끝난 뒤 인식이 확정되기까지 약 0.46초가 걸릴 것으로 예상합니다(청크 지연 320ms + 청크 처리 139ms). 다만 이 값은 세션 1개 기준입니다. 동시 세션이 늘면 CPU 경합이 생기므로 7단계에서 다시 측정해야 합니다.

## 2. TTS 조사 (프롬프트 1.4절)

### 2.1 speech-to-speech 1.1.0에 포함된 TTS 모듈

| 모듈 | 한국어 | 스트리밍 | 실행 장치 | 라이선스 (모델) | 판단 |
|---|---|---|---|---|---|
| **Qwen3-TTS** (faster-qwen3-tts) | 지원 (10개 언어, 한국어 화자 `sohee`) | 지원 (청크 단위) | GPU 필수 (CPU는 RTF 1.9로 실시간 불가) | Apache 2.0 | **1순위 후보, 실측함** |
| **Supertonic-3** | 지원 (31개 언어) | 미지원 (문장 단위 합성) | CPU (ONNX) | OpenRAIL-M (사용 제한 조항 있음) | **2순위 후보, 실측함** |
| OmniVoice (k2-fsa) | 지원 | 확인 필요 | GPU | HF에 라이선스 표시 없음 | 라이선스 불명확으로 보류 |
| Drama-3 (Fish Audio) | 지원 | 지원 | GPU (별도 SGLang 서버) | 기타(other) 라이선스 | 라이선스 확인과 별도 서버가 필요하므로 보류 |
| Facebook MMS (`mms-tts-kor`) | 지원 | 미지원 | GPU/CPU | CC-BY-NC 4.0 | **상업적 이용 불가로 제외** |
| Kokoro, Kitten, Phonon, Pocket, ChatTTS | 미지원 | - | - | - | 제외 |
| MiniMax, OpenAI | - | - | 클라우드 API | - | **외부 전송 금지로 제외** |

### 2.2 실측 결과

평가 문장 6개를 사용했습니다(`bench/data/tts_sentences_ko.txt`). 5개는 시스템 프롬프트 규칙대로 숫자를 읽는 형태로 쓴 문장이고, 마지막 1개는 숫자를 그대로 둔 문장(`3개`, `10:30`)입니다. 지연과 CER 평균은 앞의 5개 문장으로 계산했습니다. 명료도는 TTS 출력을 Nemotron ASR로 다시 인식해서 원문과 비교한 CER로 대신 측정했습니다.

speech-to-speech의 Qwen3-TTS 기본 설정은 **1.7B CustomVoice, `ggml` 백엔드, BF16, eager 어텐션, GPU**입니다. 처음에는 `torch` 백엔드로 측정했지만 패키지 기본 경로와 달라서, `ggml` 백엔드와 1.7B 모델을 추가로 측정했습니다. 청크 크기는 패키지 기본값인 8을 사용했습니다.

| 조건 | 첫 음성 지연 (평균) | RTF | GPU 메모리 | 왕복 인식 CER | 비고 |
|---|---|---|---|---|---|
| **Qwen3-TTS 0.6B, ggml, GPU** | **0.08초** | 0.10 | 3,566 MiB | 4.5% | **권고 조건** |
| Qwen3-TTS 1.7B, ggml, GPU (패키지 기본값) | 0.09초 | 0.13 | 5,494 MiB | 8.6% | |
| Qwen3-TTS 0.6B, torch, GPU | 0.27초 | 0.28 | 3,036 MiB | 3.3% | 측정 중 부하 평균 약 11.5 |
| Qwen3-TTS 0.6B, ggml, CPU | 1.20초 | **1.91** | 0 | - | 실시간 합성 불가 |
| Supertonic-3, CPU 4스레드 | 0.99초 | 0.19 | 0 | **0.8%** | 문장 전체를 합성한 뒤 재생 |
| Supertonic-3, CPU 8스레드 | 1.04초 | 0.20 | 0 | (같은 모델) | |

- 왕복 인식 CER은 숫자를 읽는 형태로 쓴 문장 5개의 평균입니다. 숫자를 그대로 둔 문장(`3개`, `10:30`)은 모든 TTS가 "십시 삼십분에 맞나요?", "삼십에 만나요"처럼 잘못 읽었습니다(CER 38~52%).
- Qwen3-TTS는 샘플링 방식(temperature 0.9)으로 생성하므로, 같은 문장도 실행할 때마다 발음이 조금씩 달라집니다. "세 개"를 "색이", "세계에"처럼 발음한 경우가 있었고, 1.7B가 0.6B보다 CER이 높게 나온 것도 이 변동의 영향일 수 있습니다. 문장 5개로는 두 모델의 품질 차이를 판단할 수 없으므로, 4단계에서 temperature를 낮추고 문장 수를 늘려 다시 비교하겠습니다.
- `ggml` 백엔드는 CUDA 12용으로 빌드되어 있어서, CUDA 13 기반의 가상환경에서는 CUDA 12 런타임을 추가로 설치하고 `LD_LIBRARY_PATH`를 지정해야 동작했습니다. 자세한 내용은 `docs/version_check.md`에 적었습니다.
- 두 TTS 모두 숫자와 기호를 제대로 읽지 못하므로, **`text_normalizer.py`(숫자·기호를 읽는 형태로 변환)는 필수입니다.**
- 합성된 음성은 서버의 `korean-voice-agent/bench/out/`에 있습니다. 음질과 아동 대상 적합성은 직접 들어 보고 판단해 주셔야 합니다.

### 2.3 외부 후보 (참고용, 도입하지 않음)

같은 서버의 `Korean_interface` 프로젝트에는 CosyVoice3, GPT-SoVITS, Chatterbox Multilingual, MeloTTS-Korean, MiniCPM-o 4.5가 설치되어 있습니다. 프롬프트 1.4절에 따라 포함된 모듈 가운데 적합한 것이 있으므로, 이 외부 모델들은 측정하지 않았고 도입하지 않았습니다.

### 2.4 권고

- **Qwen3-TTS 0.6B CustomVoice를 `ggml` 백엔드로 GPU에서 실행하는 구성을 기본 TTS로 권고합니다.** speech-to-speech에 포함되어 있고, Apache 2.0이며, 측정한 후보 가운데 첫 음성 지연(0.08초)이 가장 짧습니다.
- **패키지 기본값인 1.7B 대신 0.6B를 권고합니다.** 측정한 범위에서는 1.7B가 지연과 명료도에서 나은 점이 없었고, GPU 메모리는 약 1.9GB 더 사용했습니다. GPU가 부족한 이 서버에서는 메모리 차이가 중요합니다. 다만 음질은 직접 들어 보고 판단해 주셔야 합니다.
- **Supertonic-3(CPU)은 GPU를 쓸 수 없을 때의 대체 수단으로 둡니다.** 명료도는 가장 좋지만 첫 음성이 약 1초 걸려서, 이것만으로는 지연 목표를 맞추기 어렵습니다. 또한 OpenRAIL-M 라이선스에는 사용 제한 조항이 있으므로 상업적 이용 전에 법무 검토가 필요합니다. Qwen3-TTS는 CPU에서 실시간 합성이 불가능하므로 대체 수단이 될 수 없습니다.
- 4단계에서 연결할 Gemma 4 경로와 합치면, 예상 지연은 대략 턴 판정(미측정, VAD 무음 기준에 따라 0.2~0.5초로 추정) + STT 확정 약 0.46초 + LLM 첫 문장(미측정) + TTS 약 0.08초입니다. p50 1.0초를 맞추려면 턴 판정과 LLM 첫 문장 시간을 매우 짧게 줄여야 합니다.

## 3. 다음 단계로 넘어가기 전에 결정이 필요한 사항

1. **GPU 확보 (가장 시급함):** baek 사용자의 `app_flash.py`(포트 7860)가 GPU 메모리를 30.8~42.7GB 사용하고 있습니다. 03:40 기준 GPU 여유는 약 3GB입니다. 이 상태에서는 Gemma 4 26B A4B 4비트(약 17GB와 KV 캐시)와 TTS(약 3.6GB)를 올릴 수 없으므로, **3단계에 들어가기 전에 GPU 사용 일정 조율이 필요합니다.** gpu-mode 스크립트는 다른 사용자의 프로세스를 멈추지 않습니다.
2. **Realtime 서버 구현 방식:** speech-to-speech 1.1.0을 기반으로 사용할지, 프롬프트대로 직접 구현할지 정해 주십시오. 저는 기반으로 사용하는 방안을 권고합니다.
3. **TTS 선택:** Qwen3-TTS 0.6B CustomVoice(`ggml`, GPU)를 기본으로 하고 Supertonic-3(CPU)을 대체 수단으로 두는 권고안을 승인해 주십시오. 패키지 기본값(1.7B)과 다르게 설정하는 것도 포함됩니다.
4. **2단계 평가 자료:** 아동 한국어 발화 평가 세트의 위치를 알려 주십시오. 비교 대상인 "기존 Whisper 파인튜닝 모델"이 무엇인지도 알려 주십시오. 서버 캐시에는 `seastar105/whisper-small-komixv2`와 `vitaldb/qwen3-asr-1.7b-kor-or-lora`가 있어서 어느 것인지 판단하기 어렵습니다.

## 4. 이번 작업에서 만든 파일

| 파일 | 내용 |
|---|---|
| `bench/asr_cpu_rtf.py` | lookahead 0 경계 조건 수정, 생성 스레드 예외 처리 추가 |
| `bench/run_asr_sweep.sh` | 4·8스레드 ASR 측정 실행 스크립트 |
| `bench/summarize_asr.py` | ASR 측정 결과 요약 |
| `bench/tts_bench.py` | TTS 첫 음성 지연, RTF, GPU 메모리 측정 |
| `bench/tts_roundtrip.py` | TTS 출력의 왕복 인식 CER 측정 |
| `bench/setup_venv.sh` | 프로젝트 가상환경(`.venv`, Python 3.11) 생성 |
| `bench/data/ko_ref/` | ASR 평가 음성 링크 8개 |
| `bench/data/tts_sentences_ko.txt` | TTS 평가 문장 |
| `docs/version_check.md` | 버전 확인 기록 |
| `../ada-ops/gpu_layout.md` | GPU 실측값과 1단계 재확인 결과 추가 |
