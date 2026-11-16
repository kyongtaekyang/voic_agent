# 버전 확인 (영어 Agent)

프롬프트 7절에 따라 설치한 버전의 소스, `--help`, README로 문서 내용을 다시 확인한 결과입니다. 작성일은 2026-11-16입니다.

## 1. 설치 환경

| 항목 | 값 |
|---|---|
| 가상환경 | `.venv`(Moshi 서버 전용)와 `.venv-core`(Gateway·화면·ASR, CPU torch, transformers 5.15.1) 두 개입니다. moshi 0.2.13은 huggingface-hub<1.0, safetensors<0.8을 요구하고 transformers 5.15.1은 huggingface-hub≥1.5, safetensors≥0.8을 요구해서 한 환경에 둘 수 없습니다(`bench/setup_venvs.sh`). 한국어 Agent의 `.venv`와도 분리했습니다. |
| moshi (PyPI) | 0.2.13 |
| torch | 2.9.1 (CUDA 12.8 휠) |
| sphn (Opus 입출력) | 0.2.1 |
| sentencepiece | 0.2.3 |
| 참고 소스 | `github.com/kyutai-labs/moshi` 커밋 `e6a55d2` (2026-05-16), `english-agent/.src/moshi` (저장소에 올리지 않음) |
| Rust 도구 | 서버에 `cargo`, `rustc`가 없습니다. Rust/Candle 백엔드는 2단계에서 설치합니다. |

전체 패키지 목록은 `logs/venv_freeze.txt`에 있습니다(저장소에 올리지 않음).

## 2. Moshi PyTorch 서버 (`python -m moshi.server`)

| 문서·프롬프트 내용 | 설치 버전에서 확인한 내용 | 근거 |
|---|---|---|
| 기본 포트 8998 | 같습니다. `--host` 기본값은 `localhost`입니다. | `moshi/server.py:174-175` |
| BF16 기준 GPU 약 24GB 이상 | 기본 dtype은 bfloat16이고, `--half`를 주면 float16입니다. 메모리는 아직 측정하지 않았습니다. | `server.py:193` |
| 남성·여성 모델 선택 | `--hf-repo`로 고릅니다. 기본값은 `kyutai/moshiko-pytorch-bf16`(남성)입니다. 여성 모델은 `kyutai/moshika-pytorch-bf16`입니다. | `moshi/models/loaders.py:35`, `server.py:184` |
| 오디오 24kHz | Mimi 샘플레이트 24kHz, 프레임 12.5Hz(80ms)입니다. | `loaders.py:84` |
| LoRA 적용 | `--lora-weight`, `--no_fuse_lora` 옵션이 있습니다. 8단계(파인튜닝)에서 사용합니다. | `server.py:187, 191` |
| (문서에 없음) 정적 화면 | `--static`을 주지 않으면 HF `kyutai/moshi-artifacts`에서 `dist.tgz`를 받아서 `/`에 웹 화면을 띄웁니다. `--static none`이면 화면을 띄우지 않습니다. | `server.py:243-255` |
| (문서에 없음) 동시 세션 | 서버 하나가 `asyncio.Lock`으로 **한 번에 한 연결만** 처리합니다. 두 번째 연결은 앞 연결이 끝날 때까지 기다립니다. 동시 3·5세션(KPI)은 서버 프로세스를 여러 개 띄우거나 배치 처리로 바꿔야 합니다. | `server.py:160-167` |
| (문서에 없음) 외부 터널 | `--gradio-tunnel`은 미국을 거치는 터널을 엽니다. **음성 외부 전송 금지 원칙에 따라 사용하지 않습니다.** | `server.py:177, 280-284` |

## 3. 텍스트 토큰 추출 (프롬프트 2.2절 1번)

**결론: 공식 WebSocket 규격으로 이미 제공됩니다. PyTorch, Rust 백엔드 모두 코드를 수정하지 않아도 텍스트 토큰을 받을 수 있습니다.**

WebSocket `/api/chat`의 메시지는 첫 바이트가 종류를 나타냅니다(`rust/protocol.md`).

| 종류 | 방향 | 내용 |
|---|---|---|
| `0x00` Handshake | 서버 → 화면 | 연결 준비 완료. PyTorch는 1바이트만, Rust는 버전 필드 8바이트를 붙입니다. |
| `0x01` Audio | 양방향 | Ogg/Opus 24kHz 모노 |
| `0x02` Text | 서버 → 화면 | Moshi 텍스트 토큰 하나(UTF-8). SentencePiece의 `▁`는 공백으로 바뀌어 옵니다. 패딩 토큰(0, 3)은 보내지 않습니다. |
| `0x03`~`0x06` | - | Control, Metadata, Error, Ping. 전이중 대화에서는 거의 쓰지 않습니다. |

- PyTorch: `moshi/server.py:86-92`에서 매 프레임 텍스트 토큰을 `b"\x02" + 텍스트`로 보냅니다.
- Rust: `rust/moshi-backend/src/stream_both.rs:189-222`(종류 정의), `:255`(전송)에서 같은 형식을 씁니다.

## 4. 텍스트와 음성의 시간 관계 (프롬프트 2.2절 2번, 소스 기준 예측)

실측은 GPU가 필요하므로 2단계에서 합니다. 아래 내용은 소스를 읽고 예측한 것입니다.

- **PyTorch 서버:** `LMGen`은 코드북 지연(`delays = [0, 0, 1, 1, ...]`, `loaders.py:118`)을 되돌린 뒤 같은 시점의 텍스트와 음성 토큰을 함께 돌려줍니다(`models/lm.py:774-781`). 서버는 한 프레임마다 **음성(0x01)을 먼저, 텍스트(0x02)를 나중에** 보냅니다(`server.py:85-92`). 따라서 프로토콜 수준에서는 텍스트가 음성보다 앞서지 않습니다. 텍스트 토큰은 실제로는 한 프레임(80ms) 먼저 만들어지지만, 지연을 맞추느라 붙잡혀 있습니다. 이 80ms를 쓰려면 서버 코드를 수정해야 합니다.
- **Rust 서버:** 텍스트를 샘플링한 직후에 보내고(`stream_both.rs:369-374`), 음성은 그 뒤 Mimi 디코딩을 거쳐 나갑니다. 따라서 텍스트가 음성보다 약 한 프레임 이상 앞설 것으로 예상합니다.
- **영향:** 한 단어의 텍스트 토큰은 그 단어 소리가 시작될 무렵에 도착합니다. 여러 단어로 된 구절(예: "where do you live")은 마지막 단어까지 들어와야 판정할 수 있으므로, 앞 단어들의 소리는 이미 나간 뒤입니다. **차단을 소리보다 먼저 하려면 Gateway에서 출력 음성을 잠시 붙잡아 두는 지연 버퍼가 필요할 가능성이 큽니다** (프롬프트 2.2절 5번의 200~400ms 방안). 2단계 실측 결과로 버퍼 길이를 정합니다.

## 5. Qwen Audio Agent 호환성 (프롬프트 2.4절)

**결론: 그대로는 호환되지 않습니다. 1차 PoC는 자체 경량 Gateway로 진행합니다.**

| 항목 | Qwen Audio Agent `speech-to-speech` Provider | Moshi 서버 |
|---|---|---|
| 주소 | `QWEN_AUDIO_REALTIME_URL=ws://127.0.0.1:8765/v1/realtime` (`docs/Qwen_Audio_Agent.md:151-152`) | `ws://127.0.0.1:8998/api/chat` |
| 메시지 형식 | OpenAI Realtime 호환 JSON 이벤트 (`session.update`, `input_audio_buffer.append` 등). 한국어 Agent가 쓰는 speech-to-speech 1.1.0과 같은 규격 | 첫 바이트로 종류를 구분하는 바이너리 |
| 음성 형식 | PCM16 base64 | Ogg/Opus 24kHz |
| 턴 구조 | VAD로 턴을 나누고 응답(`response.*`)을 만듦 | 전이중. 턴 개념이 없고 항상 듣고 말함 |

어댑터를 만들려면 Opus↔PCM 변환, 연속 출력을 `response.*` 이벤트로 끊어 내는 규칙, 끼어들기 이벤트 대응이 필요합니다. 전이중이라는 Moshi의 장점을 턴 기반 규격에 맞추느라 잃게 되므로 비용 대비 이득이 작습니다. 추정 개발량은 3~5일이며, 실제 연동 검증에는 GPU가 필요합니다.

## 6. Nemotron ASR Streaming 0.6B (병행 감시)

- 한국어 Agent 1단계와 같은 모델 파일(`/home/yang/pgm/Korean_interface/models/stt/nemotron-3.5-asr-streaming-0.6b`)과 transformers 경로(`AutoModelForRNNT`)를 사용합니다.
- 언어 설정은 `en-US`(고정)와 `auto`(발화마다 `<en-US>`, `<ko-KR>` 같은 언어 태그를 붙임)를 비교했습니다. 결과는 `docs/reports/lang_probe_en.json`에 있습니다.
