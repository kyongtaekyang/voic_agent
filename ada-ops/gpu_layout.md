# GPU 배치 (RTX 6000 Ada 48GB 한 장)

## 0단계 측정 (2026-11-12)

- GPU: NVIDIA RTX 6000 Ada Generation × 1, 49,140 MiB, 드라이버 595.99.02, CUDA 13.2
- NVFP4 는 Blackwell 전용이므로 사용하지 않음. BF16 / FP8 / 4비트(AWQ 등)만 사용.

이 프로젝트를 시작하기 전부터 GPU를 쓰고 있던 프로세스:

| pid | 소유자 | 프로그램 | GPU MiB |
|---|---|---|---|
| 1416739 | baek | `app_flash.py --server_port 7860` | 30,260 |
| 108978 | yang | Korean_interface `qwen_tts_worker.py` | 2,826 |
| 1072827 | yang | EngVoice `engvoice_realtime_demo.py --device cuda` | 3,052 |

합계 약 36 GB 사용, **여유 약 12 GB**. 실시간 모드(목표 40 GB 이하)와 후처리 모드(약 40 GB)를 올릴 수 없음.
gpu-mode 는 다른 사용자 프로세스를 멈추지 않으며, 여유가 부족하면 전환을 거부한다.

### 1단계 재확인 (2026-11-13)

| 시각(UTC) | baek `app_flash.py` | yang `qwen_tts_worker.py` | GPU 여유 |
|---|---|---|---|
| 03:08 | 30,796 MiB | 2,694 MiB | 14,894 MiB |
| 03:40 | **42,716 MiB** | 2,694 MiB | **2,974 MiB** |

baek 의 프로세스는 30~43 GB 사이에서 변동한다. EngVoice 데모(pid 1072827)는 종료됨.
이 상태로는 한국어 실시간 모드(Gemma 26B A4B 4비트 + TTS)를 올릴 수 없으므로, 3단계 전에 GPU 사용 일정 조율이 필요하다.

### 실측값

| 서비스 | 조건 | GPU MiB (nvidia-smi 프로세스) | 비고 |
|---|---|---|---|
| Nemotron ASR Streaming 0.6B | CPU | 0 | CPU RTF: lookahead 3 에서 0.44(8스레드) / 0.67(4스레드) |
| Qwen3-TTS 12Hz 0.6B CustomVoice | faster-qwen3-tts, **ggml**, BF16, 세션 1개 (권고 조건) | 3,566 | TTS 는 Realtime 서버 프로세스 안에서 실행된다. modes.conf 의 ko-realtime-server 4000 이 TTS 여유분 포함 값 |
| Qwen3-TTS 12Hz 0.6B CustomVoice | faster-qwen3-tts, torch, BF16, CUDA graph, 세션 1개 | 3,036 | torch 최대 할당 2,372 MiB |
| Qwen3-TTS 12Hz 1.7B CustomVoice | faster-qwen3-tts, ggml, BF16, 세션 1개 (speech-to-speech 기본값) | 5,494 | |
| Supertonic-3 | CPU (ONNX) | 0 | 문장 단위 비스트리밍, 첫 음성 약 1.0초 |

## 모드별 배치 (추정치, 실측으로 갱신)

| 모드 | 순서 | 서비스 | vLLM `--gpu-memory-utilization` | 추정 MiB |
|---|---|---|---|---|
| realtime | 10 | Gemma 4 E4B, vLLM 온라인 FP8 (:8001). 26B A4B 는 GPU 여유 확보 후 3단계에서 재검토 | 0.22 (≈10,800 MiB, korean_agent/config/services.yaml llm.gpu_memory_utilization) | 11,000 |
| realtime | 20 | 한국어 Realtime 서버 (:8766), Qwen3-TTS 0.6B ggml 포함 | - | 4,000 |
| realtime | 50 | Moshi Rust INT8 (:8998) | - | 12,000 |
| realtime | - | Nemotron ASR Streaming 0.6B | CPU 우선 | 0 |
| batch | 10 | Nemotron Transcribe 30B FP8 (vLLM :8002) | 0.85 | 40,000 |
| train | 10 | moshi-finetune LoRA | - | 40,000 |

vLLM 은 기동 시점의 전체 메모리 비율로 미리 잡으므로, realtime 모드에서는 Gemma vLLM 을 **가장 먼저** 올린다.

## 영어 단독 프로필 en-solo (2026-11-16 추가)

| 모드 | 순서 | 서비스 | 추정 MiB |
|---|---|---|---|
| en-solo | 10 | Moshi PyTorch BF16 (:8998, `services/en-moshi-bf16.sh`) | 24,000 (프롬프트 2.1절 "약 24GB 이상", 실측으로 갱신) |

- 1단계(브라우저 왕복 대화, 메모리·지연 측정)에만 쓴다. 한국어 서비스는 이 모드로 바꿀 때 gpu-mode 가 멈춘다.
- realtime 모드의 `en-moshi`(Rust/Candle INT8)는 아직 미구현이다. 서버에 Rust 도구(cargo)가 없어서 2단계에서 설치한다.
- 2026-11-16 06:00 UTC 기준 GPU 여유는 약 10.6 GB (baek `app_flash.py` 30.8 GB, yang Korean_interface 7.4 GB). en-solo 도 지금은 올릴 수 없다.

## 야간 배치 summary 모드 (2026-11-16 추가)

| 모드 | 순서 | 서비스 | 추정 MiB |
|---|---|---|---|
| summary | 10 | Gemma 4 vLLM (:8001) 단독. 전사 후 세션 요약용 | 11,000 |

`ada-ops/nightly-batch`가 batch(Transcribe 30B) → summary(Gemma) → stop 순서로 전환한다 (프롬프트 2.2절 4번).
`services/transcribe-vllm.sh`는 미검증이다 (가중치·NeMo 플러그인 미설치, FP8 로 48GB 에 올라가는지 GPU 시간에 확인).

## 자동 기동: gpu-wait (2026-11-17 추가)

`ada-ops/gpu-wait` 가 GPU 사용량을 1분마다 기록하고, 여유가 충분히 오래 유지될 때 `gpu-mode <모드>` 를 부른다.
baek `app_flash.py` 처럼 30~43 GB 를 오가는 프로그램이 있으므로 **지금 여유가 아니라 최근 최고 사용량으로 판단한다.**

- 판단: `전체 - 다른 프로그램의 최근 2시간(window) 최고 사용량 - 우리 사용량 - 여유분 3000 MiB >= gpu-mode need <모드>` 가 5번 연속(5분)
- 기록이 2시간 쌓이기 전에는 올리지 않는다. 기록: `logs/gpu-usage.tsv` (7일 보관), 마지막 판단: `run/gpu-wait.state`
- 양보: 올린 뒤 실제 여유가 1500 MiB 아래로 두 번(10초 간격) 내려가면 우리 서비스를 내리고 다시 기다린다. 다른 사용자의 프로세스는 건드리지 않는다
- 다른 모드가 올라가 있거나 nightly-batch 가 실행 중이면 기다린다. gpu-mode 가 3번 연속 실패하면(포트 충돌, 가중치 없음 등) 감시를 끝낸다
- 사용: `gpu-wait start realtime`, `gpu-wait status`, `gpu-wait stop`(감시만 중지), `gpu-wait check realtime`(한 번만 판단). 옵션은 `gpu-wait help`
- 2026-11-17 02:29 UTC 부터 `realtime` 모드로 감시 중 (시간대 제한 없음). 지금 need 는 15000 MiB (Gemma E4B 11000 + Realtime 서버 4000. Moshi Rust INT8 은 미구현이라 제외)
- 서버를 재시작하면 감시도 멈춘다 (crontab @reboot 미등록)

`gpu-mode` 에 `need <모드>`, `mem` 명령을 추가했다. 또한 GPU 프로세스가 nvidia-smi 조회와 ps 조회 사이에 끝나면
`pipefail` 때문에 status 가 실패해서, 기동에 성공하고도 종료 코드 1 을 돌려주던 문제를 고쳤다.
