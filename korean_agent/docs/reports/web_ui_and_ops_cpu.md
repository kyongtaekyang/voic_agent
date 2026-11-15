# 한국어 Agent 화면, 영어 발화 감지, 운영 스크립트 (2026-11-13, GPU 미사용)

`realtime_server_cpu_smoke.md` 이후 작업이다. 모두 CPU 로만 확인했다. 단위 시험 103개 통과.

**단계 구분:** 화면, 끼어들기, 지연 설정은 명세상 5단계 산출물을 미리 만든 것이다. 2단계(아동 평가 세트), 3단계(Gemma 4 실측), 4단계는 아직 끝나지 않았으므로, 이 문서의 내용은 **사전 구현이며 단계 승인 대상이 아니다.** 실제 모델로 다시 확인한 뒤 해당 단계에서 보고한다.

## 1. 대화 이력 정리 (출력 검사 보강)

- 문제: 출력 검사는 TTS 로 가는 문장만 바꾸고, speech-to-speech 는 LLM 원문을 대화 이력에 따로 기록한다. 차단한 문장이 다음 턴 프롬프트에 다시 들어갈 수 있었다.
- 해결: `realtime_server/history_guard.py` 가 `AssistantMessage.to_chat_item()` 을 감싸서, 처음 걸린 문장부터 끝까지를 대체 문장으로 바꾼 뒤 이력에 넣는다 (`OutputGuard.sanitize`).
- 확인: 2턴 스모크에서 두 번째 LLM 요청의 assistant 이력이 "사과가 3개 있어요. 지금은 10:30이에요. 그 이야기는 하지 않을게요. 우리 하던 공부를 계속해 볼까요?" 로 바뀌어 있었다. 화면이 받는 `response.output_item.done` 텍스트도 실제로 읽은 문장과 같다.

## 2. 영어 발화 감지 (language_mismatch)

- 측정(`bench/lang_mismatch_probe.py`, `docs/reports/lang_mismatch_probe.json`): Nemotron 을 `language=ko-KR` 로 고정하면 영어 발화는 **빈 문자열** 또는 "이" 같은 한두 음절로 나온다. 한글 비율로는 감지할 수 없었다. `language=auto` 는 영어를 정확히 받아 쓰고 대부분 `<en-US>` 태그를 붙였다(3문장 중 1개는 태그 없음).
- 결정: 한국어 정확도를 지키기 위해 `ko-KR` 을 유지하고, 확정 결과의 한글 음절이 발화 길이에 비해 적으면(초당 1.5 음절 미만, 0.6초 이상 발화) 같은 구간을 `auto` 로 한 번 더 인식해 언어 태그(없으면 한글 비율)로 판단한다.
- 확인: 영어 합성 음성 2턴 → 두 번째 턴에 `language_mismatch` 발생, 화면에 `kva.language_mismatch` 전달. 한국어 음성 2턴 → 재인식 없음.
- 한계: 합성 음성으로만 확인했다. 아동의 실제 영어·한국어 섞인 발화로 다시 확인해야 한다.

## 3. 사용자 발화 원문 로그

- `features.log_transcripts: false` 인데도 speech-to-speech 의 STT/TTS 처리기가 콘솔에 `USER: ...`, `ASSISTANT: ...` 를 찍어 서비스 로그에 아동 발화가 남는 것을 발견했다. 공용 console(`speech_to_speech.pipeline.console`)을 `quiet` 로 바꿔서 막았다(스모크 로그의 `USER:` 0줄 확인).
- `language_mismatch` 이벤트의 발화 원문도 `log_transcripts` 가 켜져 있을 때만 남긴다.

## 4. 한국어 Agent 화면 (7201)

- `web/server.py`(FastAPI) + `web/static/`(index.html, app.js, capture-worklet.js, style.css). 화면 문구는 모두 한국어.
- 사용자 화면: 대화 시작·종료, 마이크 상태, Agent 상태(듣는 중·생각하는 중·말하는 중), 실시간 자막(부분 자막은 스냅샷으로 교체), 영어 발화 안내와 영어 Agent 링크, "현재 점검 중", "다른 사람이 대화하고 있습니다".
- 개발자 화면(`?dev=1`, `web.dev_screen: true` 일 때만): 운영 모드, 모델과 정밀도, GPU 메모리, 단계별 지연(서버 기준), 발화 끝→첫 음성(화면 기준), 끼어들기 감지 지연, 출력 검사 차단 기록.
- 화면 서버는 브라우저 이벤트를 허용 목록으로 거른다. 시스템 프롬프트를 바꾸는 `instructions`, `tools`, 텍스트 입력, `response.create` 는 막고 턴 판정 설정은 고정한다. 규격은 `docs/event_schema.md`.
- 끼어들기: `speech_started` 를 받으면 브라우저가 남은 재생을 즉시 멈추고 들려준 길이를 `conversation.item.truncate` 로 보낸다.
- **끼어들기 후 대화 이력 (명세 2.1절 5번)**: speech-to-speech 1.1.0 은 truncate 를 받기만 하고 이력을 고치지 않는다. 또 TTS 가 실시간보다 빨라 `response.done` 이 재생보다 먼저 오므로, 끼어들어도 응답 전체가 이력에 남았다(500ms 에서 끊은 시험에서 확인). 중계 서버(`web/relay.py: PlaybackTracker`)가 문장별 음성 시작 위치를 기록해 두었다가, truncate 를 받으면 그 assistant 항목을 지우고(`conversation.item.delete`) 끝까지 들려준 문장만으로 다시 만든다(`conversation.item.create`).
  - 2,500ms 에서 끊음 → 다음 LLM 요청 이력: "사과가 세 개 있어요." 만 남음 (순서 system, user, assistant, user 유지)
  - 300ms 에서 끊음 → assistant 항목 삭제. speech-to-speech 가 답이 없는 앞 턴의 사용자 발화도 이력에서 뺐다(system, user).
  - 한계: 화면 서버를 거치지 않고 8766 에 직접 붙는 클라이언트에는 적용되지 않는다. 이력에는 정규화된 문장("세 개")이 들어간다.
- 확인 범위: 중계 서버를 거친 스모크(한국어 2턴, 영어 2턴)에서 모든 `kva.*` 이벤트 도착 확인, 정적 파일 응답 200, `node --check` 문법 확인. **브라우저에서 마이크로 대화하는 시험은 아직 하지 않았다**(브라우저 자동화 연결 불가, GPU 미사용). 실제 모델을 올린 뒤 확인한다.

## 5. 지연 관련 발견 (5단계 조정 대상)

speech-to-speech 1.1.0 기본값이 지연을 정한다. `config/services.yaml` `turn:` 에 명시해 조정할 수 있게 했다.

| 설정 | 기본값 | 영향 |
|---|---|---|
| `min_speech_ms` | 384 | 발화 시작(끼어들기) 인정에 필요한 음성 길이. 측정된 감지 지연 384~391ms 가 이 값과 같다. 명세의 "끼어들기 후 300ms 이내 재생 중단"을 감지 시점부터로 보면 브라우저 중단은 즉시이고, 발화 시작부터로 보면 이 값을 낮춰야 한다 |
| `speech_pad_ms` | 500 | `speech_started` 의 `audio_start_ms` 가 이만큼 앞당겨진다 (중계 서버가 보정) |
| `speculative_reopen_ms` | 800 | Smart Turn 이 턴 완료로 판단한 뒤 이 시간 동안 응답 음성을 붙잡아 둔다. STT+LLM+TTS 가 이보다 빨리 끝나면 그대로 응답 지연에 더해진다 |

CPU 스모크(가짜 LLM, Supertonic)의 발화 끝→첫 음성은 2.5~2.7초였다. 실제 Gemma·Qwen3-TTS(GPU)로 다시 재야 한다.

## 6. 운영 스크립트 (ada-ops)

- `ada-ops/services/` 에 `ko-gemma-vllm`, `ko-realtime-server`, `ko-web` 의 `.sh`(실행)와 `.ready`(준비 확인) 설치. 원본은 `deploy/ada-ops-services/`.
- `modes.conf`: TTS 는 Realtime 서버 프로세스 안에서 돌므로 `ko-tts` 줄을 지우고 4,000 MiB 를 `ko-realtime-server` 로 옮겼다. `ko-gemma-vllm` 은 `gpu_memory_utilization 0.22 × 49,140 MiB ≈ 10,800 MiB` 에 맞춰 20,000 → 11,000 MiB 로 고쳤다. 백업: `modes.conf.bak.20261113`.
- `gpu-mode realtime --dry-run`: 필요 15,000 MiB, 여유 14,800 MiB 로 **시작을 거부**했다(정상 동작). 현재 GPU 사용: baek `app_flash.py` 30.8 GB, yang 의 Korean_interface `.venv-qwen-tts` 프로세스 2.7 GB(이 프로젝트가 아닌 기존 프로그램).
- **주의: 여유 메모리 검사를 통과해도 안전하다는 뜻이 아니다.** `gpu-mode` 는 실행하는 순간의 여유만 본다. baek 프로세스는 30~43 GB 사이에서 변하므로, 여유가 15 GB 일 때 시작하더라도 이후 그 프로세스가 늘어나면 메모리 부족으로 그 프로세스가 중단될 수 있다. 실제 안전 장치는 **사전에 합의한 GPU 사용 시간대**이며, 합의 없이 `gpu-mode realtime` 을 실행하지 않는다.
- `gpu_layout.md` 의 realtime 표(Gemma E4B 0.22 ≈ 11,000 MiB, Realtime 서버 4,000 MiB)와 후처리 포트(8002)를 `modes.conf` 와 맞췄다. 백업: `gpu_layout.md.bak.20261113`.
- 서비스는 `HF_HUB_OFFLINE=1` 로 실행하고, vLLM 사용 통계 전송을 끈다. 오프라인 모드에서 모든 모델이 로드되는 것을 확인했다.

## 7. 남은 일

- GPU 시간 조율 후: Gemma 4 E4B 실측(3단계), 실제 모델로 E2E·끼어들기·지연 측정(5단계), 브라우저 대화 시험.
- 아동 평가 음성 세트(2단계), Whisper 파인튜닝 선택.
- Slow Path(세션 녹음, `queue/korean/`)는 보호자 동의 절차와 함께 구현한다.
- Smart Turn v3, Silero VAD 라이선스 확인 (`THIRD_PARTY_NOTICES.md`).
