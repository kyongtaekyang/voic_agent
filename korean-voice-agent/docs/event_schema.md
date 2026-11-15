# 화면 ↔ 서버 이벤트 규격 (한국어 Agent)

프롬프트 2.3절에 따라 화면과 서버 사이 이벤트를 정리한다. 나중에 디지털휴먼 UI 를 붙일 때 이 문서를 기준으로 한다.

## 연결 구조

```
브라우저(web/static/app.js) ──WebSocket /ws── 화면 서버(web/server.py, 127.0.0.1:7201)
                                                   │  허용된 이벤트만 전달, kva.* 이벤트 추가
                                                   └──WebSocket── Realtime 서버(speech-to-speech 1.1.0, 127.0.0.1:8766/v1/realtime)
```

- 화면 서버는 Realtime 서버 이벤트를 **그대로** 넘기고, 자체 이벤트를 `kva.` 접두어로 덧붙인다.
- 음성 형식: 입력과 출력 모두 16kHz 모노 PCM16 (little-endian), base64. `session.update` 에 형식을 지정하지 않으면 이 값이다.
- Realtime 서버는 동시에 한 세션만 처리한다(`num_pipelines=1`). 두 번째 연결은 `kva.unavailable`(reason=busy)을 받는다.

## 1. 화면 → 서버 (허용 목록)

화면 서버는 아래 다섯 가지 이벤트만 넘기고 나머지는 버린다 (`web/relay.py: filter_client_event`).

| type | 넘기는 필드 | 비고 |
|---|---|---|
| `session.update` | `session.audio.input.format`, `session.audio.output.format`, `session.extensions` 의 스냅샷 확장 | `instructions`, `tools`, `tool_choice`, `voice` 는 버린다. 턴 판정은 `{"type":"server_vad","interrupt_response":true}` 로 고정 |
| `input_audio_buffer.append` | `audio` (base64, 최대 256KB) | 브라우저는 64ms 단위로 보낸다 |
| `input_audio_buffer.clear` | 없음 | |
| `response.cancel` | 없음 | |
| `conversation.item.truncate` | `item_id`, `content_index`, `audio_end_ms` | 끼어들기 때 실제로 들려준 길이. speech-to-speech 1.1.0 은 이 이벤트로 이력을 고치지 않으므로, 화면 서버가 대신 `conversation.item.delete` + (끝까지 들려준 문장이 있으면) assistant `conversation.item.create` 로 바꿔 보낸다. truncate 자체는 Realtime 서버로 넘기지 않는다 |

막는 이벤트: `conversation.item.create`(텍스트 입력), `response.create`(지시문 변경 가능) 등. 아동 화면에서 시스템 프롬프트를 바꾸거나 음성 인식을 거치지 않은 입력을 넣지 못하게 하기 위해서다.

권장 `session.update`:

```json
{"type": "session.update", "session": {"type": "realtime",
  "extensions": ["speech_to_speech.input_audio_transcription.snapshot"],
  "audio": {"input": {"turn_detection": {"type": "server_vad", "interrupt_response": true}}, "output": {}}}}
```

## 2. 서버 → 화면: Realtime 이벤트 (화면이 쓰는 것)

| type | 화면 동작 |
|---|---|
| `session.created`, `session.updated` | 상태 "듣는 중" |
| `input_audio_buffer.speech_started` | 재생 중이면 즉시 멈춤(끼어들기) 후 `conversation.item.truncate` 전송, 사용자 자막 줄 시작 |
| `speech_to_speech.input_audio_transcription.snapshot` | 사용자 부분 자막을 `transcript` 로 **교체** (추측 결과가 바뀔 수 있음) |
| `conversation.item.input_audio_transcription.delta` | 스냅샷 확장을 쓰지 않을 때만 이어 붙임 |
| `conversation.item.input_audio_transcription.completed` | 사용자 자막 확정 (빈 문자열이면 줄 삭제) |
| `input_audio_buffer.speech_stopped` | 상태 "생각하는 중" |
| `response.created` | 새 답변 시작 |
| `response.output_audio_transcript.delta` | 답변 자막 이어 붙임. 이미 출력 검사·정규화를 거친 실제 읽는 문장이다 |
| `response.output_audio.delta` | 16kHz PCM16 재생 대기열에 추가, 상태 "말하는 중" |
| `response.done` | 재생이 끝나면 "듣는 중" |
| `error` | 개발자 화면 기록 |

## 3. 서버 → 화면: kva.* 이벤트

| type | 필드 | 대상 | 의미 |
|---|---|---|---|
| `kva.status` | `mode`, `maintenance`, `dev_screen`, `english_agent_url`, `models{llm, llm_precision, stt, stt_device, tts, tts_backend}` | 공통 | 연결 직후 한 번. `GET /api/status` 와 같은 내용 |
| `kva.maintenance` | `mode` | 사용자 | 운영 모드가 realtime 이 아님 → "현재 점검 중" 표시 후 연결 종료. 연결 중에도 5초마다 확인 |
| `kva.unavailable` | `reason`: `busy` \| `connect_failed` | 사용자 | 다른 세션이 사용 중이거나 Realtime 서버에 연결하지 못함 |
| `kva.language_mismatch` | `streak` | 사용자 | 한국어가 아닌 발화가 `stt.mismatch_turns`(기본 2) 번 이어짐 → "이 프로그램은 한국어 전용입니다" 안내와 영어 Agent 링크. 세션은 넘기지 않는다 |
| `kva.vad` | `event`=`speech_started`, `detect_delay_ms` | 개발자 | 실제 발화 시작부터 서버 감지까지 (= 보낸 음성 길이 − `audio_start_ms` − `turn.speech_pad_ms`). 하한은 `turn.min_speech_ms` |
| `kva.metrics` | `turn`, `status`, `turn_decision_ms`, `e2e_ms` | 개발자 | 화면 기준. `turn_decision_ms` = 발화 끝 → `speech_stopped` 도착, `e2e_ms` = 발화 끝 → 첫 응답 음성 도착 |
| `kva.server_latency` | `turn_id`, `revision`, `vad_decision_ms`, `hold_ms`, `stt_ms`, `llm_ms`, `tts_ttfa_ms`, `e2e_ms`, `smart_turn_status`, `status`, `source` | 개발자 | speech-to-speech 서버가 잰 단계별 지연 (`realtime_server/latency_log.py`). 값이 없으면 null |
| `kva.history_truncated` | `item_id`, `audio_end_ms`, `kept` | 개발자 | 끼어들기로 이력을 고쳤다. `kept=false` 면 끝까지 들려준 문장이 없어 assistant 항목을 지웠다 |
| `kva.guard` | `event`: `guard_blocked` \| `history_sanitized`, `rule`, `matched`, `ts` | 개발자 | 출력 검사 차단과 대화 이력 정리 기록. 차단된 문장 전체는 보내지 않는다 |

단계별 지연을 화면 이벤트 시각으로 재지 않는 이유: speech-to-speech 는 턴이 확정되기 전에 STT 와 LLM 을 미리 실행하고(speculative turn), `speech_stopped` 를 그 뒤에 보낸다. 그래서 화면 쪽 시각 차이는 실제 단계 시간을 나타내지 않는다.

## 4. 개인정보 처리

- `kva.*` 이벤트에는 사용자 발화 원문을 넣지 않는다. `language_mismatch` 의 원문은 `features.log_transcripts: true` 일 때만 서버 `logs/events.jsonl` 에 남는다.
- `log_transcripts: false`(기본)이면 speech-to-speech 의 `USER:`/`ASSISTANT:` 콘솔 출력도 끈다.
- 사용자 자막은 브라우저 화면에만 표시되고 서버에 따로 저장되지 않는다 (2.2절 Slow Path 녹음은 별도 단계에서 동의 절차와 함께 구현).
