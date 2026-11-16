# 화면 ↔ Core 이벤트 규격 (영어 Agent)

프롬프트 2.6절에 따라 화면과 Core 사이 이벤트를 정리합니다. 나중에 디지털휴먼 UI를 붙일 때 이 문서를 기준으로 합니다. 자체 이벤트는 한국어 Agent의 `kva.`와 구분하기 위해 `eva.` 접두어를 씁니다.

## 연결 구조

```
브라우저(web/static/app.js) ──WebSocket /ws── 화면 서버(web/server.py, 127.0.0.1:7101)
                                                │  메시지를 그대로 이어 줌
                                                └──WebSocket /session── Core(gateway/server.py, 127.0.0.1:7102)
                                                                          ├─ Moshi 서버 ws://127.0.0.1:8998/api/chat (바이너리, Opus)
                                                                          ├─ Safety Monitor (core/safety_monitor.py)
                                                                          ├─ ASR 병행 감시 (gateway/asr_worker.py, CPU)
                                                                          └─ 녹음 queue/english/<세션>/ (gateway/recorder.py)
```

- **음성 형식:** 브라우저와 Core 사이는 양방향 모두 **바이너리 WebSocket 메시지 = PCM16 little-endian, 24kHz 모노**입니다. 브라우저는 80ms(1,920 샘플) 단위로 보냅니다. Opus 변환은 Core가 합니다.
- **텍스트 메시지 = JSON 이벤트**입니다. 아래 표의 `type`을 가집니다.
- Moshi는 전이중입니다. 브라우저는 Agent가 말하는 동안에도 마이크를 계속 보냅니다(끼어들기는 Moshi가 스스로 처리). 에코 제거는 브라우저의 `echoCancellation`에 맡깁니다.
- Core는 한 번에 한 세션만 받습니다(Moshi PyTorch 서버의 제약, `docs/version_check.md` 2절).

## 1. 화면 → Core

| 메시지 | 내용 |
|---|---|
| 바이너리 | 마이크 PCM16 24kHz |
| 텍스트 | 현재는 없습니다(받아도 무시). 대화 종료는 WebSocket을 닫는 것으로 합니다 |

## 2. Core → 화면

| type | 필드 | 화면 동작 |
|---|---|---|
| `eva.session.started` | `sample_rate` | 상태를 "듣는 중"으로 |
| `eva.state` | `state`: `listening` \| `speaking` | 상태 표시. Agent 출력 음성 크기로 판단(최근 300ms 안에 RMS 0.01 이상이면 speaking) |
| `eva.agent_text` | `delta` | Agent 자막에 이어 붙임. **안전 판정이 끝난 토큰만 오므로 음성보다 토큰 하나(약 80ms) 늦습니다** |
| `eva.user_text` | `text`, `final`(현재 항상 true), `lang`(`en`, `ko`, ... 또는 null), `audio_s`, `asr_ms` | 사용자 자막. 발화가 끝난 뒤(0.5초 무음) 확정 자막만 옵니다 |
| `eva.language_mismatch` | `detected_language`, `turns`, `message`, `korean_agent_url` | "This program is English only." 안내와 한국어 Agent 링크. 세션은 넘기지 않음 |
| `eva.safety_block` | `rule`, `category`, `token_index`, `dropped_ms`(버린 음성), `already_played_ms`(이 발화에서 이미 내보낸 음성), `redirect_text` | **브라우저에 쌓인 재생 대기 음성도 즉시 버림**, 안내 문구 표시. 이어서 안내 음성(있으면)이 바이너리로 옴 |
| `eva.safety_resumed` | 없음 | 차단 뒤 Moshi가 `unmute_quiet_ms` 동안 조용해져서 출력 재개 |
| `eva.unavailable` | `reason`: `maintenance` \| `busy` \| `moshi_unreachable` \| `moshi_busy` \| `moshi_bad_handshake` \| `core_unreachable`, `detail` | 안내 표시 후 세션 종료 |
| `eva.error` | `source`(`moshi`, `asr`), `detail` | 개발자 화면에만 표시 |
| `eva.session.ended` | `duration_s`, `user_s`, `agent_s`, `text_tokens`, `safety_blocks` | 녹음 `events.jsonl`에는 항상 남음. 브라우저가 먼저 닫으면 받지 못함 |

## 3. HTTP

| 경로 | 서버 | 내용 |
|---|---|---|
| `GET /api/status` | 화면 → Core | `mode`(ada-ops 운영 모드), `maintenance`, `busy`, `output_delay_ms`, `asr`, `redirect_audio`, `last_session`(직전 세션의 차단 지연·이미 나간 음성 길이), 화면 서버가 덧붙이는 `developer_panel`, `korean_agent_url` |
| `GET /api/dev/gpu` | 화면 → Core | GPU 메모리. 설정 `web.developer_panel: true`일 때만 응답(아니면 403) |

**점검 중 판단:** `ada-ops/run/mode`가 `realtime` 또는 `en-solo`가 아니면 `maintenance: true`입니다(프롬프트 8.3절). 설정 `gateway.require_mode: false`로 끌 수 있으며, 가짜 Moshi로 시험할 때만 씁니다.

## 4. Gateway 안전 처리 순서

1. Moshi 텍스트 토큰(0x02)마다 `SafetyMonitor.feed()`를 호출합니다.
2. 위반이 없으면 **직전 토큰**을 자막으로 내보내고, 이번 토큰은 붙잡아 둡니다. `word` 규칙은 다음 토큰에서 확정되기 때문입니다.
3. 텍스트 토큰 없이 `idle_ms`(기본 160ms)가 지나면 `idle()`을 호출합니다. 위반이 없으면 붙잡아 둔 토큰을 내보냅니다.
4. 위반이면 붙잡아 둔 토큰과 출력 지연 버퍼(`output_delay_ms`)의 음성을 버리고, `eva.safety_block`과 안내 음성을 보냅니다. Moshi가 `unmute_quiet_ms` 동안 조용해질 때까지 음성과 자막을 내보내지 않습니다.

**한계:** `output_delay_ms: 0`(현재 기본값)이면 위반 단어의 소리는 텍스트와 거의 같이 나가므로 막지 못합니다. 막을 수 있는 것은 "그 뒤의 소리"와 자막입니다. 2단계 실측(텍스트와 음성의 시간 차이)을 보고 지연 버퍼 길이를 정합니다.
