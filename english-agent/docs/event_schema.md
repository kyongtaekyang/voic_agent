# 화면 ↔ Core 이벤트 규격 (영어 Agent, 초안)

프롬프트 2.6절에 따라 화면과 Core 사이의 이벤트를 정리합니다. 5단계(화면·Gateway 구현)에서 완성하며, 지금은 사전 구축한 모듈이 내는 이벤트만 적었습니다. 자체 이벤트는 한국어 Agent의 `kva.` 접두어와 구분하기 위해 `eva.` 접두어를 씁니다.

## 연결 구조 (예정)

```
브라우저 ──WebSocket── 화면 서버(127.0.0.1:7101) ──── Core/Gateway(127.0.0.1:7102)
                                                        ├─ Moshi 서버 ws://127.0.0.1:8998/api/chat (바이너리, docs/version_check.md 3절)
                                                        ├─ Safety Monitor (core/safety_monitor.py)
                                                        └─ ASR 병행 감시 (core/asr_monitor.py, CPU)
```

## Core → 화면 이벤트

| type | 내는 곳 | 필드 | 화면 동작 |
|---|---|---|---|
| `eva.language_mismatch` | `core/asr_monitor.py: LanguageMismatchDetector.feed` | `detected_language`(예: `ko`), `turns`(연속 비영어 발화 수), `message`, `korean_agent_url` | "This program is English only." 안내와 한국어 Agent 링크 표시. 세션은 넘기지 않음 |
| `eva.safety_block` (예정) | Gateway (`SafetyMonitor.feed`/`idle` 이 위반을 돌려줄 때) | `rule`, `category`, `token_index`, `detect_ms`(텍스트 도착부터 판정까지) | Agent 음성 재생 즉시 중단, 안내 음성 재생, 개발자 화면에 차단 로그 표시. 사용자 화면에는 위반 문구를 보이지 않음 |

## Gateway가 지켜야 할 것 (Safety Monitor 관련)

- Moshi 텍스트 토큰(0x02)마다 `SafetyMonitor.feed()`를 호출합니다.
- 텍스트 토큰 없이 오디오 프레임이 2개(약 160ms) 이상 이어지면 `SafetyMonitor.idle()`을 호출합니다. Moshi는 전이중이라 턴 끝 신호가 없으므로, 이 호출이 없으면 발화 마지막 단어의 위반을 확정하지 못합니다.
- 세션이 끝나면 `reset()`을 호출합니다.
