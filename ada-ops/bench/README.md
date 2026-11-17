# 7단계 동시 실행 측정 (ada-ops/bench)

영어·한국어 프롬프트 7단계("두 Agent 동시 구동 시 메모리·지연 보고")와 KPI 동시성 항목(동시 1·3·5세션의 지연, GPU 메모리, 오류율)을 재는 도구다.
2026-11-17 에 CPU 와 가짜 모델로만 미리 만들었다. **GPU 에서 실제 모델로 측정한 값은 아직 없다.**

## 파일

| 파일 | 내용 |
|---|---|
| `concurrency.py` | 두 Agent 화면(/ws, 7201·7101)에 가짜 사용자 세션을 N개씩 동시에 붙인다. 서버 안의 WAV 를 실시간 속도로 보내고 지연·오류·GPU 메모리를 기록한다 |
| `cpu_concurrency_smoke.sh` | GPU 없이 도구를 시험한다. 가짜 LLM·가짜 Moshi 로 두 Agent 를 띄우고 세션 1·3개를 붙인다 |
| `out/` | 결과 JSON (저장소에 올리지 않음). 음성과 인식 원문은 저장하지 않고 수치만 남긴다 |

## 실제 측정 순서 (GPU 시간대가 정해진 뒤)

```bash
ada-ops/gpu-mode realtime          # 두 Agent 를 함께 올리는 모드 (지금은 Korean 만 들어 있다. 아래 '남은 일' 참고)
english-agent/.venv-core/bin/python ada-ops/bench/concurrency.py --levels 1,3,5 --agents ko,en --turns 3
```

- 지연: 사용자 발화 끝(WAV 마지막 샘플을 보낸 시각)부터 첫 응답 음성까지. 한국어는 서버 VAD 의 무음 판정 시간이 포함된다.
- 영어는 full-duplex 라서 사용자가 말하는 중에 Agent 가 이미 말하고 있었으면 지연을 정의할 수 없다. 이런 턴은 `overlap` 으로 따로 세고 지연에는 넣지 않는다.
- 세션 결과: `ok`, `partial`, `busy`(서버가 세션을 받지 않음), `error`, `timeout`, `unavailable`. `error_rate` 는 끝까지 응답받지 못한 세션의 비율(busy 포함)이고, `busy_rate` 로 수용 한계 때문에 생긴 몫을 따로 본다.
- GPU 메모리: 1초마다 `gpu-mode mem` 을 읽어 측정 전 사용량, 최고 사용량, 우리 프로세스 최고 사용량, 최소 여유를 남긴다.

## CPU 시험 결과 (2026-11-17, 가짜 모델, KPI 아님)

| 세션 수(Agent 별) | Agent | 응답받은 세션 | busy | 지연 p50 / p95 |
|---|---|---|---|---|
| 1 | ko | 1 | 0 | 1402 / 1861 ms (CPU Supertonic TTS, 가짜 LLM) |
| 1 | en | 1 | 0 | 측정 안 됨 (가짜 Moshi 가 쉬지 않고 말해서 2턴 모두 overlap) |
| 3 | ko | 1 | 2 | 1097 / 1422 ms |
| 3 | en | 1 | 2 | 측정 안 됨 (overlap) |

수치 자체는 의미가 없다. 확인한 것은 도구가 끝까지 돌고, 거절된 세션을 오류와 구분해서 센다는 점이다.

## 확인된 구조 제약: 두 Agent 모두 한 번에 한 세션만 받는다

- 한국어: speech-to-speech 1.1.0 Realtime 서버의 세션 슬롯이 1개다. 두 번째 연결은 `session_limit_reached`("All 1 session slots are in use") 로 거절된다.
- 영어: Moshi 서버가 한 번에 한 연결만 받고, Gateway 도 세션 하나만 처리한다(`eva.unavailable` reason `busy`).
- 따라서 지금 구조로 동시 3·5세션을 재면 세션 1개만 응답받고 나머지는 busy 가 된다. 수용 가능한 최대 세션 수는 현재 1이다.
- 여러 세션을 받으려면 서버 인스턴스를 늘리거나(GPU 메모리가 인스턴스 수만큼 늘어남), 다중 세션을 지원하는 서빙 방식을 검토해야 한다. 어느 쪽을 택할지는 결정이 필요하다.

## 남은 일

- `gpu-mode realtime` 에는 지금 Korean 서비스만 들어 있다. 프롬프트의 실시간 모드는 영어 Moshi Rust INT8 을 함께 올리지만 Rust 빌드 도구(cargo)가 서버에 없어서 아직 넣지 못했다. 두 Agent 동시 측정 전에 영어 서비스를 이 모드에 추가해야 한다.
- 영어 지연은 실제 Moshi 로 재야 의미가 있다. 실제 Moshi 도 사용자가 말하는 중에 맞장구를 칠 수 있으므로 overlap 비율도 함께 보고한다.
