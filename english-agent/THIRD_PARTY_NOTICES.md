# 제3자 고지 (영어 Agent)

프롬프트 7절에 따라 사용하는 모델과 코드의 라이선스를 정리합니다. 배포 전에 법무 검토가 필요합니다.

| 구성 요소 | 용도 | 라이선스 | 확인 위치 |
|---|---|---|---|
| Moshi 가중치 (`kyutai/moshiko-pytorch-bf16`, `moshika-*`, `moshiko-candle-q8`) | 대화 엔진 | CC BY 4.0 (출처 표시 필요) | Hugging Face 모델 카드 |
| Moshi Python 코드 (`moshi` 0.2.13) | 서버, 클라이언트 | MIT | `.src/moshi/LICENSE-MIT` |
| Moshi Rust 코드 (`moshi-backend`, `moshi-core`) | 운영 백엔드 (2단계) | Apache 2.0 | `.src/moshi/LICENSE-APACHE` |
| Nemotron 3.5 ASR Streaming 0.6B, Transcribe 30B | 병행 감시, Slow Path | OpenMDW-1.1 | `docs/Nemotron_3.5_Transcribe.md`, 모델 카드 |
| Qwen Audio Agent | 연동 검토 (1차 PoC 미사용) | Apache 2.0 | `docs/Qwen_Audio_Agent.md` |
| Supertonic-3 | 측정용 합성 음성 생성에만 사용 (서비스에 포함하지 않음) | OpenRAIL-M | 한국어 Agent `THIRD_PARTY_NOTICES.md` |

CC BY 4.0 출처 표시 예: "This product uses Moshi by Kyutai (https://github.com/kyutai-labs/moshi), licensed under CC BY 4.0."
