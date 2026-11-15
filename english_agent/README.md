# 영어 음성 Agent (english_agent)

영어 전용 실시간 음성 튜터. 한국어 Agent(`../korean_agent/`)와는 별개의 프로그램·화면이며, Ada 서버 GPU 한 장을 함께 쓴다 (`../ada-ops/`).
명세: `../english_voice_agent_prompt.md`, 참고: `../Moshi_오픈소스_실시간_음성AI_사용설명서.md`.

- 대화 엔진: Kyutai Moshi (Mimi 코덱, 24 kHz, Full-Duplex)
- 화면/Core: 127.0.0.1:7101 / 7102, Moshi 서버: 127.0.0.1:8998
- 음성과 대화 기록은 서버 밖으로 보내지 않는다.
- Safety Monitor 검증이 끝나기 전에는 아동 사용자에게 공개하지 않는다 (1차 대상: 성인 학습자, 데모).

상태: 0단계 준비 중. GPU 모델 로딩은 GPU 사용 시간 합의 후 진행한다.
