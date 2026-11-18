"""대화 주고받기 지표 (프롬프트 2.5절: 끼어들기 횟수, 사용자 발화 비율). user.wav, agent.wav 의 에너지 발화 구간으로 계산한다.

Moshi 는 전이중이라 Gateway 가 끼어들기 이벤트를 따로 내지 않는다 (docs/event_schema.md). 그래서 두 음성 트랙이 겹치는 모양으로 센다.
- 사용자 끼어들기(barge_in): Agent 가 말하는 도중에 사용자 발화가 시작됨.
  그 Agent 발화가 yield_ms 안에 끝나면 '양보함(yielded)' 으로 센다 → 끼어들기 처리 성공률 (KPI 대화 자연스러움)
- Agent 가 사용자 말을 끊음(agent_cuts_user): 사용자가 말하는 도중에 Agent 발화가 시작되고, 사용자 발화가 그 뒤로 cut_min_ms 이상 이어짐.
  사용자 발화 끝과 거의 같이 시작한 응답(겹침이 짧음)은 끊은 것으로 보지 않는다. 맞장구(backchannel)도 여기에 섞인다.
- 사용자 발화 비율: 사용자 발화 시간 / (사용자 + Agent 발화 시간)

**근사치:** 에너지 기준이므로 에코 제거가 불완전해서 Agent 음성이 user.wav 에 섞이면 끼어들기가 많게 나온다.
user.wav 는 받은 순서대로 이어 쓰고 agent.wav 는 벽시계 기준이라 시간이 조금씩 어긋날 수 있다 (slow_path/run.py response_latency 와 같은 한계).
"""

from __future__ import annotations

Seg = tuple[float, float]


def _speaking(segs: list[Seg], t: float) -> Seg | None:
    return next((s for s in segs if s[0] < t < s[1]), None)


def turn_taking(user: list[Seg], agent: list[Seg], *, yield_ms: int = 1000, cut_min_ms: int = 500) -> dict:
    user_s = sum(b - a for a, b in user)
    agent_s = sum(b - a for a, b in agent)
    barge, yielded = 0, 0
    for u0, _ in user:
        a = _speaking(agent, u0)
        if a is None:
            continue
        barge += 1
        if a[1] - u0 <= yield_ms / 1000:
            yielded += 1
    cuts = 0
    for a0, _ in agent:
        u = _speaking(user, a0)
        if u is not None and u[1] - a0 >= cut_min_ms / 1000:
            cuts += 1
    return {
        "user_turns": len(user),
        "agent_turns": len(agent),
        "user_speech_s": round(user_s, 2),
        "agent_speech_s": round(agent_s, 2),
        "user_speech_ratio": round(user_s / (user_s + agent_s), 3) if user_s + agent_s else None,
        "barge_ins": barge,
        "barge_ins_yielded": yielded,
        "barge_in_yield_rate": round(yielded / barge, 3) if barge else None,
        "agent_cuts_user": cuts,
        "agent_cuts_user_rate": round(cuts / len(user), 3) if user else None,
        "note": "음성 에너지 기준 근사치 (에코가 섞이면 끼어들기가 많게 나옴)",
    }
