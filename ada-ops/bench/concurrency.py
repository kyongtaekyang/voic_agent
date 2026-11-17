"""7단계 동시 실행 측정: 한국어·영어 Agent 화면(/ws)에 가짜 사용자 세션 N개씩을 동시에 붙여 지연, 오류율, GPU 메모리를 잰다.

브라우저 대신 서버 안의 WAV 를 실시간 속도로 보낸다. 음성과 인식 원문은 저장하지 않고 수치만 남긴다.
실행 (english-agent 의 .venv-core 사용, aiohttp·numpy 필요):
  english-agent/.venv-core/bin/python ada-ops/bench/concurrency.py --levels 1,3,5 --agents ko,en
결과: ada-ops/bench/out/concurrency_<UTC>.json 과 표 출력.

지연 정의 (프롬프트 KPI): 사용자 발화 끝(WAV 마지막 샘플을 보낸 시각)부터 첫 응답 음성을 받은 시각까지.
- 한국어: 발화 끝 뒤 첫 response.output_audio.delta. 서버 VAD 의 무음 판정 시간이 포함된다.
- 영어(full-duplex): 발화 끝 뒤 처음으로 RMS 가 --en-rms 를 넘은 응답 음성 묶음. Moshi 는 사용자가 말하는 중에도 말할 수 있으므로
  발화 중에 이미 말하고 있었으면 overlap 으로 따로 센다.
세션 결과: ok(모든 턴 응답) / partial / busy(서버가 받지 않음: Moshi·speech-to-speech 는 한 번에 한 연결) / error / timeout.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import numpy as np

OPS = Path(__file__).resolve().parent.parent
ROOT = OPS.parent
DEFAULT_KO_WAV = ROOT / "korean_agent/bench/data/ko_ref/clip06_1e8d231b.wav"
DEFAULT_EN_WAV = ROOT / "english-agent/bench/out/lang_probe_en/00_en.wav"


def load_wav(path: Path, rate: int) -> np.ndarray:
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
        sr = w.getframerate()
    if sr != rate:
        n = int(round(x.size * rate / sr))
        x = np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)
    return x


def pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


class Session:
    def __init__(self, agent: str, idx: int):
        self.agent, self.idx = agent, idx
        self.status = "timeout"
        self.latencies_ms: list[float] = []
        self.overlap_turns = 0
        self.errors: list[str] = []
        self.turns_done = 0
        self.connect_ms: float | None = None

    def result(self) -> dict:
        return {"agent": self.agent, "idx": self.idx, "status": self.status, "turns_done": self.turns_done,
                "latencies_ms": [round(v) for v in self.latencies_ms], "overlap_turns": self.overlap_turns,
                "errors": self.errors[:5], "connect_ms": self.connect_ms}


async def paced_send(chunks, rate: int, send) -> None:
    """chunks 를 실시간 속도로 보낸다. 늦어지면 따라잡되 앞서 가지 않는다."""
    t0, sent = time.monotonic(), 0
    for c in chunks:
        await send(c)
        sent += c.size
        await asyncio.sleep(max(0.0, t0 + sent / rate - time.monotonic()))


async def run_ko(s: Session, http: aiohttp.ClientSession, a) -> None:
    rate, chunk = 16000, 512
    speech = load_wav(a.ko_wav, rate)
    silence = np.zeros(int(a.ko_silence * rate), np.float32)
    t_conn = time.monotonic()
    async with http.ws_connect(a.ko_url, max_msg_size=0, timeout=a.connect_timeout) as ws:
        s.connect_ms = round((time.monotonic() - t_conn) * 1000)
        await ws.send_str(json.dumps({"type": "session.update", "session": {
            "type": "realtime", "audio": {"input": {"turn_detection": {"type": "server_vad", "interrupt_response": True}},
                                          "output": {}}}}))
        state = {"speech_end": None, "first_audio": None, "done": asyncio.Event(), "started": False}

        async def reader():
            async for m in ws:
                if m.type != aiohttp.WSMsgType.TEXT:
                    if m.type == aiohttp.WSMsgType.ERROR:
                        s.errors.append("ws_error")
                    continue
                ev = json.loads(m.data)
                kind = ev.get("type", "")
                if kind == "session.created":
                    state["started"] = True
                elif kind in ("kva.unavailable", "kva.maintenance"):
                    s.status = "busy" if ev.get("reason") == "busy" else kind.split(".")[1]
                    state["done"].set()
                    return
                elif kind == "error":
                    s.errors.append(json.dumps(ev.get("error", ev), ensure_ascii=False)[:160])
                elif kind == "response.output_audio.delta":
                    if state["speech_end"] is not None and state["first_audio"] is None:
                        state["first_audio"] = time.monotonic()
                elif kind == "response.done" and state["first_audio"] is not None:
                    state["done"].set()
            state["done"].set()

        rt = asyncio.create_task(reader())
        try:
            for _ in range(a.turns):
                state.update(speech_end=None, first_audio=None)
                state["done"].clear()

                async def send(c):
                    await ws.send_str(json.dumps({"type": "input_audio_buffer.append",
                                                  "audio": base64.b64encode(pcm16(c)).decode()}))

                await paced_send([speech[i:i + chunk] for i in range(0, speech.size, chunk)], rate, send)
                state["speech_end"] = time.monotonic()
                silence_task = asyncio.create_task(
                    paced_send([silence[i:i + chunk] for i in range(0, silence.size, chunk)], rate, send))
                try:
                    await asyncio.wait_for(state["done"].wait(), a.turn_timeout)
                except asyncio.TimeoutError:
                    s.errors.append("turn_timeout")
                finally:
                    silence_task.cancel()
                    await asyncio.gather(silence_task, return_exceptions=True)
                if s.status in ("busy", "maintenance", "unavailable") or ws.closed:
                    break
                if state["first_audio"] is None:
                    continue
                s.latencies_ms.append((state["first_audio"] - state["speech_end"]) * 1000)
                s.turns_done += 1
                await asyncio.sleep(a.pause)
        finally:
            rt.cancel()
            await asyncio.gather(rt, return_exceptions=True)


async def run_en(s: Session, http: aiohttp.ClientSession, a) -> None:
    rate, chunk = 24000, 1920
    speech = load_wav(a.en_wav, rate)
    listen = np.zeros(int(a.en_listen * rate), np.float32)
    t_conn = time.monotonic()
    async with http.ws_connect(a.en_url, max_msg_size=0, timeout=a.connect_timeout) as ws:
        s.connect_ms = round((time.monotonic() - t_conn) * 1000)
        state = {"speaking": False, "speech_end": None, "first_audio": None, "ended": False}

        async def reader():
            async for m in ws:
                if m.type == aiohttp.WSMsgType.BINARY:
                    x = np.frombuffer(m.data, "<i2").astype(np.float32) / 32768
                    loud = x.size > 0 and float(np.sqrt(np.mean(x * x))) > a.en_rms
                    if loud and state["speech_end"] is None:
                        state["speaking"] = True
                    if loud and state["speech_end"] is not None and state["first_audio"] is None:
                        state["first_audio"] = time.monotonic()
                elif m.type == aiohttp.WSMsgType.TEXT:
                    ev = json.loads(m.data)
                    kind = ev.get("type", "")
                    if kind == "eva.unavailable":
                        # busy: Gateway 가 이미 세션을 처리 중, moshi_busy: Moshi 서버가 연결을 받지 않음
                        s.status = "busy" if ev.get("reason") in ("busy", "moshi_busy") else "unavailable"
                        s.errors.append(ev.get("reason", ""))
                        state["ended"] = True
                        return
                    if kind == "eva.error":
                        s.errors.append(str(ev.get("detail", ""))[:160])
                    elif kind == "eva.session.ended":
                        state["ended"] = True
            state["ended"] = True

        rt = asyncio.create_task(reader())

        async def send(c):
            if not ws.closed:
                await ws.send_bytes(pcm16(c))

        try:
            # 연결 직후 eva.unavailable(busy) 가 오는지 잠깐 본다
            await asyncio.sleep(0.5)
            for _ in range(a.turns):
                if state["ended"] or ws.closed:
                    break
                state.update(speaking=False, speech_end=None, first_audio=None)
                await paced_send([speech[i:i + chunk] for i in range(0, speech.size, chunk)], rate, send)
                state["speech_end"] = time.monotonic()
                overlap = state["speaking"]
                await paced_send([listen[i:i + chunk] for i in range(0, listen.size, chunk)], rate, send)
                if state["ended"]:
                    break
                if overlap:
                    # 발화 중에 이미 말하고 있었으면 '발화 끝 → 첫 응답' 지연이 정의되지 않으므로 지연에 넣지 않는다
                    s.overlap_turns += 1
                    s.turns_done += 1
                elif state["first_audio"] is not None:
                    s.latencies_ms.append((state["first_audio"] - state["speech_end"]) * 1000)
                    s.turns_done += 1
                else:
                    s.errors.append("no_response")
        finally:
            rt.cancel()
            await asyncio.gather(rt, return_exceptions=True)


async def one(agent: str, idx: int, http, a) -> dict:
    s = Session(agent, idx)
    try:
        await asyncio.wait_for((run_ko if agent == "ko" else run_en)(s, http, a), a.session_timeout)
    except asyncio.TimeoutError:
        s.errors.append("session_timeout")
    except (aiohttp.ClientError, OSError) as e:
        s.status = "error"
        s.errors.append(f"{type(e).__name__}: {e}"[:160])
    if s.status == "timeout" and "session_timeout" not in s.errors:
        s.status = "ok" if s.turns_done == a.turns else ("partial" if s.turns_done else "error")
    return s.result()


class GpuSampler:
    """gpu-mode mem (total used free ours) 을 주기적으로 읽는다. nvidia-smi 가 없으면 비워 둔다."""

    def __init__(self, interval: float):
        self.interval, self.rows, self._stop = interval, [], asyncio.Event()

    @staticmethod
    def read() -> list[int] | None:
        try:
            out = subprocess.run([str(OPS / "gpu-mode"), "mem"], capture_output=True, text=True, timeout=10).stdout
            return [int(v) for v in out.split()[:4]]
        except Exception:  # noqa: BLE001
            return None

    async def run(self):
        while not self._stop.is_set():
            r = await asyncio.to_thread(self.read)
            if r:
                self.rows.append(r)
            try:
                await asyncio.wait_for(self._stop.wait(), self.interval)
            except asyncio.TimeoutError:
                pass

    def summary(self, baseline) -> dict:
        if not self.rows:
            return {}
        used = [r[1] for r in self.rows]
        return {"baseline_used_mib": baseline[1] if baseline else None, "peak_used_mib": max(used),
                "peak_ours_mib": max(r[3] for r in self.rows), "min_free_mib": min(r[2] for r in self.rows),
                "samples": len(self.rows)}


def pct(v: list[float], q: float) -> float | None:
    if not v:
        return None
    v = sorted(v)
    return round(v[min(len(v) - 1, int(round(q * (len(v) - 1))))])


def summarize(results: list[dict], agent: str) -> dict:
    rs = [r for r in results if r["agent"] == agent]
    lat = [x for r in rs for x in r["latencies_ms"]]
    count = {k: sum(r["status"] == k for r in rs) for k in ("ok", "partial", "busy", "error", "timeout", "maintenance", "unavailable")}
    return {"sessions": len(rs), **{k: v for k, v in count.items() if v},
            "served": sum(r["turns_done"] > 0 for r in rs),
            # error_rate: 끝까지 응답받지 못한 세션 비율 (busy 포함). busy_rate 로 수용 한계 때문인 몫을 따로 본다
            "error_rate": round(sum(r["status"] != "ok" for r in rs) / len(rs), 3) if rs else None,
            "busy_rate": round(sum(r["status"] == "busy" for r in rs) / len(rs), 3) if rs else None,
            "turns": len(lat), "p50_ms": pct(lat, 0.5), "p95_ms": pct(lat, 0.95),
            "mean_ms": round(statistics.mean(lat)) if lat else None,
            "overlap_turns": sum(r["overlap_turns"] for r in rs)}


async def level(n: int, a) -> dict:
    sampler = GpuSampler(a.gpu_interval)
    baseline = await asyncio.to_thread(GpuSampler.read)
    st = asyncio.create_task(sampler.run())
    t0 = time.monotonic()
    async with aiohttp.ClientSession(trust_env=False) as http:
        jobs = []
        for agent in a.agents:
            for i in range(n):
                async def delayed(agent=agent, i=i):
                    await asyncio.sleep(i * a.stagger)
                    return await one(agent, i, http, a)
                jobs.append(delayed())
        results = await asyncio.gather(*jobs)
    sampler._stop.set()
    await st
    return {"level": n, "wall_s": round(time.monotonic() - t0, 1), "gpu": sampler.summary(baseline),
            "summary": {ag: summarize(results, ag) for ag in a.agents}, "sessions": results}


def mode_now() -> str | None:
    p = OPS / "run" / "mode"
    return p.read_text().strip() if p.exists() else None


async def main(a) -> dict:
    report = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": mode_now(),
              "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, "levels": []}
    for n in a.levels:
        r = await level(n, a)
        report["levels"].append(r)
        print_level(r, a.agents)
        await asyncio.sleep(a.cooldown)
    return report


def print_level(r: dict, agents) -> None:
    g = r["gpu"]
    print(f"\n== {r['level']} session(s) per agent, {r['wall_s']}s, GPU peak {g.get('peak_used_mib')} MiB "
          f"(baseline {g.get('baseline_used_mib')}, ours peak {g.get('peak_ours_mib')}, min free {g.get('min_free_mib')})")
    print("agent  sessions  served  ok  busy  err  error_rate  latency_turns  p50_ms  p95_ms  overlap")
    for ag in agents:
        s = r["summary"][ag]
        err = sum(s.get(k, 0) for k in ("error", "timeout", "partial", "maintenance", "unavailable"))
        print(f"{ag:5}  {s['sessions']:8}  {s['served']:6}  {s.get('ok', 0):2}  {s.get('busy', 0):4}  "
              f"{err:3}  {s['error_rate']!s:10}  {s['turns']:13}  {s['p50_ms']!s:6}  {s['p95_ms']!s:6}  {s['overlap_turns']}")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default="1,3,5", type=lambda v: [int(x) for x in v.split(",")])
    ap.add_argument("--agents", default="ko,en", type=lambda v: [x for x in v.split(",") if x])
    ap.add_argument("--ko-url", default="ws://127.0.0.1:7201/ws")
    ap.add_argument("--en-url", default="ws://127.0.0.1:7101/ws")
    ap.add_argument("--ko-wav", type=Path, default=DEFAULT_KO_WAV)
    ap.add_argument("--en-wav", type=Path, default=DEFAULT_EN_WAV)
    ap.add_argument("--turns", type=int, default=3, help="세션마다 보낼 발화 수")
    ap.add_argument("--ko-silence", type=float, default=3.0, help="한국어 발화 뒤 보내는 무음(초). 서버 VAD 가 턴 끝을 잡도록")
    ap.add_argument("--en-listen", type=float, default=4.0, help="영어 발화 뒤 무음을 보내며 응답을 기다리는 시간(초)")
    ap.add_argument("--en-rms", type=float, default=0.01, help="영어 응답 음성으로 볼 최소 RMS")
    ap.add_argument("--pause", type=float, default=1.0, help="한국어 턴 사이 쉬는 시간(초)")
    ap.add_argument("--stagger", type=float, default=0.3, help="세션 시작 간격(초)")
    ap.add_argument("--turn-timeout", type=float, default=30.0)
    ap.add_argument("--session-timeout", type=float, default=300.0)
    ap.add_argument("--connect-timeout", type=float, default=10.0)
    ap.add_argument("--gpu-interval", type=float, default=1.0)
    ap.add_argument("--cooldown", type=float, default=5.0, help="수준 사이 쉬는 시간(초)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    for name in ("ko", "en"):
        if name in a.agents and not getattr(a, f"{name}_wav").exists():
            ap.error(f"--{name}-wav not found: {getattr(a, f'{name}_wav')}")
    return a


if __name__ == "__main__":
    args = parse()
    rep = asyncio.run(main(args))
    out = args.out or OPS / "bench" / "out" / f"concurrency_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved {out}")
    sys.exit(0)
