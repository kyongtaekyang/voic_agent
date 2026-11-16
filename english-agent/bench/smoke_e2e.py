"""화면 서버 /ws 로 wav 파일을 실시간 속도로 흘려보내고 받은 이벤트를 출력한다 (브라우저 대신 쓰는 점검용 클라이언트).

예: .venv-core/bin/python bench/smoke_e2e.py --url ws://127.0.0.1:7101/ws a.wav b.wav
파일 사이에는 1.5초 무음을 넣는다. 받은 Agent 음성은 길이만 센다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave
from pathlib import Path

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gateway.asr_worker import resample_linear  # noqa: E402

RATE = 24000
CHUNK = 1920  # 80ms


def load(path: str) -> np.ndarray:
    with wave.open(path) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
        return resample_linear(x, w.getframerate(), RATE)


async def main(url: str, files: list[str], tail_s: float) -> None:
    gap = np.zeros(int(1.5 * RATE), np.float32)
    audio = np.concatenate([np.concatenate([load(f), gap]) for f in files] + [np.zeros(int(tail_s * RATE), np.float32)])
    got = 0
    t0 = time.monotonic()
    async with aiohttp.ClientSession() as http, http.ws_connect(url) as ws:
        async def reader():
            nonlocal got
            async for m in ws:
                if m.type == aiohttp.WSMsgType.TEXT:
                    ev = json.loads(m.data)
                    if ev["type"] != "eva.agent_text":
                        print(f"{time.monotonic() - t0:6.2f}s", json.dumps(ev, ensure_ascii=False), flush=True)
                elif m.type == aiohttp.WSMsgType.BINARY:
                    got += len(m.data) // 2
        rt = asyncio.create_task(reader())
        for i in range(0, audio.size, CHUNK):
            if ws.closed:
                break
            await ws.send_bytes((audio[i:i + CHUNK] * 32767).astype("<i2").tobytes())
            await asyncio.sleep(max(0.0, t0 + (i + CHUNK) / RATE - time.monotonic()))
        await ws.close()
        await asyncio.wait_for(rt, 10)
    print(f"sent {audio.size / RATE:.1f}s, received agent audio {got / RATE:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:7101/ws")
    ap.add_argument("--tail", type=float, default=2.0)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    asyncio.run(main(a.url, a.files, a.tail))
