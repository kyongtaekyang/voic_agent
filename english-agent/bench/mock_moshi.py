"""GPU 없이 Gateway·화면을 시험하기 위한 가짜 Moshi 서버. 실제 Moshi 와 같은 바이너리 규격을 쓴다 (docs/version_check.md 3절).

- 연결되면 0x00 을 보낸다. 0x01(Ogg/Opus 24kHz) 입력을 풀어서 80ms(1920 샘플)마다 한 프레임씩 응답한다.
- 응답 프레임마다 대본의 토큰 하나를 0x02 로 보내고, 같은 프레임의 음성(사인파)을 0x01 로 보낸다 (PyTorch 서버처럼 음성 먼저).
  대본 줄 사이에는 텍스트 없이 무음 프레임을 gap 개 보낸다. 대본은 끝나면 처음부터 반복한다.
- 실제 Moshi 처럼 한 번에 한 연결만 받는다. 두 번째 연결은 handshake 를 받지 못하고 기다린다.
실행 (화면 시연용): .venv/bin/python bench/mock_moshi.py --port 8999   그리고 config 의 moshi.url 을 8999 로 바꾼다.
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
import sphn
from aiohttp import WSMsgType, web

RATE = 24000
FRAME = 1920
DEFAULT_SCRIPT = [
    ["Hello", " there", "!", " How", " are", " you", " today", "?"],
    [" Let's", " practice", " ordering", " food", "."],
    [" What", " would", " you", " like", " to", " drink", "?"],
]


def make_app(script=None, gap: int = 6) -> web.Application:
    script = script or DEFAULT_SCRIPT
    lock = asyncio.Lock()
    frames = []
    for line in script:
        frames += [(tok, True) for tok in line] + [(None, False)] * gap

    async def chat(request):
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        async with lock:
            reader, writer = sphn.OpusStreamReader(RATE), sphn.OpusStreamWriter(RATE)
            await ws.send_bytes(b"\x00")
            buf = np.zeros(0, np.float32)
            i = 0
            t = np.arange(FRAME) / RATE
            async for m in ws:
                if m.type != WSMsgType.BINARY or not m.data or m.data[0] != 1:
                    if m.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                        break
                    continue
                buf = np.concatenate([buf, reader.append_bytes(m.data[1:])])
                while buf.size >= FRAME:
                    buf = buf[FRAME:]
                    tok, voiced = frames[i % len(frames)]
                    i += 1
                    pcm = (0.2 * np.sin(2 * np.pi * 220 * (t + i * FRAME / RATE))).astype(np.float32) if voiced \
                        else np.zeros(FRAME, np.float32)
                    opus = writer.append_pcm(pcm)
                    if opus:
                        await ws.send_bytes(b"\x01" + opus)
                    if tok is not None:
                        await ws.send_bytes(b"\x02" + tok.encode("utf-8"))
        return ws

    app = web.Application()
    app.router.add_get("/api/chat", chat)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8999)
    a = ap.parse_args()
    web.run_app(make_app(), host="127.0.0.1", port=a.port)
