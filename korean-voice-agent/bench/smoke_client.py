"""Realtime 서버(/v1/realtime)에 서버 안의 WAV 한 개를 실시간 속도로 보내고 이벤트를 모은다.

출력 음성은 서버의 logs/ 아래에만 저장한다 (음성 파일을 서버 밖으로 복사하지 않는다).
사용: python bench/smoke_client.py <wav> [--url ws://127.0.0.1:8766/v1/realtime] [--timeout 120]
"""

import argparse
import asyncio
import base64
import collections
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from websockets.asyncio.client import connect

RATE = 16000
CHUNK = 512  # 32 ms


def load_pcm16(path: str) -> bytes:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != RATE:
        n = int(round(len(audio) * RATE / sr))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


async def run(args) -> dict:
    pcm = load_pcm16(args.wav) + b"\x00\x00" * int(RATE * 2.0)
    counts = collections.Counter()
    turns, out_audio, kva_events = [], bytearray(), []
    t0 = time.monotonic()
    async with connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime",
            "audio": {"input": {"turn_detection": {"type": "server_vad", "interrupt_response": True}},
                      "output": {}}}}))
        for _turn in range(args.turns):
            user_text, bot_text, item_texts, marks, last_item = [], [], [], {}, None
            ts = time.monotonic()

            async def sender():
                for i in range(0, len(pcm), CHUNK * 2):
                    await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                              "audio": base64.b64encode(pcm[i:i + CHUNK * 2]).decode()}))
                    await asyncio.sleep(CHUNK / RATE)
                marks["audio_sent"] = time.monotonic() - ts

            send_task = asyncio.create_task(sender())
            try:
                while time.monotonic() - t0 < args.timeout:
                    try:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                    except asyncio.TimeoutError:
                        continue
                    kind = msg.get("type", "?")
                    counts[kind] += 1
                    if kind.startswith("kva."):
                        kva_events.append(msg)
                    if kind == "conversation.item.input_audio_transcription.completed":
                        user_text.append(msg.get("transcript", ""))
                    elif kind == "response.output_audio_transcript.delta":
                        bot_text.append(msg.get("delta", ""))
                    elif kind == "response.output_audio.delta":
                        marks.setdefault("first_audio", time.monotonic() - ts)
                        last_item = msg.get("item_id") or last_item
                        out_audio.extend(base64.b64decode(msg["delta"]))
                    elif kind == "response.output_item.done":
                        for c in (msg.get("item") or {}).get("content") or []:
                            item_texts.append(c.get("transcript") or c.get("text"))
                    elif kind == "error":
                        print("ERROR EVENT:", json.dumps(msg, ensure_ascii=False)[:500])
                    elif kind == "response.done" and bot_text:
                        marks["response_done"] = time.monotonic() - ts
                        break
            finally:
                send_task.cancel()
                await asyncio.gather(send_task, return_exceptions=True)
            if args.truncate_ms is not None and last_item:
                # 끼어들기 흉내: 응답을 audio_end_ms 까지만 들었다고 서버에 알린다 (브라우저 app.js 와 같은 이벤트)
                await ws.send(json.dumps({"type": "conversation.item.truncate", "item_id": last_item,
                                          "content_index": 0, "audio_end_ms": args.truncate_ms}))
            turns.append({"user_transcript": user_text, "bot_transcript": "".join(bot_text), "item_id": last_item,
                          "output_item_text": item_texts, "marks_sec": {k: round(v, 2) for k, v in marks.items()}})
            await asyncio.sleep(1.0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out_audio:
        sf.write(out, np.frombuffer(bytes(out_audio), dtype="<i2"), RATE)
    return {"events": dict(counts), "turns": turns, "kva_events": [e for e in kva_events if e["type"] != "kva.status"], "out_audio_sec": round(len(out_audio) / 2 / RATE, 2),
            "out_wav": str(out) if out_audio else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--url", default="ws://127.0.0.1:8766/v1/realtime")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--turns", type=int, default=1)
    ap.add_argument("--truncate-ms", dest="truncate_ms", type=int,
                    help="각 응답이 끝난 뒤 이 길이(ms)까지만 들었다고 conversation.item.truncate 를 보낸다")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "logs" / "smoke_out.wav"))
    print(json.dumps(asyncio.run(run(ap.parse_args())), ensure_ascii=False, indent=1))
