"""전체 시험용 가짜 Transcribe 30B 서버 (GPU 없음). 응답 형식은 실제 NeMo ctc_timestamp_middleware 와 같다
(NeMo Speech main d55f2a4: completion 최상위 "ctc_timestamps" = {"words":[{"word","start","end","speaker"}], "diarization",
"speaker_tag_to_diarization_speaker", "error"}). 단어는 음성 에너지 구간에서 만든다 (내용은 가짜, 시각은 실제 소리 위치).

  python mock_transcribe.py --port 18002
"""

import argparse
import base64
import io
import time
import wave

import numpy as np
from aiohttp import web

CALLS = {"chat": 0}


def voiced_runs(x: np.ndarray, rate: int, frame_ms: int = 20, rms: float = 0.02) -> list[tuple[float, float]]:
    n = int(rate * frame_ms / 1000)
    f = x[: x.size // n * n].reshape(-1, n)
    on = np.sqrt((f ** 2).mean(axis=1)) >= rms
    runs, start = [], None
    for i, v in enumerate(list(on) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start * frame_ms / 1000, i * frame_ms / 1000))
            start = None
    return runs


async def models(_):
    return web.json_response({"object": "list", "data": [{"id": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"}]})


async def chat(request):
    body = await request.json()
    CALLS["chat"] += 1
    if (body.get("mm_processor_kwargs") or {}).get("capture_ctc_timestamps") is not True or body.get("stream"):
        return web.json_response({"error": "e2e mock: 타임스탬프 요청 형식이 아님"}, status=400)
    url = next(c["audio_url"]["url"] for c in body["messages"][-1]["content"] if c["type"] == "audio_url")
    raw = base64.b64decode(url.split(",", 1)[1])
    with wave.open(io.BytesIO(raw)) as w:
        rate = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    words = []
    for s, e in voiced_runs(x, rate):
        words.append({"word": f"w{len(words)}", "start": round(s, 3), "end": round(e, 3), "speaker": "0"})
    text = " ".join(w["word"] for w in words)
    return web.json_response({
        "id": f"chatcmpl-e2e-{CALLS['chat']}", "object": "chat.completion", "created": int(time.time()), "model": body["model"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "ctc_timestamps": {"words": words, "diarization": [{"speaker": 0, "start": 0.0, "end": round(x.size / rate, 3)}],
                           "speaker_tag_to_diarization_speaker": {}, "error": None},
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18002)
    a = ap.parse_args()
    app = web.Application(client_max_size=200 * 1024 * 1024)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat)
    web.run_app(app, host="127.0.0.1", port=a.port, access_log=None, print=None)


if __name__ == "__main__":
    main()
