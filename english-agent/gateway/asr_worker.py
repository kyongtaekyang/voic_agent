"""사용자 음성 병행 인식 (프롬프트 2.3절). Moshi 와 별도로 사용자 발화를 받아써서 자막과 언어 감지에 쓴다.

에너지 기반으로 발화 구간을 자르고(발화 끝 = end_silence_ms 이상 조용함), 구간마다 인식기를 스레드에서 실행한다.
인식기는 Nemotron ASR Streaming 0.6B 를 CPU 에서 language=auto 로 쓴다 (docs/reports/cpu_prebuild_20261116.md 4절).
지금은 발화가 끝난 뒤 한 번에 인식하는 확정 자막만 낸다. 말하는 중 부분 자막은 5단계 후반에 lookahead 스트리밍으로 바꾼다.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Callable

import numpy as np

from core.asr_monitor import LanguageMismatchDetector, split_lang_tag

log = logging.getLogger("eva.asr")
ASR_RATE = 16000


def resample_linear(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or x.size == 0:
        return x
    n = int(round(x.size * dst / src))
    return np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)


class NemotronRecognizer:
    """16kHz float32 → 인식 결과(언어 태그 포함 원문). 처음 호출할 때 모델을 읽는다."""

    def __init__(self, model: str, language: str = "auto", threads: int = 8):
        self.model_path, self.language, self.threads = model, language, threads
        self._proc = self._model = None

    def load(self) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor
        torch.set_num_threads(self.threads)
        self._proc = AutoProcessor.from_pretrained(self.model_path)
        self._model = AutoModelForRNNT.from_pretrained(self.model_path).eval()  # 기본 장치 CPU (device_map 은 accelerate 필요)

    def __call__(self, audio16: np.ndarray) -> str:
        import torch
        if self._model is None:
            self.load()
        inputs = self._proc(audio16, sampling_rate=ASR_RATE, language=self.language)
        inputs.to(self._model.device, dtype=self._model.dtype)
        with torch.inference_mode():
            out = self._model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=512)
        hyp = self._proc.decode(out.sequences, skip_special_tokens=self.language != "auto")
        return (hyp[0] if isinstance(hyp, list) else hyp).strip()


class AsrWorker:
    def __init__(self, recognizer: Callable[[np.ndarray], str], detector: LanguageMismatchDetector | None = None, *,
                 rate: int = 24000, frame_ms: int = 20, start_rms: float = 0.02, end_silence_ms: int = 500,
                 min_speech_ms: int = 300, max_utt_s: float = 15.0, korean_agent_url: str | None = None):
        self.recognize = recognizer
        self.detector = detector or LanguageMismatchDetector()
        self.rate, self.frame = rate, int(rate * frame_ms / 1000)
        self.frame_ms, self.start_rms = frame_ms, start_rms
        self.end_frames = end_silence_ms // frame_ms
        self.min_frames = min_speech_ms // frame_ms
        self.max_frames = int(max_utt_s * 1000 / frame_ms)
        self.korean_agent_url = korean_agent_url
        self.q: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self.utterances = 0

    def feed(self, pcm: np.ndarray) -> None:
        self.q.put_nowait(pcm)

    def close(self) -> None:
        self.q.put_nowait(None)

    async def run(self, emit) -> None:
        loop = asyncio.get_running_loop()
        buf = np.zeros(0, np.float32)
        utt: list[np.ndarray] = []
        voiced = silent = 0
        pending: set[asyncio.Task] = set()

        async def finish(frames: list[np.ndarray]) -> None:
            audio = resample_linear(np.concatenate(frames), self.rate, ASR_RATE)
            t0 = time.perf_counter()
            try:
                raw = await loop.run_in_executor(None, self.recognize, audio)
            except Exception as e:  # noqa: BLE001
                log.exception("ASR 실패")
                await emit({"type": "eva.error", "source": "asr", "detail": str(e)[:200]})
                return
            tag, text = split_lang_tag(raw)
            self.utterances += 1
            await emit({"type": "eva.user_text", "text": text, "final": True, "lang": tag,
                        "audio_s": round(audio.size / ASR_RATE, 2), "asr_ms": round(1000 * (time.perf_counter() - t0))})
            ev = self.detector.feed(raw)
            if ev:
                ev["korean_agent_url"] = self.korean_agent_url
                await emit(ev)

        while True:
            pcm = await self.q.get()
            if pcm is None:
                break
            buf = np.concatenate([buf, pcm])
            while buf.size >= self.frame:
                fr, buf = buf[:self.frame], buf[self.frame:]
                loud = float(np.sqrt(np.mean(fr * fr))) >= self.start_rms
                if loud:
                    voiced += 1
                    silent = 0
                    utt.append(fr)
                elif utt:
                    silent += 1
                    utt.append(fr)
                if utt and (silent >= self.end_frames or len(utt) >= self.max_frames):
                    if voiced >= self.min_frames:
                        t = asyncio.create_task(finish(utt))
                        pending.add(t)
                        t.add_done_callback(pending.discard)
                    utt, voiced, silent = [], 0, 0
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
