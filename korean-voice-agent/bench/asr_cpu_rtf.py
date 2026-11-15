"""1단계: Nemotron 3.5 ASR Streaming 0.6B 의 한국어 인식과 CPU 실시간 처리 속도(RTF) 측정.

- GPU 는 쓰지 않는다 (CUDA_VISIBLE_DEVICES="" 로 강제).
- 오프라인(파일 전체) 전사와, 모델 카드의 Transformers 스트리밍 예시를 그대로 따른 청크 스트리밍을 각각 잰다.
- RTF = 처리 시간 / 음성 길이. 1.0 미만이어야 실시간 처리 가능.
- CER 은 reference.txt 와 비교 (공백·문장부호 제거 후 문자 단위 편집 거리).

실행 예:
  CUDA_VISIBLE_DEVICES= python bench/asr_cpu_rtf.py --threads 8 --lookahead 0 3 6 13 \
      --wav a.wav --wav b.wav --out docs/reports/asr_cpu_rtf.json
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from threading import Thread

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer  # noqa: E402
from transformers.audio_utils import load_audio  # noqa: E402


def norm(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"[\s\.,!?~·…\"'“”‘’()\[\]{}:;\-]", "", s)


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    if not r:
        return float("nan")
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def offline(model, processor, audio, sr, lang):
    t0 = time.perf_counter()
    inputs = processor(audio, sampling_rate=sr, language=lang)
    inputs.to(model.device, dtype=model.dtype)
    out = model.generate(**inputs, return_dict_in_generate=True)
    text = processor.decode(out.sequences, skip_special_tokens=True)
    if isinstance(text, list):
        text = text[0]
    return text, time.perf_counter() - t0


def streaming(model, processor, audio, sr, lang, lookahead):
    """모델 카드 예시와 같은 방식. 청크를 가능한 한 빨리 넣어 처리 시간을 잰다."""
    processor.set_num_lookahead_tokens(lookahead)
    latency_ms = processor.streaming_latency_ms
    t0 = time.perf_counter()
    first = processor(audio[: processor.num_samples_first_audio_chunk], sampling_rate=sr,
                      is_streaming=True, is_first_audio_chunk=True, language=lang, return_tensors="pt")
    first = first.to(model.device, dtype=model.dtype)
    fed = {"chunks": 0, "last_fed": None}

    def gen():
        yield first.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]
        fed["chunks"] += 1
        idx = processor.num_mel_frames_first_audio_chunk
        hop, n_fft = processor.feature_extractor.hop_length, processor.feature_extractor.n_fft
        start = idx * hop - n_fft // 2
        while (end := start + processor.num_samples_per_audio_chunk) < audio.shape[0]:
            # lookahead 0 에서는 첫 청크가 짧아 start 가 음수가 된다. 모델 카드 예시를 그대로 쓰면
            # audio[음수:end] 가 빈 배열이 되어 stft 가 실패하므로, 모자란 앞부분을 0 으로 채운다.
            seg = audio[start:end] if start >= 0 else np.concatenate([np.zeros(-start, dtype=audio.dtype), audio[:end]])
            x = processor(seg, sampling_rate=sr, is_streaming=True, is_first_audio_chunk=False,
                          language=lang, return_tensors="pt").to(model.device, dtype=model.dtype)
            yield x.input_features
            fed["chunks"] += 1
            idx += processor.num_mel_frames_per_audio_chunk
            start = idx * hop - n_fft // 2
        fed["last_fed"] = time.perf_counter()

    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
    err = {}

    def run():
        # 생성 스레드가 예외로 죽으면 streamer 가 끝나지 않아 메인 스레드가 영원히 기다린다. 예외를 넘겨받고 streamer 를 닫는다.
        try:
            model.generate(**{**first, "input_features": gen(), "streamer": streamer})
        except Exception as e:  # noqa: BLE001
            err["e"] = e
            streamer.end()

    th = Thread(target=run)
    th.start()
    text = "".join(streamer)
    th.join()
    if "e" in err:
        raise err["e"]
    t1 = time.perf_counter()
    return text, t1 - t0, latency_ms, fed["chunks"], (t1 - fed["last_fed"]) if fed["last_fed"] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/nemotron-3.5-asr-streaming-0.6b")
    ap.add_argument("--wav", action="append", required=True)
    ap.add_argument("--lang", default="ko-KR")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--lookahead", type=int, nargs="+", default=[0, 3, 6, 13])
    ap.add_argument("--out")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    t = time.perf_counter()
    processor = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForRNNT.from_pretrained(a.model, device_map="cpu", torch_dtype=torch.float32).eval()
    load_s = time.perf_counter() - t
    sr = processor.feature_extractor.sampling_rate
    print(f"loaded in {load_s:.1f}s, sr={sr}, threads={a.threads}, device={model.device}", flush=True)

    # 워밍업 (첫 실행의 초기화 비용 제외)
    warm = load_audio(a.wav[0], sampling_rate=sr)[: sr * 3]
    offline(model, processor, warm, sr, a.lang)

    rows = []
    for w in a.wav:
        audio = load_audio(w, sampling_rate=sr)
        dur = audio.shape[0] / sr
        ref_f = Path(w).with_suffix(".txt")
        ref = ref_f.read_text(encoding="utf-8").strip() if ref_f.exists() else ""
        with torch.inference_mode():
            text, sec = offline(model, processor, audio, sr, a.lang)
        row = {"wav": w, "dur_s": round(dur, 2), "mode": "offline", "proc_s": round(sec, 2),
               "rtf": round(sec / dur, 3), "cer": round(cer(ref, text), 4) if ref else None, "text": text}
        rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
        for la in a.lookahead:
            try:
                with torch.inference_mode():
                    text, sec, lat, n, tail = streaming(model, processor, audio, sr, a.lang, la)
            except Exception as e:  # noqa: BLE001  한 조건이 실패해도 나머지 측정은 계속한다
                row = {"wav": w, "dur_s": round(dur, 2), "mode": f"stream_la{la}", "error": repr(e)[:300]}
                rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
                continue
            row = {"wav": w, "dur_s": round(dur, 2), "mode": f"stream_la{la}", "chunk_latency_ms": lat,
                   "chunks": n, "proc_s": round(sec, 2), "rtf": round(sec / dur, 3),
                   "per_chunk_ms": round(sec * 1000 / max(n, 1), 1),
                   "final_after_last_chunk_ms": round(tail * 1000, 1) if tail is not None else None,
                   "cer": round(cer(ref, text), 4) if ref else None, "text": text}
            rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"threads": a.threads, "load_s": round(load_s, 1), "rows": rows},
                                          ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
