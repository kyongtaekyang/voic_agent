"""1단계: speech-to-speech 1.1.0 에 포함된 TTS 모듈 가운데 한국어 후보의 첫 음성 지연과 자원 사용량 측정.

- supertonic : Supertonic-3 (ONNX, CPU). 문장 전체를 한 번에 합성하므로 TTFA = 합성 시간.
- qwen3      : Qwen3-TTS 12Hz 0.6B CustomVoice (faster-qwen3-tts, GPU). 청크 스트리밍이므로 첫 청크 시각이 TTFA.

호출 방식은 speech_to_speech/TTS/*_handler.py 와 같게 맞췄다. 합성 음성은 서버 안 bench/out/ 에만 저장한다.

실행 예:
  .venv/bin/python bench/tts_bench.py --engine supertonic --threads 4 --out docs/reports/tts_supertonic.json
  .venv/bin/python bench/tts_bench.py --engine qwen3 --backend torch --out docs/reports/tts_qwen3_torch.json
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SUPERTONIC_DIR = "/home/yang/pgm/Korean_interface/models/tts/supertonic3"
QWEN3_ID = "/home/yang/pgm/Korean_interface/models/tts/qwen3-tts-0.6b-custom"  # Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice 로컬 사본


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return int(out.decode().split()[0])
    except Exception:  # noqa: BLE001
        return None


def proc_gpu_mib() -> int | None:
    """이 프로세스가 nvidia-smi 에 보이는 GPU 메모리 (CUDA 컨텍스트 포함)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).decode()
        for line in out.strip().splitlines():
            pid, mem = [x.strip() for x in line.split(",")]
            if int(pid) == os.getpid():
                return int(mem)
    except Exception:  # noqa: BLE001
        pass
    return None


def bench_supertonic(a, sentences, outdir):
    from supertonic import TTS
    t = time.perf_counter()
    tts = TTS(model="supertonic-3", model_dir=SUPERTONIC_DIR, auto_download=False,
              intra_op_num_threads=a.threads, inter_op_num_threads=1)
    style = tts.get_voice_style(voice_name=a.voice or "F1")
    load_s = time.perf_counter() - t
    tts.synthesize(text="준비", lang="ko", voice_style=style)  # 워밍업
    rows = []
    for i, s in enumerate(sentences):
        t0 = time.perf_counter()
        wav, _dur = tts.synthesize(text=s, lang="ko", voice_style=style, speed=1.0)
        sec = time.perf_counter() - t0
        wav = np.asarray(wav).squeeze()
        sr = getattr(tts, "sample_rate", None) or 44100
        dur = wav.shape[-1] / sr
        sf.write(outdir / f"supertonic_{i}.wav", wav, sr)
        rows.append({"i": i, "text": s, "ttfa_s": round(sec, 3), "total_s": round(sec, 3),
                     "audio_s": round(dur, 2), "rtf": round(sec / dur, 3), "sr": sr})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    return {"engine": "supertonic-3", "device": "cpu", "threads": a.threads, "voice": a.voice or "F1",
            "load_s": round(load_s, 1), "rows": rows}


def bench_qwen3(a, sentences, outdir):
    import torch
    from faster_qwen3_tts import FasterQwen3TTS
    base = gpu_used_mib()
    t = time.perf_counter()
    kw = {"attn_implementation": a.attn} if a.attn else {}
    model = FasterQwen3TTS.from_pretrained(a.qwen3_model, device=a.device, dtype=torch.bfloat16,
                                           backend=a.backend, **kw)
    load_s = time.perf_counter() - t
    speakers = None
    for cand in (model, getattr(model, "model", None)):
        f = getattr(cand, "get_supported_speakers", None)
        if callable(f):
            speakers = list(f() or [])
            break
    speaker = a.voice or "Sohee"
    print(f"speakers={speakers} use={speaker}", flush=True)
    # 워밍업
    for _ in model.generate_custom_voice_streaming(text="준비", speaker=speaker, language="Korean", chunk_size=a.chunk):
        pass
    rows = []
    for i, s in enumerate(sentences):
        t0 = time.perf_counter()
        first = None
        chunks, sr = [], None
        for audio, sr, _info in model.generate_custom_voice_streaming(
                text=s, speaker=speaker, language="Korean", chunk_size=a.chunk):
            if first is None and np.asarray(audio).size:
                first = time.perf_counter() - t0
            chunks.append(np.asarray(audio, dtype=np.float32).squeeze())
        sec = time.perf_counter() - t0
        wav = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
        dur = wav.shape[-1] / sr
        size = "1.7b" if "1.7B" in a.qwen3_model else "0.6b"
        sf.write(outdir / f"qwen3-{size}_{a.backend}_{a.device}_{i}.wav", wav, sr)
        rows.append({"i": i, "text": s, "ttfa_s": round(first or sec, 3), "total_s": round(sec, 3),
                     "audio_s": round(dur, 2), "rtf": round(sec / dur, 3), "sr": sr})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    on_gpu = a.device.startswith("cuda")
    return {"engine": "qwen3-tts", "model": a.qwen3_model, "device": a.device, "backend": a.backend,
            "attn": a.attn or "library default", "chunk_size": a.chunk,
            "speaker": speaker, "speakers": speakers, "load_s": round(load_s, 1),
            "gpu_mib_process": proc_gpu_mib() if on_gpu else 0,
            "torch_max_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20) if on_gpu else 0,
            "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["supertonic", "qwen3"], required=True)
    ap.add_argument("--sentences", default=str(ROOT / "bench/data/tts_sentences_ko.txt"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--backend", default="torch", help="qwen3: torch | ggml")
    ap.add_argument("--chunk", type=int, default=8, help="qwen3 streaming chunk_size (s2s 기본값 8)")
    ap.add_argument("--device", default="cuda", help="qwen3: cuda | cpu")
    ap.add_argument("--attn", help="qwen3 attn_implementation (s2s 기본값 eager, 라이브러리 기본값 sdpa)")
    ap.add_argument("--qwen3-model", dest="qwen3_model", default=QWEN3_ID)
    ap.add_argument("--voice")
    ap.add_argument("--out")
    a = ap.parse_args()
    sentences = [x.strip() for x in Path(a.sentences).read_text(encoding="utf-8").splitlines() if x.strip()]
    outdir = ROOT / "bench/out"
    outdir.mkdir(parents=True, exist_ok=True)
    res = bench_supertonic(a, sentences, outdir) if a.engine == "supertonic" else bench_qwen3(a, sentences, outdir)
    r = [x for x in res["rows"] if x["i"] < len(sentences) - 1]  # 마지막 문장(정규화 안 한 숫자)은 지연 통계에서 뺀다
    res["ttfa_mean_s"] = round(float(np.mean([x["ttfa_s"] for x in r])), 3)
    res["rtf_mean"] = round(float(np.mean([x["rtf"] for x in r])), 3)
    res["load_avg"] = os.getloadavg()
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
