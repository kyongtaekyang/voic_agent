"""텍스트 정규화 수락 시험: 1단계에서 모든 TTS 가 잘못 읽은 6번 문장(숫자 원문)을 정규화한 뒤 다시 합성하고,
Nemotron 오프라인 인식 결과와의 CER 을 정규화 전 음성(bench/out/*_5.wav)과 비교한다. GPU 를 쓰지 않는다.

실행: .venv/bin/python bench/normalizer_acceptance.py --out docs/reports/normalizer_acceptance.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForRNNT, AutoProcessor  # noqa: E402
from transformers.audio_utils import load_audio  # noqa: E402

from asr_cpu_rtf import cer, offline  # noqa: E402
from realtime_server.text_normalizer import normalize  # noqa: E402

ASR = "/home/yang/pgm/Korean_interface/models/stt/nemotron-3.5-asr-streaming-0.6b"
SUPERTONIC_DIR = "/home/yang/pgm/Korean_interface/models/tts/supertonic3"
QWEN3_HF_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"  # ggml 백엔드는 HF id 만 받는다 (로컬 캐시 사용)


def synth_supertonic(text: str, path: Path) -> None:
    from supertonic import TTS
    tts = TTS(model="supertonic-3", model_dir=SUPERTONIC_DIR, auto_download=False,
              intra_op_num_threads=8, inter_op_num_threads=1)
    wav, _ = tts.synthesize(text=text, lang="ko", voice_style=tts.get_voice_style(voice_name="F1"), speed=1.0)
    sf.write(path, np.asarray(wav).squeeze(), getattr(tts, "sample_rate", None) or 44100)


def synth_qwen3_ggml_cpu(text: str, path: Path) -> None:
    from faster_qwen3_tts import FasterQwen3TTS
    model = FasterQwen3TTS.from_pretrained(QWEN3_HF_ID, device="cpu", dtype=torch.float32, backend="ggml")
    chunks, sr = [], 24000
    for audio, sr, _ in model.generate_custom_voice_streaming(text=text, speaker="Sohee", language="Korean",
                                                             chunk_size=8):
        chunks.append(np.asarray(audio, dtype=np.float32).squeeze())
    sf.write(path, np.concatenate(chunks), sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    raw = [x.strip() for x in (ROOT / "bench/data/tts_sentences_ko.txt").read_text(encoding="utf-8").splitlines()
           if x.strip()][5]
    spoken = normalize(raw)
    print(json.dumps({"raw": raw, "normalized": spoken}, ensure_ascii=False), flush=True)
    outdir = ROOT / "bench/out/normalized"
    outdir.mkdir(parents=True, exist_ok=True)
    new = {"supertonic": outdir / "supertonic_5n.wav", "qwen3-0.6b_ggml_cpu": outdir / "qwen3-0.6b_ggml_cpu_5n.wav"}
    synth_supertonic(spoken, new["supertonic"])
    try:
        synth_qwen3_ggml_cpu(spoken, new["qwen3-0.6b_ggml_cpu"])
    except Exception as e:  # noqa: BLE001
        print(f"qwen3 ggml cpu failed: {type(e).__name__}: {e}", flush=True)
        new.pop("qwen3-0.6b_ggml_cpu")

    torch.set_num_threads(8)
    proc = AutoProcessor.from_pretrained(ASR)
    model = AutoModelForRNNT.from_pretrained(ASR, device_map="cpu").eval()
    sr = proc.feature_extractor.sampling_rate

    def hyp(p: Path) -> str:
        with torch.inference_mode():
            return offline(model, proc, load_audio(str(p), sampling_rate=sr), sr, "ko-KR")[0].strip()

    rows = []
    for p in sorted((ROOT / "bench/out").glob("*_5.wav")):
        h = hyp(p)
        rows.append({"audio": p.name, "tts_input": "raw", "hyp": h,
                     "cer_vs_raw": round(cer(raw, h), 4), "cer_vs_normalized": round(cer(spoken, h), 4)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    for name, p in new.items():
        h = hyp(p)
        rows.append({"audio": f"normalized/{p.name}", "engine": name, "tts_input": "normalized", "hyp": h,
                     "cer_vs_normalized": round(cer(spoken, h), 4)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"raw": raw, "normalized": spoken, "rows": rows}, ensure_ascii=False,
                                          indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
