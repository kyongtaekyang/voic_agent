"""TTS 출력(bench/out/*.wav)을 Nemotron ASR(오프라인)로 다시 인식해서 원문과의 CER 을 잰다. 명료도의 보조 지표.

실행: CUDA_VISIBLE_DEVICES= <Korean_interface venv python> bench/tts_roundtrip.py --out docs/reports/tts_roundtrip.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from transformers import AutoModelForRNNT, AutoProcessor  # noqa: E402
from transformers.audio_utils import load_audio  # noqa: E402

from asr_cpu_rtf import cer, offline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
M = "/home/yang/pgm/Korean_interface/models/stt/nemotron-3.5-asr-streaming-0.6b"

ap = argparse.ArgumentParser()
ap.add_argument("--out")
a = ap.parse_args()
torch.set_num_threads(8)
sents = [x.strip() for x in (ROOT / "bench/data/tts_sentences_ko.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
proc = AutoProcessor.from_pretrained(M)
model = AutoModelForRNNT.from_pretrained(M, device_map="cpu").eval()
sr = proc.feature_extractor.sampling_rate
rows = []
for w in sorted((ROOT / "bench/out").glob("*.wav")):
    engine, idx = w.stem.rsplit("_", 1)
    ref = sents[int(idx)]
    with torch.inference_mode():
        hyp, _ = offline(model, proc, load_audio(str(w), sampling_rate=sr), sr, "ko-KR")
    rows.append({"engine": engine, "i": int(idx), "ref": ref, "hyp": hyp.strip(), "cer": round(cer(ref, hyp), 4)})
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
summary = {}
for e in sorted({r["engine"] for r in rows}):
    rs = [r for r in rows if r["engine"] == e and r["i"] < len(sents) - 1]
    summary[e] = round(sum(r["cer"] for r in rs) / len(rs), 4)
print(json.dumps({"mean_cer_excl_last": summary}, ensure_ascii=False))
if a.out:
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
