"""영어 발화 감지 방식 확인: 영어 합성 음성을 Nemotron 에 language=ko-KR 과 auto 로 각각 넣어 결과를 비교한다.

영어 음성은 Supertonic(CPU)으로 서버 안에서 만들고 bench/out/lang_probe/ 에만 둔다. GPU 를 쓰지 않는다.
실행: .venv/bin/python bench/lang_mismatch_probe.py --out docs/reports/lang_mismatch_probe.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForRNNT, AutoProcessor  # noqa: E402
from transformers.audio_utils import load_audio  # noqa: E402

from realtime_server.stt_streaming import hangul_ratio  # noqa: E402

ASR = "/home/yang/pgm/Korean_interface/models/stt/nemotron-3.5-asr-streaming-0.6b"
SUPERTONIC_DIR = "/home/yang/pgm/Korean_interface/models/tts/supertonic3"
SENTS = [
    ("en", "Hello, can you help me with my homework today?"),
    ("en", "I like playing soccer with my friends after school."),
    ("en", "What is your favorite animal?"),
    ("ko", "오늘은 공룡에 대해서 배우고 싶어요."),
    ("mix", "선생님, apple 은 영어로 사과예요?"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    from supertonic import TTS
    tts = TTS(model="supertonic-3", model_dir=SUPERTONIC_DIR, auto_download=False,
              intra_op_num_threads=8, inter_op_num_threads=1)
    style = tts.get_voice_style(voice_name="F1")
    outdir = ROOT / "bench/out/lang_probe"
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (lang, text) in enumerate(SENTS):
        wav, _ = tts.synthesize(text=text, lang="ko" if lang != "en" else "en", voice_style=style, speed=1.0)
        p = outdir / f"{i}_{lang}.wav"
        sf.write(p, np.asarray(wav).squeeze(), getattr(tts, "sample_rate", None) or 44100)
        paths.append(p)

    torch.set_num_threads(8)
    proc = AutoProcessor.from_pretrained(ASR)
    model = AutoModelForRNNT.from_pretrained(ASR, device_map="cpu").eval()
    sr = proc.feature_extractor.sampling_rate
    rows = []
    for (lang, text), p in zip(SENTS, paths):
        audio = load_audio(str(p), sampling_rate=sr)
        row = {"lang": lang, "ref": text}
        for setting in ("ko-KR", "auto"):
            inputs = proc(audio, sampling_rate=sr, language=setting)
            inputs.to(model.device, dtype=model.dtype)
            with torch.inference_mode():
                out = model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=512)
            hyp = proc.decode(out.sequences, skip_special_tokens=setting != "auto")
            hyp = hyp[0] if isinstance(hyp, list) else hyp
            row[setting] = {"hyp": hyp.strip(), "hangul_ratio": hangul_ratio(hyp)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
