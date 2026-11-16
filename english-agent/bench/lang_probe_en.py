"""영어 Agent 비영어 발화 감지 확인 (프롬프트 2.3절, 4단계 준비).

Nemotron ASR Streaming 0.6B 에 영어·한국어 음성을 language=en-US 와 auto 로 넣고, core/asr_monitor.py 의 판정 규칙이
비영어 발화를 얼마나 잡는지 본다. GPU 를 쓰지 않는다.

- 합성 음성: Supertonic(CPU)으로 서버 안에서 만들고 bench/out/lang_probe_en/ 에만 둔다.
- 실제 한국어 음성: korean_agent/bench/data/ko_ref (직원 참조 녹음 링크). 인식 원문은 결과 파일에 쓰지 않고 판정만 남긴다.
실행 (한국어 Agent venv 사용: transformers·supertonic 이 있음):
  ../korean_agent/.venv/bin/python bench/lang_probe_en.py --out docs/reports/lang_probe_en.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForRNNT, AutoProcessor  # noqa: E402
from transformers.audio_utils import load_audio  # noqa: E402

from core.asr_monitor import LanguageMismatchDetector  # noqa: E402

ASR = "/home/yang/pgm/Korean_interface/models/stt/nemotron-3.5-asr-streaming-0.6b"
SUPERTONIC_DIR = "/home/yang/pgm/Korean_interface/models/tts/supertonic3"
KO_REF = ROOT.parent / "korean_agent/bench/data/ko_ref"
SENTS = [
    ("en", "Hello, can you help me with my homework today?"),
    ("en", "I like playing soccer with my friends after school."),
    ("en", "What is your favorite animal?"),
    ("en", "Yesterday I went to the park with my little sister."),
    ("en", "Could you say that again a little more slowly, please?"),
    ("en", "My mom made pancakes for breakfast this morning."),
    ("en", "I don't know the answer."),
    ("en", "How do you spell elephant?"),
    ("ko", "오늘은 공룡에 대해서 배우고 싶어요."),
    ("ko", "선생님, 이거 영어로 뭐라고 해요?"),
    ("ko", "잘 모르겠어요. 다시 말해 주세요."),
    ("ko", "저는 강아지를 제일 좋아해요."),
    ("mix", "선생님, apple 은 영어로 사과예요?"),
    ("mix", "I like 떡볶이 very much."),
]
SETTINGS = ("en-US", "auto")


def recognize(proc, model, audio, sr, setting):
    inputs = proc(audio, sampling_rate=sr, language=setting)
    inputs.to(model.device, dtype=model.dtype)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=512)
    dt = time.perf_counter() - t0
    hyp = proc.decode(out.sequences, skip_special_tokens=setting != "auto")
    return (hyp[0] if isinstance(hyp, list) else hyp).strip(), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    from supertonic import TTS
    tts = TTS(model="supertonic-3", model_dir=SUPERTONIC_DIR, auto_download=False,
              intra_op_num_threads=a.threads, inter_op_num_threads=1)
    style = tts.get_voice_style(voice_name="F1")
    outdir = ROOT / "bench/out/lang_probe_en"
    outdir.mkdir(parents=True, exist_ok=True)
    synth = []
    for i, (lang, text) in enumerate(SENTS):
        wav, _ = tts.synthesize(text=text, lang="en" if lang == "en" else "ko", voice_style=style, speed=1.0)
        p = outdir / f"{i:02d}_{lang}.wav"
        sf.write(p, np.asarray(wav).squeeze(), getattr(tts, "sample_rate", None) or 44100)
        synth.append((lang, text, p))

    torch.set_num_threads(a.threads)
    proc = AutoProcessor.from_pretrained(ASR)
    model = AutoModelForRNNT.from_pretrained(ASR, device_map="cpu").eval()
    sr = proc.feature_extractor.sampling_rate
    det = LanguageMismatchDetector()
    rows, real = [], []
    for lang, text, p in synth:
        audio = load_audio(str(p), sampling_rate=sr)
        row = {"lang": lang, "ref": text, "sec": round(len(audio) / sr, 2)}
        for s in SETTINGS:
            hyp, dt = recognize(proc, model, audio, sr, s)
            u = det.classify(hyp)
            row[s] = {"hyp": hyp, "detected": u.lang, "english": u.english, "rtf": round(dt / row["sec"], 3)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    for p in sorted(KO_REF.glob("*.wav")):
        audio = load_audio(str(p), sampling_rate=sr)
        row = {"clip": p.stem.split("_")[0], "sec": round(len(audio) / sr, 1)}
        for s in SETTINGS:
            hyp, dt = recognize(proc, model, audio, sr, s)
            u = det.classify(hyp)
            # 직원 녹음의 인식 원문은 남기지 않는다
            row[s] = {"chars": len(u.text), "detected": u.lang, "english": u.english, "rtf": round(dt / row["sec"], 3)}
        real.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def rate(items, s, want_english):
        judged = [r[s]["english"] for r in items]
        hit = sum(1 for e in judged if e is want_english)
        undecided = sum(1 for e in judged if e is None)
        return {"n": len(judged), "correct": hit, "undecided": undecided}

    summary = {}
    for s in SETTINGS:
        summary[s] = {
            "synth_en_kept_english": rate([r for r in rows if r["lang"] == "en"], s, True),
            "synth_ko_flagged": rate([r for r in rows if r["lang"] == "ko"], s, False),
            "synth_mix": [r[s]["detected"] for r in rows if r["lang"] == "mix"],
            "real_ko_flagged": rate(real, s, False),
        }
    res = {"model": Path(ASR).name, "threads": a.threads, "summary": summary, "synth": rows, "real_ko": real}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
