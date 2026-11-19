"""튜터(Agent) 음성의 단어 타임스탬프: Transcribe 30B + NeMo ctc_timestamp_middleware (후처리 모드, GPU 서버 :8002).

  python -m slow_path.words pending                 # 타임스탬프가 필요한 세션 수 (GPU 없이)
  python -m slow_path.words studio [--limit N]      # 녹음 스튜디오 세션 → <세션>/tutor_words.json
  python -m slow_path.words queue  [--limit N]      # 영어 대기열 candidate 세션 → transcript.json agent.words

응답 형식 (NeMo Speech main d55f2a4, nemo/collections/speechlm2/vllm/salm/ctc_serving.py·ctc_timestamps.py 원문과 모델 카드로 확인):
- 요청: chat completions 에 "mm_processor_kwargs": {"capture_ctc_timestamps": true}, stream=false, n=1, skip_special_tokens=false
- 응답 최상위에 "ctc_timestamps": {"words": [{"word", "start", "end", "speaker"}], "diarization": [...],
  "speaker_tag_to_diarization_speaker": {...}, "error": null | "실패 이유"}. 시각은 입력 음성 기준 초(소수 셋째 자리).
- 서버를 --middleware 없이 띄우면 ctc_timestamps 가 없다 → 오류로 멈춘다 (조용히 빈 정렬을 만들지 않음).

결과
- 스튜디오: tutor_words.json = [{"word","start","end"}] (finetune/prepare.py 가 읽는 중립 형식),
  tutor_words.meta.json = 모델·지시문·전사문·시각, tutor_words.raw.json = 응답의 ctc_timestamps 원본 (형식 변경 확인용)
- 대기열: transcript.json 의 agent.words (+ agent.ctc_raw)
음성은 127.0.0.1 서버로만 보낸다 (slow_path.run.local_only).
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import requests
import yaml

from finetune.prepare import training_gate
from slow_path.run import ROOT, local_only, read_wav, wav16k_bytes

log = logging.getLogger("eva.words")

# 모델 카드 "The words as spoken" 지시문 (축어 전사: 군말·반복 포함). 학습 정렬은 실제로 소리 난 단어여야 한다
VERBATIM_PROMPT = ("Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. "
                   "Formatting: punctuation and capitalization. Output: transcript only.")


class TimestampError(RuntimeError):
    pass


def config_path() -> Path:
    import os
    return Path(os.environ.get("EVA_CONFIG") or ROOT / "config" / "services.yaml")


def load_all() -> dict:
    return yaml.safe_load(config_path().read_text(encoding="utf-8")) or {}


def rel(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def voiced_s(x: np.ndarray, rate: int, rms: float = 0.01, frame_ms: int = 20) -> float:
    n = max(1, int(rate * frame_ms / 1000))
    if x.size < n:
        return 0.0
    f = x[: x.size // n * n].reshape(-1, n)
    return float((np.sqrt((f ** 2).mean(axis=1)) >= rms).sum() * frame_ms / 1000)


def parse_ctc(completion: dict) -> tuple[list[dict], dict]:
    """응답 → ([{"word","start","end"}], ctc_timestamps 원본). 형식이 다르거나 정렬 실패면 TimestampError."""
    ct = completion.get("ctc_timestamps")
    if ct is None:
        raise TimestampError("응답에 ctc_timestamps 가 없습니다. Transcribe 서버를 --middleware "
                             "nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware 로 띄웠는지 확인하세요")
    if not isinstance(ct, dict) or "words" not in ct:
        raise TimestampError(f"ctc_timestamps 형식이 예상과 다릅니다: {str(ct)[:200]}")
    if ct.get("error") is not None:
        raise TimestampError(f"정렬 실패: {ct['error']}")
    words = []
    for w in ct["words"]:
        text = str(w.get("word", "")).strip()
        s, e = float(w["start"]), float(w["end"])
        if text and e > s >= 0:
            words.append({"word": text, "start": round(s, 3), "end": round(e, 3)})
    words.sort(key=lambda w: (w["start"], w["end"]))
    return words, ct


def request_completion(audio_wav16k: bytes, cfg: dict, url: str) -> dict:
    body = {
        "model": cfg.get("transcribe_model", "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"),
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": cfg.get("words_prompt") or VERBATIM_PROMPT},
            {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + base64.b64encode(audio_wav16k).decode()}}]}],
        "max_tokens": int(cfg.get("transcribe_max_tokens", 8192)), "temperature": 0.0, "top_p": 1.0, "n": 1, "stream": False,
        # 모델 카드 ask(timestamps=True) 와 같은 값
        "chat_template_kwargs": {"enable_thinking": False}, "skip_special_tokens": False,
        "spaces_between_special_tokens": False, "seed": 0,
        "mm_processor_kwargs": {"capture_ctc_timestamps": True},
    }
    with requests.Session() as s:
        s.trust_env = False  # 프록시를 거치지 않는다 (음성 외부 전송 방지)
        r = s.post(f"{local_only(url)}/v1/chat/completions", json=body, timeout=float(cfg.get("timeout_s", 1800)))
    if r.status_code != 200:
        raise TimestampError(f"Transcribe 서버 오류 {r.status_code}: {r.text[:300]}")
    return r.json()


def word_timestamps(wav: Path, cfg: dict, url: str) -> dict:
    """한 트랙의 단어 타임스탬프. 무음 트랙은 서버에 보내지 않는다. 소리가 있는데 단어가 0개면 오류 (빈 정렬로 학습하지 않게)."""
    x, rate = read_wav(wav)
    voiced = voiced_s(x, rate)
    if voiced < 0.2:
        return {"words": [], "text": "", "skipped": "무음", "voiced_s": round(voiced, 2)}
    t0 = time.perf_counter()
    comp = request_completion(wav16k_bytes(x, rate), cfg, url)
    words, ct = parse_ctc(comp)
    ch = (comp.get("choices") or [{}])[0]
    text = ((ch.get("message") or {}).get("content") or "").strip()
    if not words and voiced >= 1.0:
        raise TimestampError(f"음성 {voiced:.1f}초가 있는데 단어가 0개입니다 (전사: {text[:80]!r})")
    return {"words": words, "text": text, "finish_reason": ch.get("finish_reason"), "truncated": ch.get("finish_reason") == "length",
            "voiced_s": round(voiced, 2), "audio_s": round(x.size / rate, 2), "elapsed_s": round(time.perf_counter() - t0, 2), "raw": ct}


# 스튜디오 -------------------------------------------------------------------------------------------------------------

def studio_rows(queue: Path, allow_child: bool):
    """(세션 폴더, 학습 제외 이유 또는 None). 학습에 못 쓰는 세션은 GPU 시간을 쓰지 않도록 타임스탬프도 만들지 않는다."""
    for d in sorted(p for p in queue.iterdir() if p.is_dir()) if queue.is_dir() else []:
        mp = d / "meta.json"
        if not mp.is_file():
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        if m.get("status") != "recorded":
            continue
        c = m.get("consent") or {}
        why = training_gate({"consent": c.get("training_tutor") is True and c.get("training_student") is True,
                             "adult": m.get("student_group") == "adult", "child": m.get("student_group") == "child",
                             "guardian_consent": c.get("guardian") is True and bool(c.get("guardian_ref"))}, allow_child)
        yield d, why


def studio_pending(queue: Path, allow_child: bool) -> list[str]:
    return [d.name for d, why in studio_rows(queue, allow_child) if why is None and not (d / "tutor_words.json").is_file()]


def studio_stage(queue: Path, cfg: dict, url: str, allow_child: bool, limit: int | None = None) -> dict:
    done, failed, skipped = [], [], []
    for d, why in studio_rows(queue, allow_child):
        if why is not None:
            skipped.append({"source": d.name, "reason": why})
            continue
        if (d / "tutor_words.json").is_file():
            continue
        if limit is not None and len(done) + len(failed) >= limit:
            break
        try:
            r = word_timestamps(d / "tutor.wav", cfg, url)
        except (TimestampError, requests.RequestException, OSError, ValueError) as e:
            log.error("타임스탬프 실패 %s: %s", d.name, e)
            failed.append({"source": d.name, "error": str(e)[:300]})
            continue
        raw = r.pop("raw", None)
        if raw is not None:
            (d / "tutor_words.raw.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
        words = r.pop("words")
        (d / "tutor_words.meta.json").write_text(json.dumps(
            r | {"model": cfg.get("transcribe_model"), "prompt": cfg.get("words_prompt") or VERBATIM_PROMPT,
                 "words_n": len(words), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False, indent=1), encoding="utf-8")
        # 마지막에 쓴다: 이 파일이 있으면 '끝난 세션' 으로 본다
        (d / "tutor_words.json").write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
        done.append({"source": d.name, "words": len(words)})
    return {"done": done, "failed": failed, "skipped": skipped}


# 대기열 (Gateway 녹음 세션) ------------------------------------------------------------------------------------------------

def queue_pending(queue: Path) -> list[str]:
    out = []
    for d in sorted(p for p in queue.iterdir() if p.is_dir()) if queue.is_dir() else []:
        mp, tp = d / "meta.json", d / "transcript.json"
        if not (mp.is_file() and tp.is_file() and (d / "agent.wav").is_file()):
            continue
        if json.loads(mp.read_text(encoding="utf-8")).get("finetune") != "candidate":
            continue
        if not ((json.loads(tp.read_text(encoding="utf-8")).get("agent") or {}).get("words")):
            out.append(d.name)
    return out


def queue_stage(queue: Path, cfg: dict, url: str, limit: int | None = None) -> dict:
    done, failed = [], []
    for name in queue_pending(queue):
        if limit is not None and len(done) + len(failed) >= limit:
            break
        d = queue / name
        try:
            r = word_timestamps(d / "agent.wav", cfg, url)
        except (TimestampError, requests.RequestException, OSError, ValueError) as e:
            failed.append({"source": name, "error": str(e)[:300]})
            continue
        tp = d / "transcript.json"
        tr = json.loads(tp.read_text(encoding="utf-8"))
        tr.setdefault("agent", {})
        tr["agent"]["words"] = r["words"]
        tr["agent"]["ctc_raw"] = r.get("raw")
        tmp = tp.with_suffix(".tmp")
        tmp.write_text(json.dumps(tr, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(tp)
        done.append({"source": name, "words": len(r["words"])})
    return {"done": done, "failed": failed}


# CLI -------------------------------------------------------------------------------------------------------------------

def paths(cfg_all: dict) -> tuple[dict, Path, Path, bool]:
    sp = dict(cfg_all.get("slow_path") or {})
    st = cfg_all.get("studio") or {}
    tq = rel(st.get("queue_dir") or "queue/tutor")
    eq = rel(sp.get("queue_dir") or cfg_all.get("queue_dir") or "queue/english")
    allow_child = (cfg_all.get("finetune") or {}).get("allow_child_data") is True
    return sp, tq, eq, allow_child


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pending", "studio", "queue"])
    ap.add_argument("--url", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sp, tq, eq, allow_child = paths(load_all())
    url = a.url or sp.get("transcribe_url", "http://127.0.0.1:8002")
    if a.stage == "pending":
        res = {"studio": studio_pending(tq, allow_child), "queue": queue_pending(eq)}
    elif a.stage == "studio":
        res = studio_stage(tq, sp, url, allow_child, a.limit)
    else:
        res = queue_stage(eq, sp, url, a.limit)
    print(json.dumps(res, ensure_ascii=False))
    return 1 if res.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
