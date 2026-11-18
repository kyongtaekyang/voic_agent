"""영어 Agent Slow Path 배치 (프롬프트 5절 6단계). 후처리 모드에서 ada-ops/nightly-batch 가 부른다.

  python -m slow_path.run transcribe [--url http://127.0.0.1:8002]   # Transcribe 30B 가 GPU 에 있을 때
  python -m slow_path.run classify                                   # 파인튜닝 후보 다시 분류 (GPU 불필요)
  python -m slow_path.run report                                     # 전사된 세션 지표를 모아 queue/english/_kpi.json
  python -m slow_path.run status                                     # 대기열 상태

queue/english/<세션>/ (gateway/recorder.py) 를 읽고, 결과를 같은 폴더에 쓴 뒤 meta.json 의 status 를 바꾼다 (queued → transcribed).
실패한 세션은 status 를 그대로 두고 meta.json 의 errors 에 남긴다. 다음 배치에서 다시 시도한다.
결과 파일
- transcript.json : Transcribe 30B 축어 전사. user (user.wav), agent (agent.wav, slow_path.audit_agent_audio 가 true 일 때)
- metrics.json    : ASR Streaming 자막 WER (축어 전사 기준), 안전 차단 지표, 사후 안전 검사, 언어 안내, 응답 지연 근사치,
                    끼어들기·사용자 발화 비율 근사치 (turn_taking)
- candidate.json  : 파인튜닝 후보 분류와 개인정보 검사 (slow_path/candidate.py). meta.json 의 finetune 에도 결과를 쓴다
음성과 결과는 서버 밖으로 보내지 않는다 (url 은 127.0.0.1 만 허용, 프록시 환경 변수 무시).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import re
import statistics
import sys
import time
import wave
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import yaml
from scipy.signal import resample_poly

from core.safety_monitor import SafetyMonitor
from slow_path.candidate import PiiScanner, classify
from slow_path.turns import turn_taking
from slow_path.wer import wer

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("eva.slow_path")
MODEL_RATE = 16000  # Transcribe 30B 입력 규격 (docs/Nemotron_3.5_Transcribe.md: 16kHz 모노)


def load_cfg() -> dict:
    cfg = yaml.safe_load((ROOT / "config" / "services.yaml").read_text(encoding="utf-8"))
    sp = dict(cfg.get("slow_path") or {})
    sp.setdefault("queue_dir", cfg.get("queue_dir", "queue/english"))
    return sp


def local_only(url: str) -> str:
    host = urlparse(url).hostname
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit(f"외부 주소로는 보내지 않습니다: {url}")
    return url.rstrip("/")


# 대기열 -----------------------------------------------------------------------------------------------------------

def sessions(queue: Path, status: str | None = None):
    for d in sorted(p for p in queue.iterdir() if p.is_dir()) if queue.is_dir() else []:
        meta_p = d / "meta.json"
        if not meta_p.is_file():
            continue  # 아직 녹음 중 (Recorder.close() 가 meta.json 을 마지막에 쓴다)
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if status is None or meta.get("status") == status:
            yield d, meta


def save_meta(d: Path, meta: dict) -> None:
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def read_events(d: Path) -> list[dict]:
    p = d / "events.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()] if p.is_file() else []


# 음성 ------------------------------------------------------------------------------------------------------------

def read_wav(p: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(p), "rb") as w:
        rate, n = w.getframerate(), w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
    return x, rate


def wav16k_bytes(x: np.ndarray, rate: int) -> bytes:
    if rate != MODEL_RATE:
        from math import gcd
        g = gcd(rate, MODEL_RATE)
        x = resample_poly(x, MODEL_RATE // g, rate // g).astype(np.float32)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(MODEL_RATE)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def speech_segments(x: np.ndarray, rate: int, *, frame_ms: int = 20, rms: float = 0.01,
                    min_silence_ms: int = 500, min_speech_ms: int = 100) -> list[tuple[float, float]]:
    """에너지 기준 발화 구간 (초). min_silence_ms 보다 짧은 침묵은 이어 붙이고, min_speech_ms 보다 짧은 구간은 버린다."""
    hop = max(1, rate * frame_ms // 1000)
    n = x.size // hop
    if n == 0:
        return []
    e = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)) >= rms
    segs, start = [], None
    for i, on in enumerate(e):
        if on and start is None:
            start = i
        elif not on and start is not None:
            segs.append([start, i])
            start = None
    if start is not None:
        segs.append([start, n])
    gap = min_silence_ms // frame_ms
    merged: list[list[int]] = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    f = frame_ms / 1000
    return [(a * f, b * f) for a, b in merged if (b - a) * frame_ms >= min_speech_ms]


def response_latency(user: list[tuple[float, float]], agent: list[tuple[float, float]], max_gap_s: float = 5.0) -> list[int]:
    """사용자 발화 끝 → 그 뒤 처음 시작하는 Agent 음성 (ms). **근사치:** user.wav 는 받은 순서대로 이어 쓰고(벽시계 보정 없음)
    agent.wav 는 벽시계 기준으로 무음을 채우므로 시간이 조금씩 어긋날 수 있다. 전이중이라 맞장구(backchannel)도 응답으로 센다.
    max_gap_s 안에 응답이 없거나, Agent 가 사용자 발화 중에 이미 말하고 있었으면(겹침) 세지 않는다."""
    out = []
    for _, u_end in user:
        talking = any(a0 < u_end < a1 for a0, a1 in agent)
        if talking:
            continue
        nxt = next((a0 for a0, _ in agent if a0 >= u_end), None)
        if nxt is not None and nxt - u_end <= max_gap_s:
            out.append(round(1000 * (nxt - u_end)))
    return out


def q(xs: list, p: float):
    return sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else None


# Transcribe 30B ---------------------------------------------------------------------------------------------------

def chat(url: str, model: str, messages: list, *, max_tokens: int, timeout: float, extra: dict | None = None) -> tuple[str, str | None]:
    """(내용, finish_reason). finish_reason 이 "length" 면 max_tokens 나 문맥 길이에서 잘린 것이다."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, **(extra or {})}
    with requests.Session() as s:
        s.trust_env = False  # HTTP(S)_PROXY 가 있어도 프록시를 거치지 않는다 (음성 외부 전송 방지)
        r = s.post(f"{url}/v1/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    return ch["message"]["content"] or "", ch.get("finish_reason")


def transcribe_wav(p: Path, cfg: dict, url: str) -> dict:
    x, rate = read_wav(p)
    if x.size == 0 or float(np.max(np.abs(x))) < 1e-4:
        return {"text": "", "audio_s": round(x.size / max(rate, 1), 2), "skipped": "무음"}
    audio = base64.b64encode(wav16k_bytes(x, rate)).decode()
    prompt = cfg.get("transcribe_prompt", "Transcribe the following audio.")
    terms = cfg.get("context_terms") or []
    if terms:
        prompt += " Context terms: " + ", ".join(terms) + "."
    t0 = time.perf_counter()
    # 시스템 프롬프트 없음, temperature 0, Thinking 끔 (docs/Nemotron_3.5_Transcribe.md, 한국어 Slow Path 와 같은 호출)
    text, finish = chat(url, cfg.get("transcribe_model", "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"),
                [{"role": "user", "content": [{"type": "text", "text": prompt},
                                              {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio}"}}]}],
                max_tokens=int(cfg.get("transcribe_max_tokens", 8192)), timeout=float(cfg.get("timeout_s", 1800)),
                extra={"chat_template_kwargs": {"enable_thinking": False}, "skip_special_tokens": False})
    # 잘려도 오류가 나지 않으므로 표시해 둔다. 잘린 전사로 계산한 WER 은 삭제 오류로 커지고, 사후 검사는 뒷부분을 보지 못한다
    if finish == "length":
        log.warning("전사가 max_tokens 에서 잘림: %s (%.0f초)", p, x.size / rate)
    return {"text": text.strip(), "prompt": prompt, "audio_s": round(x.size / rate, 2), "finish_reason": finish,
            "truncated": finish == "length", "elapsed_s": round(time.perf_counter() - t0, 2)}


# 지표 ------------------------------------------------------------------------------------------------------------

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def safety_audit(text: str, monitor: SafetyMonitor) -> list[dict]:
    """Agent 음성 전사를 문장 단위로 다시 검사한다. 실시간 감시를 통과해 실제로 소리로 나간 위반을 찾는 용도.

    SafetyMonitor 는 토큰마다 끝부분(window, 240자)만 다시 검사하므로, 긴 문장을 check_text() 로 한 번에 넣으면 앞부분을 놓친다.
    그래서 Gateway 처럼 단어를 하나씩 feed() 한다 (문장부호 없는 긴 전사도 검사됨).
    """
    found = []
    for s in (x.strip() for x in _SENT.split(text or "")):
        if not s:
            continue
        monitor.reset()
        for w in s.split():
            monitor.feed(" " + w)
        monitor.idle()
        for v in monitor.violations:
            found.append({"rule": v.rule, "category": v.category, "matched": v.matched.strip(), "sentence": s[:200]})
    return found


def session_metrics(d: Path, events: list[dict], meta: dict, transcript: dict | None, vad: dict,
                    monitor: SafetyMonitor | None = None) -> dict:
    caps = [e for e in events if e["type"] == "eva.user_text" and e.get("text")]
    non_en = [e for e in caps if e.get("lang") not in ("en", None)]
    blocks = [e for e in events if e["type"] == "eva.safety_block"]
    played = [int(e.get("already_played_ms") or 0) for e in blocks]
    agent_caption = "".join(e.get("delta", "") for e in events if e["type"] == "eva.agent_text")
    ended = next((e for e in reversed(events) if e["type"] == "eva.session.ended"), {})

    vad_kw = {k: vad[k] for k in ("frame_ms", "rms", "min_silence_ms", "min_speech_ms") if k in vad}
    lat: list[int] = []
    tt = None
    if (d / "user.wav").is_file() and (d / "agent.wav").is_file():
        ux, ur = read_wav(d / "user.wav")
        ax, ar = read_wav(d / "agent.wav")
        us, ags = speech_segments(ux, ur, **vad_kw), speech_segments(ax, ar, **vad_kw)
        lat = response_latency(us, ags, float(vad.get("max_gap_s", 5.0)))
        tt = turn_taking(us, ags, **{k: int(vad[k]) for k in ("yield_ms", "cut_min_ms") if k in vad})

    out = {
        "duration_s": ended.get("duration_s"),
        "user_s": meta.get("user_s"),
        "agent_audio_s": meta.get("agent_audio_s"),
        "text_tokens": ended.get("text_tokens"),
        "asr_turns": len(caps),
        "non_english_turns": len(non_en),  # 한국어 등 비영어 발화는 WER 을 키운다 (자막·전사 모두 포함)
        "language_mismatch": sum(1 for e in events if e["type"] == "eva.language_mismatch"),
        "safety": {
            "blocks": len(blocks),
            "by_category": {c: sum(1 for e in blocks if e.get("category") == c) for c in sorted({e.get("category") for e in blocks})},
            "already_played_ms": {"n": len(played), "p50": q(played, 0.5), "max": max(played) if played else None},
            "dropped_ms_total": sum(int(e.get("dropped_ms") or 0) for e in blocks),
        },
        "response_latency_ms_approx": {"n": len(lat), "p50": q(lat, 0.5), "p95": q(lat, 0.95),
                                       "mean": round(statistics.mean(lat)) if lat else None,
                                       "note": "음성 에너지 기준 근사치. KPI(300ms) 판정에는 실시간 측정값을 쓴다"},
        "turn_taking": tt,  # 끼어들기, 사용자 발화 비율 (slow_path/turns.py, 근사치)
        "errors":sum(1 for e in events if e["type"] == "eva.error"),
    }
    if transcript is not None:
        hyp = " ".join(e["text"] for e in caps)
        out["asr_wer"] = wer(transcript["user"]["text"], hyp)
        agent_tr = transcript.get("agent")
        out["transcript_truncated"] = any(bool((t or {}).get("truncated")) for t in (transcript["user"], agent_tr))
        if agent_tr is not None and monitor is not None:
            found = safety_audit(agent_tr["text"], monitor)
            out["safety"]["audit"] = {"violations_in_agent_audio": len(found), "items": found}
            # 차단한 구간은 자막에서 빠지므로 전사와 다를 수 있다. 크게 다르면 텍스트 토큰과 음성이 어긋난다는 뜻
            out["agent_caption_vs_audio_wer"] = wer(agent_tr["text"], agent_caption)
    return out


# 실행 ------------------------------------------------------------------------------------------------------------

def classify_session(d: Path, meta: dict, metrics: dict, transcript: dict | None, cfg: dict, scanner: PiiScanner) -> dict:
    """파인튜닝 후보 분류 결과를 candidate.json 과 meta["finetune"] 에 쓴다 (meta 저장은 호출한 쪽에서)."""
    c = classify(meta, metrics, transcript, cfg.get("finetune_candidate"), scanner)
    c["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (d / "candidate.json").write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
    meta["finetune"] = c["decision"]
    return c


def transcribe_stage(queue: Path, cfg: dict, url: str, limit: int | None) -> dict:
    done = failed = 0
    monitor = SafetyMonitor()
    scanner = PiiScanner()
    for d, meta in sessions(queue, "queued"):
        if limit is not None and done + failed >= limit:
            break
        try:
            tr = {"user": transcribe_wav(d / "user.wav", cfg, url)}
            if cfg.get("audit_agent_audio", True) and (d / "agent.wav").is_file():
                tr["agent"] = transcribe_wav(d / "agent.wav", cfg, url)
            (d / "transcript.json").write_text(json.dumps(tr, ensure_ascii=False, indent=1), encoding="utf-8")
            m = session_metrics(d, read_events(d), meta, tr, cfg.get("vad") or {}, monitor)
            (d / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
            classify_session(d, meta, m, tr, cfg, scanner)
            meta["status"] = "transcribed"
            meta.setdefault("history", []).append({"stage": "transcribe", "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            done += 1
        except Exception as e:  # noqa: BLE001
            log.warning("transcribe 실패 %s: %s", d.name, e)
            meta.setdefault("errors", []).append({"stage": "transcribe", "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                  "error": f"{type(e).__name__}: {e}"[:300]})
            failed += 1
        save_meta(d, meta)
    return {"stage": "transcribe", "done": done, "failed": failed}


def classify_stage(queue: Path, cfg: dict) -> dict:
    """전사된 세션을 GPU 없이 다시 분류한다 (기준값이나 개인정보 규칙을 바꾼 뒤, 동의 기록을 붙인 뒤)."""
    scanner = PiiScanner()
    counts: dict[str, int] = {}
    for d, meta in sessions(queue, "transcribed"):
        mp, tp = d / "metrics.json", d / "transcript.json"
        if not mp.is_file():
            continue
        tr = json.loads(tp.read_text(encoding="utf-8")) if tp.is_file() else None
        c = classify_session(d, meta, json.loads(mp.read_text(encoding="utf-8")), tr, cfg, scanner)
        save_meta(d, meta)
        counts[c["decision"]] = counts.get(c["decision"], 0) + 1
    return {"stage": "classify", "decisions": counts}


def report(queue: Path) -> dict:
    """전사된 세션 지표를 모은다 (KPI: WER, 차단 수, 차단 전에 나간 음성, 사후 검사에서 찾은 위반, 끼어들기, 사용자 발화 비율).
    파인튜닝 후보 목록은 queue/english/_finetune_candidates.json (세션 이름과 이유만, 개인정보 내용은 적지 않음)."""
    ms, ft = [], {"candidate": [], "needs_review": [], "rejected": []}
    for d, meta in sessions(queue, "transcribed"):
        p = d / "metrics.json"
        if p.is_file():
            ms.append(json.loads(p.read_text(encoding="utf-8")))
        cp = d / "candidate.json"
        if cp.is_file():
            c = json.loads(cp.read_text(encoding="utf-8"))
            ft.setdefault(c["decision"], []).append(
                {"session": d.name, "reasons": c["reasons"], "pii_rules": sorted({f["rule"] for f in c["pii"]})})
    tts = [m["turn_taking"] for m in ms if m.get("turn_taking")]
    barge = sum(t["barge_ins"] for t in tts)
    ratios = [t["user_speech_ratio"] for t in tts if t["user_speech_ratio"] is not None]
    user_turns = sum(t["user_turns"] for t in tts)
    errs = sum(m["asr_wer"]["errors"] for m in ms if m.get("asr_wer"))
    ref = sum(m["asr_wer"]["ref_words"] for m in ms if m.get("asr_wer"))
    played = [m["safety"]["already_played_ms"]["max"] for m in ms if m["safety"]["already_played_ms"]["max"] is not None]
    lat = [m["response_latency_ms_approx"]["p50"] for m in ms if m["response_latency_ms_approx"]["p50"] is not None]
    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sessions": len(ms),
        "sessions_truncated": sum(1 for m in ms if m.get("transcript_truncated")),  # 0 이 아니면 WER·사후 검사 값을 믿지 말 것
        "asr_wer_pooled": round(errs / ref, 4) if ref else None,
        "asr_ref_words": ref,
        "non_english_turns": sum(m["non_english_turns"] for m in ms),
        "safety_blocks": sum(m["safety"]["blocks"] for m in ms),
        "sessions_with_block": sum(1 for m in ms if m["safety"]["blocks"]),
        "already_played_ms_max": max(played) if played else None,
        "audit_violations_in_agent_audio": sum(m["safety"].get("audit", {}).get("violations_in_agent_audio", 0) for m in ms),
        "session_latency_p50_ms_approx_median": q(lat, 0.5),
        # 아래는 음성 에너지 기준 근사치 (slow_path/turns.py)
        "barge_ins": barge,
        "barge_in_yield_rate": round(sum(t["barge_ins_yielded"] for t in tts) / barge, 3) if barge else None,
        "agent_cuts_user_rate": round(sum(t["agent_cuts_user"] for t in tts) / user_turns, 3) if user_turns else None,
        "user_speech_ratio_median": q(ratios, 0.5),
        "finetune": {k: len(v) for k, v in ft.items()},
    }
    (queue / "_kpi.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (queue / "_finetune_candidates.json").write_text(json.dumps(ft, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["transcribe", "classify", "report", "status"])
    ap.add_argument("--queue", default=None)
    ap.add_argument("--url", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_cfg()
    queue = Path(a.queue or ROOT / cfg["queue_dir"])
    if a.stage == "status":
        counts: dict[str, int] = {}
        for _, meta in sessions(queue):
            counts[meta.get("status", "?")] = counts.get(meta.get("status", "?"), 0) + 1
        print(json.dumps({"queue": str(queue), "sessions": counts}, ensure_ascii=False))
        return 0
    if a.stage == "report":
        if not queue.is_dir():
            print(json.dumps({"queue": str(queue), "sessions": 0}, ensure_ascii=False))
            return 0
        print(json.dumps(report(queue), ensure_ascii=False))
        return 0
    if a.stage == "classify":
        print(json.dumps(classify_stage(queue, cfg) if queue.is_dir() else {"stage": "classify", "decisions": {}},
                         ensure_ascii=False))
        return 0
    url =local_only(a.url or cfg.get("transcribe_url", "http://127.0.0.1:8002"))
    res = transcribe_stage(queue, cfg, url, a.limit)
    print(json.dumps(res, ensure_ascii=False))
    return 0 if res["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
