"""Slow Path 배치 실행 (프롬프트 2.2절). 후처리 모드에서 ada-ops/nightly-batch 가 단계별로 부른다.

  python -m slow_path.run transcribe [--url http://127.0.0.1:8002]   # Transcribe 30B 가 GPU 에 있을 때
  python -m slow_path.run summarize  [--url http://127.0.0.1:8001]   # Transcribe 를 내리고 Gemma 4 를 올린 뒤
  python -m slow_path.run status                                     # 대기열 상태

단계마다 queue/korean/<세션>/ 에 결과를 쓰고 meta.json 의 status 를 바꾼다 (queued → transcribed → summarized).
실패한 세션은 status 를 그대로 두고 meta.json 의 errors 에 남긴다. 다음 배치에서 다시 시도한다.
결과 파일
- transcript.json : Transcribe 30B 축어 전사 (사용자 음성 user.wav)
- metrics.json    : 턴 수, 사용자 발화 비율, 끼어들기, 응답 지연, 실시간 STT 의 CER (축어 전사 기준)
- summary.json    : Gemma 4 세션 요약과 학습 피드백
음성과 결과는 서버 밖으로 보내지 않는다 (url 은 127.0.0.1 만 허용).
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml

from slow_path.cer import cer

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("kva.slow_path")


def load_cfg() -> dict:
    cfg = yaml.safe_load((ROOT / "config" / "services.yaml").read_text(encoding="utf-8"))
    return cfg.get("slow_path", {}) | {"_ports": cfg["ports"], "_llm": cfg["llm"]}


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
            continue  # 아직 녹음 중
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if status is None or meta.get("status") == status:
            yield d, meta


def save_meta(d: Path, meta: dict) -> None:
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def read_events(d: Path) -> list[dict]:
    p = d / "events.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()] if p.is_file() else []


# 1단계: 축어 전사 --------------------------------------------------------------------------------------------------

def chat(url: str, model: str, messages: list, *, max_tokens: int, temperature: float, timeout: float, extra: dict | None = None) -> str:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, **(extra or {})}
    with requests.Session() as s:
        s.trust_env = False  # HTTP(S)_PROXY 환경 변수가 있어도 프록시를 거치지 않는다 (음성 외부 전송 방지)
        r = s.post(f"{url}/v1/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def transcribe_one(d: Path, cfg: dict, url: str) -> dict:
    wav = d / "user.wav"
    audio = base64.b64encode(wav.read_bytes()).decode()
    prompt = cfg.get("transcribe_prompt", "Transcribe the following audio.")
    terms = cfg.get("context_terms") or []
    if terms:
        prompt += " Context terms: " + ", ".join(terms) + "."
    t0 = time.perf_counter()
    # 시스템 프롬프트는 비우고, temperature 0, Thinking 끔 (프롬프트 2.2절 2번, docs/Nemotron_3.5_Transcribe.md 사용법)
    text = chat(url, cfg.get("transcribe_model", "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"),
                [{"role": "user", "content": [{"type": "text", "text": prompt},
                                              {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio}"}}]}],
                max_tokens=int(cfg.get("transcribe_max_tokens", 8192)), temperature=0.0,
                timeout=float(cfg.get("timeout_s", 1800)),
                extra={"chat_template_kwargs": {"enable_thinking": False}, "skip_special_tokens": False})
    return {"text": text.strip(), "prompt": prompt, "elapsed_s": round(time.perf_counter() - t0, 2)}


# 지표 ------------------------------------------------------------------------------------------------------------

def session_metrics(events: list[dict], meta: dict, transcript: str | None) -> dict:
    starts = {e.get("item_id"): e for e in events if e["type"] == "input_audio_buffer.speech_started"}
    stops = [e for e in events if e["type"] == "input_audio_buffer.speech_stopped"]
    user_ms = sum(max(0, (e.get("audio_end_ms") or 0) - (starts.get(e.get("item_id"), {}).get("audio_start_ms") or 0))
                  for e in stops if e.get("item_id") in starts)
    # 응답 지연: 사용자 발화 끝(speech_stopped) → 다음 첫 응답 음성 (Gateway 가 받은 시각 기준, 브라우저 재생 지연 제외)
    lat, last_stop = [], None
    for e in events:
        if e["type"] == "input_audio_buffer.speech_stopped":
            last_stop = e["t"]
        elif e["type"] == "rec.agent_audio_start" and last_stop is not None:
            lat.append(round(1000 * (e["t"] - last_stop)))
            last_stop = None
    realtime_stt = " ".join(e.get("transcript", "") for e in events
                            if e["type"] == "conversation.item.input_audio_transcription.completed")
    kva = {}
    for e in events:
        if e["type"].startswith("kva."):
            kva[e["type"]] = kva.get(e["type"], 0) + 1
    user_s, agent_s = user_ms / 1000, float(meta.get("agent_s", 0))
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else None  # noqa: E731
    out = {
        "turns": len(stops),
        "responses": sum(1 for e in events if e["type"] == "response.done"),
        "barge_ins": sum(1 for e in events if e["type"] == "conversation.item.truncate"),
        "user_speech_s": round(user_s, 2),
        "agent_audio_s": round(agent_s, 2),
        "user_speech_ratio": round(user_s / (user_s + agent_s), 3) if user_s + agent_s else None,
        "response_latency_ms": {"n": len(lat), "p50": q(lat, 0.5), "p95": q(lat, 0.95),
                                "mean": round(statistics.mean(lat)) if lat else None},
        "kva_events": kva,
    }
    if transcript is not None:
        out["realtime_stt_cer"] = cer(transcript, realtime_stt)
    return out


def dialogue(events: list[dict]) -> list[dict]:
    rows = []
    for e in events:
        if e["type"] == "conversation.item.input_audio_transcription.completed" and e.get("transcript"):
            rows.append({"role": "child", "text": e["transcript"]})
        elif e["type"] == "response.done" and e.get("text"):
            rows.append({"role": "tutor", "text": e["text"]})
    return rows


# 2단계: 요약 ------------------------------------------------------------------------------------------------------

def summarize_one(d: Path, cfg: dict, url: str) -> dict:
    events = read_events(d)
    rows = dialogue(events)
    if not rows:
        return {"skipped": "대화 기록 없음"}
    system = (ROOT / cfg.get("summary_prompt_file", "config/prompts/session_summary.md")).read_text(encoding="utf-8")
    lines = "\n".join(f"{'아이' if r['role'] == 'child' else '선생님'}: {r['text']}" for r in rows)
    tp = d / "transcript.json"
    if tp.is_file():
        lines += "\n\n[아이 발화 축어 전사 (참고)]\n" + json.loads(tp.read_text(encoding="utf-8"))["text"]
    t0 = time.perf_counter()
    raw = chat(url, cfg.get("summary_model") or cfg["_llm"].get("served_name", "gemma-4"),
               [{"role": "system", "content": system}, {"role": "user", "content": lines}],
               max_tokens=int(cfg.get("summary_max_tokens", 800)), temperature=0.2, timeout=float(cfg.get("timeout_s", 1800)),
               extra={"chat_template_kwargs": {"enable_thinking": False}})
    try:
        body = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except ValueError:
        body = {"raw": raw}
    return {"summary": body, "turns_used": len(rows), "elapsed_s": round(time.perf_counter() - t0, 2)}


# 실행 ------------------------------------------------------------------------------------------------------------

def run_stage(stage: str, queue: Path, cfg: dict, url: str, limit: int | None) -> dict:
    want = {"transcribe": "queued", "summarize": "transcribed"}[stage]
    done = failed = 0
    for d, meta in sessions(queue, want):
        if limit is not None and done + failed >= limit:
            break
        try:
            if stage == "transcribe":
                tr = transcribe_one(d, cfg, url)
                (d / "transcript.json").write_text(json.dumps(tr, ensure_ascii=False, indent=1), encoding="utf-8")
                m = session_metrics(read_events(d), meta, tr["text"])
                (d / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
                meta["status"] = "transcribed"
            else:
                sm = summarize_one(d, cfg, url)
                (d / "summary.json").write_text(json.dumps(sm, ensure_ascii=False, indent=1), encoding="utf-8")
                meta["status"] = "summarized"
            meta.setdefault("history", []).append({"stage": stage, "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            done += 1
        except Exception as e:  # noqa: BLE001
            log.warning("%s 실패 %s: %s", stage, d.name, e)
            meta.setdefault("errors", []).append({"stage": stage, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                  "error": f"{type(e).__name__}: {e}"[:300]})
            failed += 1
        save_meta(d, meta)
    return {"stage": stage, "done": done, "failed": failed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["transcribe", "summarize", "status"])
    ap.add_argument("--queue", default=None)
    ap.add_argument("--url", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_cfg()
    queue = Path(a.queue or ROOT / cfg.get("queue_dir", "queue/korean"))
    if a.stage == "status":
        counts: dict[str, int] = {}
        for _, meta in sessions(queue):
            counts[meta.get("status", "?")] = counts.get(meta.get("status", "?"), 0) + 1
        print(json.dumps({"queue": str(queue), "sessions": counts}, ensure_ascii=False))
        return 0
    port = cfg["_ports"]["transcribe_vllm" if a.stage == "transcribe" else "gemma_vllm"]
    url = local_only(a.url or f"http://127.0.0.1:{port}")
    res = run_stage(a.stage, queue, cfg, url, a.limit)
    print(json.dumps(res, ensure_ascii=False))
    return 0 if res["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
