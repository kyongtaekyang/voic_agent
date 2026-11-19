"""moshi-finetune 학습 데이터 준비 (프롬프트 3절, 4절 finetune/). GPU 를 쓰지 않는다.

  python -m finetune.prepare from-queue [--queue queue/english] [--out finetune/data]     # Slow Path 가 candidate 로 분류한 세션
  python -m finetune.prepare from-pairs MANIFEST.jsonl [--out finetune/data]              # 외부에서 확보한 녹음 (튜터/학생 음성 쌍)
  python -m finetune.prepare from-studio [--queue queue/tutor]                            # 녹음 스튜디오(studio/) 세션
  python -m finetune.prepare check [--out finetune/data]                                  # 만든 데이터 형식 검사

형식 (kyutai-labs/moshi-finetune 2acc879 의 README, annotate.py, finetune/data/interleaver.py 로 확인, finetune/README.md 참고)
- 스테레오 wav. **왼쪽(채널 0) = Moshi 가 낼 음성(튜터), 오른쪽(채널 1) = 사용자(학생)**. 24 kHz 로 쓴다 (Mimi 샘플레이트. 로더가 다시 샘플링하긴 함)
- wav 옆에 같은 이름의 .json: {"alignments": [[단어, [시작초, 끝초], "SPEAKER_MAIN"], ...]}. train.py 가 keep_main_only=True 이므로 튜터 단어만 쓴다
- 목록 jsonl: 한 줄에 {"path": "wav 경로(jsonl 기준 상대)", "duration": 초}

정렬(단어 타임스탬프)은 Whisper 대신 Transcribe 30B 의 단어 타임스탬프를 쓴다 (공식 annotate.py 는 whisper_timestamped 를 쓰지만, 세 md 파일 이외
모델이므로 프롬프트 7절에 따라 쓰지 않음). 입력은 중립 형식 [{"word", "start", "end"}, ...] 이다.
- from-queue: transcript.json 의 agent.words (Slow Path 가 타임스탬프 미들웨어 결과를 저장하도록 바꾼 뒤 생김. 미들웨어 출력 형식 미확인)
- from-pairs: 매니페스트의 tutor_words (json 파일 경로)
정렬이 없는 항목은 만들지 않고 skipped 에 이유를 남긴다 (정렬 없는 학습 데이터는 텍스트 스트림이 비어서 튜터 말투 학습에 쓸모가 없음).

데이터는 서버 안(finetune/data/, 저장소 제외)에만 둔다. 아동 음성은 법정대리인 동의와 내부관리규정 검토 전에는 쓰지 않는다 (프롬프트 3절).
from-queue 는 meta.json 의 finetune 이 candidate 이고 consent.finetune 이 true 인 세션만 읽는다 (needs_review 는 사람이 확인한 뒤 candidate 로 바꿔야 함).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
RATE = 24000
MAIN = "SPEAKER_MAIN"


# 음성 ------------------------------------------------------------------------------------------------------------

def read_mono(p: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(p), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"16비트 PCM 만 받습니다: {p}")
        rate, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, rate


def to_rate(x: np.ndarray, rate: int) -> np.ndarray:
    if rate == RATE:
        return x
    from math import gcd

    from scipy.signal import resample_poly
    g = gcd(rate, RATE)
    return resample_poly(x, RATE // g, rate // g).astype(np.float32)


def write_stereo(p: Path, left: np.ndarray, right: np.ndarray) -> float:
    n = max(left.size, right.size)
    st = np.zeros((n, 2), np.float32)
    st[: left.size, 0] = left
    st[: right.size, 1] = right
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(st, -1, 1) * 32767).astype("<i2").tobytes())
    return n / RATE


# 정렬 ------------------------------------------------------------------------------------------------------------

def alignments_from_words(words: list[dict], duration: float) -> list:
    """[{"word","start","end"}] → moshi-finetune alignments. 빈 단어, 길이 0 이하, 음성 길이 밖은 버린다 (interleaver 가 start<end 만 씀)."""
    out = []
    for w in words:
        text = str(w.get("word", "")).strip()
        s, e = float(w["start"]), float(w["end"])
        if not text or e <= s or s < 0 or s >= duration:
            continue
        out.append([text, [round(s, 3), round(min(e, duration), 3)], MAIN])  # interleaver 가 word.strip() 후 단어별로 토큰화
    return sorted(out, key=lambda a: a[1][0])


def write_item(out: Path, name: str, tutor: np.ndarray, student: np.ndarray, words: list[dict]) -> dict:
    wav = out / "data_stereo" / f"{name}.wav"
    dur = write_stereo(wav, tutor, student)
    al = alignments_from_words(words, dur)
    # interleaver.py 가 open(info_file) 을 encoding 없이 열므로(로캘 의존) ASCII 로 쓴다
    wav.with_suffix(".json").write_text(json.dumps({"alignments": al}, ensure_ascii=True), encoding="ascii")
    return {"path": f"data_stereo/{name}.wav", "duration": round(dur, 3), "_words": len(al)}


def write_manifest(out: Path, items: list[dict], name: str) -> Path:
    p = out / f"{name}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({"path": it["path"], "duration": it["duration"]}) + "\n")
    return p


# 입력 ------------------------------------------------------------------------------------------------------------

def clock_skew(d: Path, meta: dict) -> float | None:
    """user.wav 는 받은 순서대로 이어 쓰고(벽시계 보정 없음) agent.wav 는 벽시계 기준으로 무음을 채운다 (gateway/recorder.py).
    마이크 프레임이 빠지거나 밀리면 두 채널의 시간이 어긋나고, 전이중 학습에서는 그 어긋난 주고받기를 그대로 배운다.
    세션 길이(벽시계, eva.session.ended 의 duration_s) 와 user.wav 길이의 차이(초)를 돌려준다. 알 수 없으면 None."""
    mp = d / "metrics.json"
    dur = json.loads(mp.read_text(encoding="utf-8")).get("duration_s") if mp.is_file() else None
    if dur is None or meta.get("user_s") is None:
        return None
    return abs(float(dur) - float(meta["user_s"]))


def from_queue(queue: Path, out: Path, max_skew_s: float = 0.5) -> dict:
    items, skipped = [], []
    for d in sorted(p for p in queue.iterdir() if p.is_dir()) if queue.is_dir() else []:
        mp = d / "meta.json"
        if not mp.is_file():
            continue
        meta = json.loads(mp.read_text(encoding="utf-8"))
        if meta.get("finetune") != "candidate":
            continue
        if (meta.get("consent") or {}).get("finetune") is not True:  # candidate.py 가 이미 확인하지만 한 번 더
            skipped.append({"source": d.name, "reason": "동의 기록 없음"})
            continue
        tp = d / "transcript.json"
        words = ((json.loads(tp.read_text(encoding="utf-8")) if tp.is_file() else {}).get("agent") or {}).get("words")
        if not words:
            skipped.append({"source": d.name, "reason": "Agent 단어 타임스탬프 없음 (transcript.json agent.words)"})
            continue
        skew = clock_skew(d, meta)
        if skew is None or skew > max_skew_s:
            skipped.append({"source": d.name, "reason": f"두 채널 시간 어긋남 확인 불가 또는 초과 ({skew} > {max_skew_s}초)"})
            continue
        ax, ar = read_mono(d / "agent.wav")
        ux, ur = read_mono(d / "user.wav")
        items.append(write_item(out, f"q_{d.name}", to_rate(ax, ar), to_rate(ux, ur), words) | {"source": d.name})
    return finish(out, "queue", items, skipped)


def training_gate(row: dict, allow_child: bool) -> str | None:
    """학습 데이터로 바꿔도 되는지 (프롬프트 3절). 안 되면 이유를 돌려준다.
    성인: consent(학습 활용 동의) 가 true. 아동: 거기에 guardian_consent 가 true 이고, 내부관리규정 검토가 끝나
    config/services.yaml 의 finetune.allow_child_data 를 true 로 바꾼 뒤에만."""
    if row.get("consent") is not True:
        return "학습 활용 동의(consent) 가 true 가 아님"
    if row.get("adult") is True:
        return None
    if row.get("child") is True:
        if row.get("guardian_consent") is not True:
            return "아동: 법정대리인 동의 없음"
        if not allow_child:
            return "아동: 내부관리규정 검토 전 (finetune.allow_child_data: false)"
        return None
    return "adult 또는 child 표시 없음"


def build_pairs(rows: list[dict], base: Path, out: Path, name: str, allow_child: bool, prefix: str = "p_") -> dict:
    items, skipped = [], []
    for i, row in enumerate(rows):
        src = row.get("id") or Path(row["tutor_wav"]).stem
        why = row.get("_skip") or training_gate(row, allow_child)
        if why:
            skipped.append({"source": src, "reason": why})
            continue
        if not row.get("tutor_words") or not (base / row["tutor_words"]).is_file():
            skipped.append({"source": src, "reason": "tutor_words 없음 (튜터 음성 단어 타임스탬프)"})
            continue
        tx, tr = read_mono(base / row["tutor_wav"])
        sx, sr = read_mono(base / row["student_wav"])
        words = json.loads((base / row["tutor_words"]).read_text(encoding="utf-8"))
        nm = prefix + hashlib.sha1(f"{i}:{src}".encode()).hexdigest()[:12]  # 원래 파일 이름(사람 이름 등)이 데이터 폴더에 남지 않게
        items.append(write_item(out, nm, to_rate(tx, tr), to_rate(sx, sr), words) | {"source": src})
    return finish(out, name, items, skipped)


def from_pairs(manifest: Path, out: Path, allow_child: bool = False) -> dict:
    """매니페스트 jsonl 한 줄: {"id", "tutor_wav", "student_wav", "tutor_words": 단어 json, "consent": true,
    "adult": true | "child": true + "guardian_consent": true}. 경로는 매니페스트 기준 상대 경로도 된다 (training_gate 참고)."""
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    return build_pairs(rows, manifest.parent, out, "pairs", allow_child)


def from_studio(queue: Path, out: Path, allow_child: bool = False, max_drift_s: float = 0.5, max_gap_s: float = 2.0) -> dict:
    """녹음 스튜디오(studio/server.py) 세션 queue/tutor/<세션>/ → 학습 데이터. tutor_words.json(튜터 단어 타임스탬프)은
    Transcribe 30B 타임스탬프 미들웨어 형식을 확인한 뒤 만든다. 그 전에는 모든 세션이 'tutor_words 없음' 으로 건너뛰어진다.
    기기 시계 차이(drift_s)가 크거나 중간 끊김(gap_pad_s)이 길면 두 채널의 주고받기가 어긋나거나 빠지므로 제외한다."""
    rows = []
    for d in sorted(p for p in queue.iterdir() if p.is_dir()) if queue.is_dir() else []:
        mp = d / "meta.json"
        if not mp.is_file():
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        if m.get("status") != "recorded":
            continue
        c = m.get("consent") or {}
        row = {"id": d.name, "tutor_wav": f"{d.name}/tutor.wav", "student_wav": f"{d.name}/student.wav",
               "tutor_words": f"{d.name}/tutor_words.json",
               "consent": c.get("training_tutor") is True and c.get("training_student") is True,
               "adult": m.get("student_group") == "adult", "child": m.get("student_group") == "child",
               "guardian_consent": c.get("guardian") is True and bool(c.get("guardian_ref"))}
        tracks = m.get("tracks") or {}
        bad = [f"{r}: drift {t.get('drift_s')}초, 끊김 {t.get('gap_pad_s')}초" for r, t in tracks.items()
               if abs(t.get("drift_s") or 0) > max_drift_s or (t.get("gap_pad_s") or 0) > max_gap_s]
        if len(tracks) != 2:
            row["_skip"] = "트랙 정보 없음"
        elif bad:
            row["_skip"] = "두 채널 시간 어긋남 또는 끊김 초과 (" + "; ".join(bad) + ")"
        rows.append(row)
    return build_pairs(rows, queue, out, "studio", allow_child, prefix="s_")


def remove_derived(out: Path, source: str) -> int:
    """동의 철회·삭제: *_sources.json 에서 source 로 만든 학습 데이터(wav, 정렬 json)를 지우고 목록 jsonl 에서도 뺀다. 지운 항목 수."""
    removed = 0
    for sp in sorted(out.glob("*_sources.json")) if out.is_dir() else []:
        rows = json.loads(sp.read_text(encoding="utf-8"))
        gone = {r["path"] for r in rows if r["source"] == source}
        if not gone:
            continue
        for p in gone:
            for f in ((out / p), (out / p).with_suffix(".json")):
                f.unlink(missing_ok=True)
        removed += len(gone)
        sp.write_text(json.dumps([r for r in rows if r["path"] not in gone], ensure_ascii=False, indent=1), encoding="utf-8")
        mf = out / (sp.name[: -len("_sources.json")] + ".jsonl")
        if mf.is_file():
            keep = [ln for ln in mf.read_text(encoding="utf-8").splitlines() if ln.strip() and json.loads(ln)["path"] not in gone]
            mf.write_text("".join(ln + "\n" for ln in keep), encoding="utf-8")
    return removed


def finish(out: Path, name: str, items: list[dict], skipped: list[dict]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    m = write_manifest(out, items, name)
    res = {"manifest": str(m), "items": len(items), "hours": round(sum(i["duration"] for i in items) / 3600, 3),
           "words": sum(i["_words"] for i in items), "skipped": skipped}
    # 원본 대응표는 서버 안에만 둔다 (개인정보 삭제 요청 시 찾아서 지우기 위함)
    (out / f"{name}_sources.json").write_text(
        json.dumps([{"path": i["path"], "source": i["source"]} for i in items], ensure_ascii=False, indent=1), encoding="utf-8")
    return res


# 검사 ------------------------------------------------------------------------------------------------------------

def check(out: Path) -> dict:
    """moshi-finetune 이 기대하는 형식인지 확인한다: 스테레오, 길이 일치, 정렬 json, SPEAKER_MAIN, 시간 순서."""
    problems, n = [], 0
    for m in sorted(out.glob("*.jsonl")):
        for line in m.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            wav = out / row["path"]
            if not wav.is_file():
                problems.append(f"{row['path']}: wav 없음")
                continue
            with wave.open(str(wav), "rb") as w:
                ch, rate, frames = w.getnchannels(), w.getframerate(), w.getnframes()
            if ch != 2:
                problems.append(f"{row['path']}: 채널 {ch} (2 여야 함)")
            if abs(frames / rate - row["duration"]) > 0.05:
                problems.append(f"{row['path']}: duration {row['duration']} ≠ 실제 {frames / rate:.3f}")
            js = wav.with_suffix(".json")
            if not js.is_file():
                problems.append(f"{row['path']}: 정렬 json 없음")
                continue
            al = json.loads(js.read_text(encoding="utf-8")).get("alignments")
            if not al:
                problems.append(f"{row['path']}: alignments 비어 있음")
                continue
            starts = [a[1][0] for a in al]
            if starts != sorted(starts):
                problems.append(f"{row['path']}: 정렬 시간 순서가 아님")
            if any(a[2] != MAIN for a in al):
                problems.append(f"{row['path']}: {MAIN} 이외 화자 (train.py keep_main_only=True 라서 버려짐)")
            if any(a[1][1] > frames / rate + 0.01 for a in al):
                problems.append(f"{row['path']}: 음성 길이를 넘는 정렬")
    return {"items": n, "ok": not problems, "problems": problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["from-queue", "from-pairs", "from-studio", "check"])
    ap.add_argument("manifest", nargs="?")
    ap.add_argument("--queue", default=None, help="from-queue: queue/english, from-studio: queue/tutor")
    ap.add_argument("--out", default=str(ROOT / "finetune" / "data"))
    ap.add_argument("--max-skew-s", type=float, default=0.5, help="from-queue: 세션 길이와 user.wav 길이 차이 허용치")
    a = ap.parse_args(argv)
    out = Path(a.out)
    import os
    cfg = yaml.safe_load(Path(os.environ.get("EVA_CONFIG") or ROOT / "config" / "services.yaml").read_text(encoding="utf-8"))
    allow_child = (cfg.get("finetune") or {}).get("allow_child_data", False) is True
    if a.stage == "from-queue":
        res = from_queue(Path(a.queue or ROOT / "queue" / "english"), out, a.max_skew_s)
    elif a.stage == "from-studio":
        res = from_studio(Path(a.queue or ROOT / "queue" / "tutor"), out, allow_child)
    elif a.stage == "from-pairs":
        if not a.manifest:
            ap.error("from-pairs 는 MANIFEST.jsonl 이 필요합니다")
        res = from_pairs(Path(a.manifest), out, allow_child)
    else:
        res = check(out)
    print(json.dumps(res, ensure_ascii=False))
    return 0 if res.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
