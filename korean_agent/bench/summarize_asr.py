"""asr_cpu_rtf_t*.json 을 조건(mode)별로 요약한다. RTF 는 음성 길이 가중 평균(총 처리 시간 / 총 음성 길이)."""
import json
import sys
from collections import defaultdict

for path in sys.argv[1:]:
    d = json.load(open(path, encoding="utf-8"))
    agg = defaultdict(lambda: {"proc": 0.0, "dur": 0.0, "n": 0, "err": 0, "lat": None, "pcm": [], "max_rtf": 0.0})
    for r in d["rows"]:
        a = agg[r["mode"]]
        if "error" in r:
            a["err"] += 1
            continue
        a["proc"] += r["proc_s"]; a["dur"] += r["dur_s"]; a["n"] += 1
        a["max_rtf"] = max(a["max_rtf"], r["rtf"])
        a["lat"] = r.get("chunk_latency_ms")
        if "per_chunk_ms" in r:
            a["pcm"].append(r["per_chunk_ms"])
    print(f"== {path}  threads={d['threads']}  load_s={d['load_s']}")
    print(f"{'mode':12s} {'chunk_ms':>8s} {'RTF(w)':>7s} {'RTFmax':>7s} {'ms/chunk':>9s} {'n':>3s} {'err':>3s}")
    for m, a in agg.items():
        rtf = a["proc"] / a["dur"] if a["dur"] else float("nan")
        pcm = sum(a["pcm"]) / len(a["pcm"]) if a["pcm"] else float("nan")
        print(f"{m:12s} {str(a['lat'] or '-'):>8s} {rtf:7.3f} {a['max_rtf']:7.3f} {pcm:9.1f} {a['n']:3d} {a['err']:3d}")
