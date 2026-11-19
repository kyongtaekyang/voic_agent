"""학습한 튜터 말투 LoRA 를 실시간 Moshi 에 적용하거나 해제한다. **사람이 평가를 확인한 뒤 직접 실행한다** (학습 직후 자동 적용 없음).

  python -m finetune.promote list                      # 학습 결과 목록
  python -m finetune.promote set <run|latest> --by 이름  # 적용 (finetune/active_lora.json)
  python -m finetune.promote clear                     # 해제 (원래 Moshi)
  python -m finetune.promote show-args                 # en-moshi-bf16.sh 가 moshi.server 에 붙일 인자 (없으면 빈 줄)

적용 범위: **en-solo (Moshi PyTorch BF16) 만.** moshi.server 의 --lora-weight/--config-path 는 BF16 모델에 LoRA 를 합친다.
realtime 의 INT8(moshiko-pytorch-q8, core/moshi_q8.py)은 int8 가중치에 LoRA 를 합칠 수 없어 적용하지 않는다.
적용·해제는 다음에 Moshi 서버가 뜰 때부터 반영된다 (실행 중인 서버는 그대로).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from finetune.job import ft_cfg, latest_checkpoint
from slow_path import words as sw


def runs(f: dict) -> list[dict]:
    out = []
    rd = sw.rel(f["runs_dir"])
    for d in sorted(rd.glob("lora_*")) if rd.is_dir() else []:
        rep = json.loads((d / "report.json").read_text(encoding="utf-8")) if (d / "report.json").is_file() else {}
        ck = latest_checkpoint(d / "train")
        out.append({"run": d.name, "result": rep.get("result"), "checkpoint": ck,
                    "last_loss": ((rep.get("phases") or {}).get("train") or {}).get("last_loss")})
    return out


def active(f: dict) -> dict | None:
    p = sw.rel(f["active_lora"])
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def set_active(f: dict, run: str, by: str) -> dict:
    rs = [r for r in runs(f) if r["checkpoint"]]
    if not rs:
        raise SystemExit("적용할 학습 결과가 없습니다")
    r = rs[-1] if run == "latest" else next((x for x in rs if x["run"] == run), None)
    if r is None:
        raise SystemExit(f"학습 결과가 없거나 체크포인트가 없습니다: {run}")
    ck = r["checkpoint"]
    if not ck.get("config"):
        raise SystemExit(f"{r['run']}: config.json 이 없어 moshi.server --config-path 를 줄 수 없습니다")
    rec = {"run": r["run"], "lora": ck["lora"], "config": ck["config"], "step": ck["step"], "scope": "en-solo (BF16)",
           "promoted_by": by, "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    p = sw.rel(f["active_lora"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def server_args(f: dict) -> list[str]:
    a = active(f)
    if not a:
        return []
    for k in ("lora", "config"):
        if not Path(a[k]).is_file():
            raise SystemExit(f"적용된 LoRA 파일이 없습니다: {a[k]} (finetune.promote clear 로 해제하세요)")
    return ["--lora-weight", a["lora"], "--config-path", a["config"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "set", "clear", "show-args"])
    ap.add_argument("run", nargs="?")
    ap.add_argument("--by", default=None, help="적용한 사람 (기록용)")
    a = ap.parse_args(argv)
    f = ft_cfg(sw.load_all())
    if a.cmd == "list":
        print(json.dumps({"active": active(f), "runs": runs(f)}, ensure_ascii=False, indent=1))
    elif a.cmd == "set":
        if not a.run or not a.by:
            ap.error("set 은 <run|latest> 와 --by 가 필요합니다")
        print(json.dumps(set_active(f, a.run, a.by), ensure_ascii=False, indent=1))
    elif a.cmd == "clear":
        sw.rel(f["active_lora"]).unlink(missing_ok=True)
        print("해제했습니다. 다음에 Moshi 서버가 뜰 때부터 원래 모델을 씁니다")
    else:
        print(" ".join(server_args(f)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
