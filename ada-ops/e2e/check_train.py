"""전체 시험용 학습 대역 (GPU 없음). moshi-finetune 전용 환경(~/opt/moshi-finetune/.venv)에서 실행한다.

실제 학습은 하지 않지만, 우리가 만든 학습 데이터를 **moshi-finetune 의 실제 데이터 로더와 Interleaver** 로 읽는다
(Mimi 인코더 CPU 실행, SentencePiece 토크나이저, keep_main_only=True). 형식이 어긋나면 여기서 실패한다.
그 뒤 train.py 와 같은 위치·이름으로 LoRA 체크포인트(consolidated/lora.safetensors, config.json)를 쓴다.

  python check_train.py <job 이 만든 config.yaml>
"""

import json
import sys
from pathlib import Path

import torch
import yaml

MF = Path.home() / "opt" / "moshi-finetune"
sys.path.insert(0, str(MF))

from finetune.data.args import DataArgs  # noqa: E402
from finetune.data.data_loader import build_data_loader  # noqa: E402
from finetune.data.interleaver import InterleavedTokenizer, Interleaver  # noqa: E402
from moshi.models import loaders  # noqa: E402

# InterleavedTokenizer 가 .cuda() 를 직접 부르므로 CPU 시험에서는 그대로 둔다
torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[method-assign]


def main():
    # 공유 GPU 를 건드리지 않는다 (run_e2e.py 가 CUDA_VISIBLE_DEVICES="" 로 부름)
    if torch.cuda.is_available():
        raise SystemExit("e2e 학습 대역은 GPU 가 보이면 실행하지 않습니다 (CUDA_VISIBLE_DEVICES=\"\" 필요)")
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
    ci = loaders.CheckpointInfo.from_hf_repo(cfg["moshi_paths"]["hf_repo_id"])
    mimi = ci.get_mimi(device="cpu")
    mimi.eval()
    spm = ci.get_text_tokenizer()
    # train.py 와 같은 값: LMModel.text_padding_token_id=3, end_of_text_padding_id=0, zero_token_id=-1
    inter = Interleaver(spm, mimi.frame_rate, 3, 0, -1, keep_main_only=True, device="cpu")  # 기본값 "cuda"
    tok = InterleavedTokenizer(mimi, inter, duration_sec=float(cfg["duration_sec"]))
    data = DataArgs(train_data=cfg["data"]["train_data"], shuffle=False, eval_data="")
    batch = next(iter(build_data_loader(instruct_tokenizer=tok, args=data, batch_size=1, seed=0, rank=0, world_size=1, is_eval=False)))
    codes = batch.codes
    frames = int(round(float(cfg["duration_sec"]) * mimi.frame_rate))
    text = codes[0, 0]
    words = int(((text != 3) & (text != 0) & (text != -1)).sum())
    print(f"data check: codes {tuple(codes.shape)} (text 1 + audio {codes.shape[1] - 1} 코드북, {codes.shape[2]} 프레임), 단어 토큰 {words}", flush=True)
    assert codes.shape[0] == 1 and codes.shape[2] >= frames - 1, codes.shape
    assert codes.shape[1] == 17, f"Moshi 는 text 1 + audio 16(튜터 8 + 학생 8) 코드북이어야 함: {codes.shape}"
    assert words > 0, "튜터 단어 정렬이 텍스트 스트림에 들어가지 않았습니다 (SPEAKER_MAIN·시간 범위 확인)"
    print("step: 1 - loss: 0.000 (e2e 대역: 실제 학습 아님)", flush=True)
    out = Path(cfg["run_dir"]) / "checkpoints" / "checkpoint_000001" / "consolidated"
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file({"e2e.placeholder": torch.zeros(1)}, str(out / "lora.safetensors"))
    conf = dict(ci.raw_config or {})
    conf.update(lora=True, lora_rank=cfg["lora"]["rank"], lora_scaling=cfg["lora"]["scaling"], e2e_placeholder=True)
    (out / "config.json").write_text(json.dumps(conf), encoding="utf-8")


if __name__ == "__main__":
    main()
