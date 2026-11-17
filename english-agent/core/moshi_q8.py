"""Moshi PyTorch INT8 (kyutai/moshiko-pytorch-q8) 실행 래퍼.

moshi 0.2.13 의 loaders.get_moshi_lm 은 safetensors 의 부동소수 텐서를 모두 dtype(bfloat16)으로 바꾼 뒤 불러온다.
이때 int8 행렬의 행별 스케일 `*.weight_scb` (파일에는 float32)까지 bfloat16 이 되어,
QLinear.forward 가 첫 단계에서 "Expected `weight_scb` to have type float, but got bfloat16" 로 멈춘다.
장치와 무관한 문제라 GPU 에서도 같다 (2026-11-17 CPU 에서 확인, 0.2.13 이 PyPI 최신).

여기서는 패키지를 고치지 않고 CheckpointInfo.get_moshi 를 감싸서, 불러온 직후 `*.weight_scb` 를 파일의 float32 값으로
다시 넣는다 (bfloat16 을 float 로 되돌리는 것이 아니라 원본을 다시 읽으므로 정밀도 손실이 없다).

사용: python -m core.moshi_q8 [moshi.server 인자...]   (ada-ops/services/en-moshi.sh)
"""
from __future__ import annotations

import sys

import torch
from torch import nn


def fix_q8_scales(lm: nn.Module, weights_path) -> int:
    """lm 안의 QLinear.weight_scb 를 weights_path(safetensors)의 float32 값으로 바꾼다. QLinear 개수를 돌려준다."""
    from safetensors import safe_open
    from moshi.utils.quantize import QLinear

    # 파일 키를 직접 모듈에 대응시키지 않고 load_state_dict 를 다시 거친다. depformer 어텐션은 load pre-hook 에서
    # out_proj.weight_scb → out_projs.{i}.weight_scb 처럼 단계별로 나누기 때문이다 (moshi/modules/transformer.py _load_hook).
    # assign=True 는 넣는 텐서의 dtype(float32)을 그대로 쓴다.
    device = next(lm.parameters()).device
    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        scb = {k: f.get_tensor(k).to(torch.float32).to(device) for k in f.keys() if k.endswith("_scb")}
    # 키 모양: "...linear1.weight_scb", "...self_attn.in_proj_weight_scb" (점 없음) 등. 모두 _scb 로 끝난다.
    res = lm.load_state_dict(scb, strict=False, assign=True)
    if res.unexpected_keys:
        raise RuntimeError(f"모델에 없는 weight_scb: {res.unexpected_keys[:3]}")
    fixed = sum(isinstance(m, QLinear) for m in lm.modules())
    left = [n for n, m in lm.named_modules() if isinstance(m, QLinear) and m.weight_scb.dtype != torch.float32]
    if left:
        raise RuntimeError(f"float32 가 아닌 weight_scb 가 남았습니다: {left[:3]} 외 {max(0, len(left) - 3)}개")
    return fixed


def install() -> None:
    """CheckpointInfo.get_moshi 를 감싼다. quantize 가 아닌 모델에는 아무것도 하지 않는다."""
    from moshi.models import loaders

    orig = loaders.CheckpointInfo.get_moshi
    if getattr(orig, "_q8_fixed", False):
        return

    def get_moshi(self, *args, **kwargs):
        lm = orig(self, *args, **kwargs)
        if (self.lm_config or {}).get("quantize"):
            n = fix_q8_scales(lm, self.moshi_weights)
            print(f"[moshi_q8] weight_scb {n}개를 float32 로 다시 불러왔습니다", flush=True)
        return lm

    get_moshi._q8_fixed = True
    loaders.CheckpointInfo.get_moshi = get_moshi


def main() -> None:
    install()
    from moshi import server
    sys.argv[0] = "moshi.server"
    server.main()


if __name__ == "__main__":
    main()
