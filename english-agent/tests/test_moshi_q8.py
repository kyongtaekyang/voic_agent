"""core.moshi_q8: bfloat16 으로 망가진 QLinear.weight_scb 를 파일의 float32 로 되돌리는지 (CPU, 실제 가중치 불필요)."""
import pytest
import torch
from torch import nn

pytest.importorskip("bitsandbytes")
pytest.importorskip("moshi")

from moshi.utils.quantize import QLinear, replace_linear_with_qlinear  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from core.moshi_q8 import fix_q8_scales  # noqa: E402


def _tiny():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(16, 8, bias=False), nn.Sequential(nn.Linear(8, 4, bias=False)))
    replace_linear_with_qlinear(m)
    return m


def test_restores_float32_scales_exactly(tmp_path):
    m = _tiny()
    state = {k: v.detach().clone() for k, v in m.state_dict().items()}
    path = tmp_path / "q8.safetensors"
    save_file(state, str(path))
    # moshi 0.2.13 의 로더가 하는 것처럼 스케일을 bfloat16 으로 망가뜨린다
    for mod in m.modules():
        if isinstance(mod, QLinear):
            mod.weight_scb = nn.Parameter(mod.weight_scb.to(torch.bfloat16), requires_grad=False)
    assert fix_q8_scales(m, path) == 2
    for name, mod in m.named_modules():
        if isinstance(mod, QLinear):
            assert mod.weight_scb.dtype == torch.float32
            assert torch.equal(mod.weight_scb, state[f"{name}.weight_scb"])


def test_raises_when_scale_missing_from_file(tmp_path):
    m = _tiny()
    state = {k: v.detach().clone() for k, v in m.state_dict().items() if k != "1.0.weight_scb"}
    path = tmp_path / "q8.safetensors"
    save_file(state, str(path))
    m[1][0].weight_scb = nn.Parameter(m[1][0].weight_scb.to(torch.bfloat16), requires_grad=False)
    with pytest.raises(RuntimeError, match="float32"):
        fix_q8_scales(m, path)
