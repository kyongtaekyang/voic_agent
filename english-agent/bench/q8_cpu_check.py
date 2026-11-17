"""Moshi PyTorch INT8 (kyutai/moshiko-pytorch-q8) CPU 로드·1프레임 확인. GPU 를 쓰지 않는다 (CUDA_VISIBLE_DEVICES="")."""
import os, time, sys
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU 차단 필요"
import torch
from moshi.models import loaders, LMGen
from moshi.utils.quantize import QLinear
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core.moshi_q8 import install
if "--no-fix" not in sys.argv:   # --no-fix: moshi 0.2.13 그대로 (weight_scb bfloat16 오류 재현)
    install()

t0 = time.time()
info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-q8")
print("quantize in lm_config:", info.lm_config.get("quantize"))
mimi = info.get_mimi(device="cpu")
lm = info.get_moshi(device="cpu", dtype=torch.bfloat16)
nq = sum(isinstance(m, QLinear) for m in lm.modules())
print(f"loaded in {time.time()-t0:.1f}s, QLinear={nq}, weight dtype sample:",
      next(m for m in lm.modules() if isinstance(m, QLinear)).weight.dtype)
param_bytes = sum(p.numel() * p.element_size() for p in lm.parameters())
print(f"LM parameter bytes: {param_bytes/2**20:.0f} MiB")

gen = LMGen(lm, temp=0.8, temp_text=0.7)
frame = int(mimi.sample_rate / mimi.frame_rate)
ok = 0
with torch.no_grad(), gen.streaming(1), mimi.streaming(1):
    for i in range(3):
        codes = mimi.encode(torch.zeros(1, 1, frame))
        t = time.time()
        try:
            out = gen.step(codes)
        except Exception as e:
            print("step failed:", type(e).__name__, str(e)[:300]); sys.exit(2)
        print(f"step {i}: {time.time()-t:.2f}s (CPU, 참고용) out={None if out is None else tuple(out.shape)}")
        ok += 1
print("CPU_OK", ok)
