import glob
import os

DEBUG_DIR = os.environ.get("DAY18_DEBUG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug"))
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCH_COMPILE_DEBUG_DIR"] = DEBUG_DIR
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"

import torch
from torch._inductor.lowering import fallbacks, lowerings

print("torch", torch.__version__, "device cpu")


def f(x):
    y = (x * 2 + 1).relu()
    return y, y.sum(dim=1)


def g(x):
    return (x * 2 + 1).relu().sum(dim=1)


with torch.no_grad():
    torch.compile(f)(torch.randn(4, 8))
    torch.compile(g)(torch.randn(4, 8))

print("\n=== lowering table ===")
print("lowerings entries:", len(lowerings))
aten = torch.ops.aten
for op in (aten.add.Tensor, aten.relu.default, aten.sum.default, aten.mm.default, aten.convolution.default, aten._cdist_forward.default):
    kind = "in lowerings" if op in lowerings else "not in lowerings"
    if op in fallbacks:
        kind += " (fallback)"
    print(f"{str(op):32s} {kind}")

for op in (aten.relu.default, aten.sum.default):
    fn = lowerings[op]
    print(f"lowerings[{op}] -> {fn.__module__}.{fn.__qualname__}")

for name in ("ir_pre_fusion.txt", "ir_post_fusion.txt"):
    for path in sorted(glob.glob(os.path.join(DEBUG_DIR, "**", name), recursive=True)):
        model = path.split(os.sep)[-2]
        print(f"\n=== {model}/{name} ===")
        with open(path) as fh:
            print(fh.read())
