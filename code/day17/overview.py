import glob
import os

import torch
from torch._inductor.lowering import lowerings

print("torch", torch.__version__)


def f(x, y):
    return torch.relu(x + y) * 2


x = torch.randn(1024)
y = torch.randn(1024)

print("\n=== output_code: Inductor generated code (CPU) ===")
torch._logging.set_logs(output_code=True)
out = torch.compile(f)(x, y)
torch._logging.set_logs(output_code=False)
print("allclose:", torch.allclose(out, torch.relu(x + y) * 2))

print("\n=== larger input: parallel kernel ===")
torch._dynamo.reset()
xb = torch.randn(1 << 20)
yb = torch.randn(1 << 20)
torch._logging.set_logs(output_code=True)
torch.compile(f)(xb, yb)
torch._logging.set_logs(output_code=False)

print("\n=== registered lowerings ===")
print("len(lowerings):", len(lowerings))

print("\n=== TORCH_COMPILE_DEBUG artifacts ===")
for d in sorted(glob.glob("torch_compile_debug/*/torchinductor/*")):
    print(os.path.basename(d))
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            print("  ", name)

for name in ("fx_graph_readable.py", "ir_pre_fusion.txt"):
    hits = sorted(glob.glob(f"torch_compile_debug/*/torchinductor/*/{name}"))
    if hits:
        print(f"\n--- {name} ---")
        with open(hits[0]) as fh:
            print(fh.read())
