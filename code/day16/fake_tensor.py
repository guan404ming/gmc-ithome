import io
import logging
import time

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

print("torch", torch.__version__)

print("\n=== 1. meta device: shape without values ===")
m = torch.empty(4, 8, device="meta")
print(m)
print("shape:", m.shape, "| dtype:", m.dtype, "| stride:", m.stride(), "| device:", m.device)
y = m @ torch.empty(8, 16, device="meta")
print("matmul ->", y.shape, y.device)
try:
    m[0, 0].item()
except Exception as e:
    print("item() ->", type(e).__name__, "-", e)

print("\n=== 2. FakeTensorMode: run ops, get fake outputs ===")
mode = FakeTensorMode()
with mode:
    a = torch.randn(32, 64)
    b = torch.randn(64, 128)
    c = torch.relu(a @ b)
print("type:", type(c).__name__, "| shape:", tuple(c.shape), "| dtype:", c.dtype, "| device:", c.device)

real = torch.arange(6.0).reshape(2, 3)
fake = mode.from_tensor(real)
print("from_tensor:", type(fake).__name__, tuple(fake.shape), fake.device)

t0 = time.perf_counter()
with mode:
    h = torch.randn(65536, 65536)
    hh = h @ h
t1 = time.perf_counter()
print(f"fake 65536x65536 matmul (16 GB per tensor): {(t1 - t0) * 1000:.2f} ms ->", tuple(hh.shape))

print("\n=== 3. example_value inside torch.compile ===")


def peek(gm, example_inputs):
    print("backend example_inputs:", [type(t).__name__ for t in example_inputs])
    for n in gm.graph.nodes:
        ev = n.meta.get("example_value")
        if isinstance(ev, torch.Tensor):
            print(f"  {n.op:13s} {n.name:6s} example_value = {type(ev).__name__}{tuple(ev.shape)}")
    return gm.forward


def f(x, w):
    return torch.tanh(x @ w)


torch.compile(f, backend=peek)(torch.randn(8, 16), torch.randn(16, 4))

print("\n=== 3b. TORCH_LOGS evidence: graph inputs are FakeTensor ===")
buf = io.StringIO()
handler = logging.StreamHandler(buf)
logging.getLogger("torch._dynamo").addHandler(handler)
torch._dynamo.reset()
torch._logging.set_logs(dynamo=logging.DEBUG)
torch.compile(f)(torch.randn(8, 16), torch.randn(16, 4))
torch._logging.set_logs()
logging.getLogger("torch._dynamo").removeHandler(handler)
for ln in buf.getvalue().splitlines():
    if "create_graph_input" in ln:
        print(" ", ln)

print("\n=== 4. what a fake value cannot answer ===")
with mode:
    s = torch.randn(())
    try:
        s.item()
    except Exception as e:
        print("fake item() ->", type(e).__name__)
        print("  ", e)


def g(x):
    if x.sum() > 0:
        return x + 1
    return x - 1


try:
    torch.compile(g, fullgraph=True)(torch.randn(4))
except Exception as e:
    print("compile data-dependent branch ->", type(e).__name__)
    msg = str(e)
    print(msg[: msg.find("\n\n")] if "\n\n" in msg else msg[:400])
