import torch
from torch._dynamo import list_backends
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.registry import register_backend

calls = {"observer": 0}


def observer(gm, example_inputs):
    calls["observer"] += 1
    print(f"[observer] call #{calls['observer']}", flush=True)
    print(gm.graph, flush=True)
    for i, t in enumerate(example_inputs):
        if isinstance(t, torch.Tensor):
            print(f"  input[{i}]: shape={tuple(t.shape)} dtype={t.dtype}", flush=True)
        else:
            print(f"  input[{i}]: {type(t).__name__} = {t}", flush=True)
    return gm.forward


def relu_to_sigmoid(gm, example_inputs):
    n = 0
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is torch.relu:
            node.target = torch.sigmoid
            n += 1
    gm.recompile()
    print(f"[rewriter] replaced {n} node(s): relu -> sigmoid", flush=True)
    return gm.forward


def fw(gm, example_inputs):
    print("[wrapper] fw_compiler got:", flush=True)
    print(gm.graph, flush=True)
    return gm.forward


aten_backend = aot_autograd(fw_compiler=fw)


def f(x):
    y = torch.relu(x)
    return y + 1


print("== part 1: observer ==", flush=True)
cf = torch.compile(f, backend=observer)
x = torch.randn(4, 8)
out = cf(x)
print("matches eager:", torch.equal(out, f(x)), flush=True)
print("-- second call, same shape --", flush=True)
cf(x)
print("observer calls so far:", calls["observer"], flush=True)
print("-- third call, new shape (4, 8) -> (6, 8) --", flush=True)
cf(torch.randn(6, 8))
print("observer calls so far:", calls["observer"], flush=True)

print("== part 2: rewriter ==", flush=True)
torch._dynamo.reset()
cg = torch.compile(f, backend=relu_to_sigmoid)
t = torch.tensor([-2.0, 0.0, 2.0])
print("eager    f(t):", f(t), flush=True)
print("compiled f(t):", cg(t), flush=True)
print("sigmoid(t)+1 :", torch.sigmoid(t) + 1, flush=True)

print("== part 3: aten wrapper ==", flush=True)
torch._dynamo.reset()
ch = torch.compile(f, backend=aten_backend)
ch(torch.randn(4, 8))

print("== part 4: registry ==", flush=True)
print("list_backends():", list_backends(), flush=True)
register_backend(observer, name="day28_observer")
print("registered:", "day28_observer" in list_backends(), flush=True)
torch._dynamo.reset()
calls["observer"] = 0
cs = torch.compile(f, backend="day28_observer")
cs(torch.randn(2, 2))
print("observer calls via registry:", calls["observer"], flush=True)
