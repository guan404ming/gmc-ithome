import os
import re
import tempfile

os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="torchinductor_day22_")

import torch
from torch._inductor import config, cpp_builder
from torch._inductor.cpu_vec_isa import pick_vec_isa
from torch._inductor.utils import run_and_get_code

config.force_disable_caches = True
config.compile_threads = 1


def kernels(fn, *args):
    _, code = run_and_get_code(torch.compile(fn), *args)
    return re.findall(r"cpp_pybinding\(.*?'''(.*?)'''", "\n".join(code), re.S)


def f(x, y):
    return torch.relu(x + y) * 2


def sq_sum(x):
    return (x * x).sum()


isa = pick_vec_isa()
print("threads:", torch.get_num_threads(), "| isa:", isa, "| bit_width:", isa.bit_width(), flush=True)

print("== case 1: f, n=1024, default ==", flush=True)
print(kernels(f, torch.randn(1024), torch.randn(1024))[0], flush=True)

torch._dynamo.reset()
config.cpp.simdlen = 1
print("== case 2: f, n=1024, simdlen=1 ==", flush=True)
print(kernels(f, torch.randn(1024), torch.randn(1024))[0], flush=True)
config.cpp.simdlen = None

torch._dynamo.reset()
print("== case 3: f, n=3 ==", flush=True)
print(kernels(f, torch.randn(3), torch.randn(3))[0], flush=True)

torch._dynamo.reset()
print("== case 4: f, n=1048576 ==", flush=True)
print(kernels(f, torch.randn(1 << 20), torch.randn(1 << 20))[0], flush=True)

print("== case 5: omp threshold ==", flush=True)
for n in (16384, 32768):
    torch._dynamo.reset()
    src = kernels(f, torch.randn(n), torch.randn(n))[0]
    print(f"n={n}:", "omp parallel" if "#pragma omp parallel" in src else "single thread", flush=True)

torch._dynamo.reset()
print("== case 6: sq_sum, n=1048576 ==", flush=True)
print(kernels(sq_sum, torch.randn(1 << 20))[0], flush=True)

print("== case 7: compile & cache ==", flush=True)
orig = cpp_builder.run_compile_cmd
cpp_builder.run_compile_cmd = lambda cmd, cwd=None: (print("compile cmd:", cmd, flush=True), orig(cmd, cwd=cwd))[1]
torch._dynamo.reset()
kernels(f, torch.randn(1024), torch.randn(1024))
cache = os.environ["TORCHINDUCTOR_CACHE_DIR"]
for root, _, files in sorted(os.walk(cache)):
    for name in sorted(files):
        if name.endswith((".cpp", ".so", ".py")):
            print(os.path.relpath(os.path.join(root, name), cache), flush=True)
