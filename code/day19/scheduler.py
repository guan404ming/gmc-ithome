import re

import torch
from torch._inductor.utils import run_and_get_code

torch._inductor.config.force_disable_caches = True
x = torch.randn(1024, 1024)


def report(fn, *args):
    _, code = run_and_get_code(torch.compile(fn), *args)
    src = "\n".join(code)
    print("kernels:", sorted(set(re.findall(r"cpp_fused\w+", src))), flush=True)
    print("loops:", src.count("#pragma omp for"), flush=True)
    print("allocs:", re.findall(r"buf\d+ = empty_strided_cpu\(.*", src), flush=True)


def chain(x):
    return torch.relu(torch.sin(x) + 1)


def epilogue(x):
    z = torch.sin(x) + 1
    s = z.sum(dim=1)
    return torch.relu(s) * 2


def wall(x):
    y = torch.sin(x)
    s = y.sum()
    return torch.relu(y + s)


def mismatch(x):
    return torch.sin(x), torch.cos(x.t()).contiguous()


print("== case 1: chain ==", flush=True)
report(chain, x)

torch._dynamo.reset()
print("== case 2: epilogue ==", flush=True)
report(epilogue, x)

torch._dynamo.reset()
print("== case 3: wall ==", flush=True)
report(wall, x)

torch._dynamo.reset()
print("== case 4: mismatch ==", flush=True)
report(mismatch, x)
