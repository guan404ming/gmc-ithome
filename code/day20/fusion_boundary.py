import os
import re
import subprocess
import sys

CASES = [
    (
        "A: pointwise chain",
        """
import torch
def f(x):
    return torch.sin(torch.relu(x + 1) * 2)
torch.compile(f)(torch.randn(1024, 1024))
""",
    ),
    (
        "B: pointwise + row sum",
        """
import torch
def f(x):
    return torch.relu(x + 1).sum(dim=1)
torch.compile(f)(torch.randn(1024, 1024))
""",
    ),
    (
        "C: softmax (row reduction + pointwise)",
        """
import torch
def f(x):
    s = torch.exp(x - x.amax(dim=1, keepdim=True))
    return s / s.sum(dim=1, keepdim=True)
torch.compile(f)(torch.randn(1024, 1024))
""",
    ),
    (
        "D: global mean then pointwise",
        """
import torch
def f(x):
    return torch.relu(x - x.mean())
torch.compile(f)(torch.randn(1024, 1024))
""",
    ),
    (
        "E: sibling reductions over same input",
        """
import torch
def f(x):
    return x.amax(dim=1), x.sum(dim=1)
torch.compile(f)(torch.randn(1024, 1024))
""",
    ),
    (
        "F: sibling branches, nothing shared",
        """
import torch
def f(a, b):
    return (a + 1).relu(), (b * 2).sin()
torch.compile(f)(torch.randn(1024), torch.randn(1024))
""",
    ),
    (
        "G: pointwise -> mm -> pointwise",
        """
import torch
def f(x, w):
    return torch.relu((x + 1) @ w)
torch.compile(f)(torch.randn(256, 256), torch.randn(256, 256))
""",
    ),
    (
        "H: seven ops across one wall",
        """
import torch
def f(x):
    y = torch.sin(torch.relu(x + 1) * 2)
    return torch.relu(y - y.mean())
torch.compile(f)(torch.randn(1024, 1024))
""",
    ),
]


def run_case(title, code):
    env = {**os.environ, "TORCH_LOGS": "fusion,output_code", "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    out = r.stderr
    print(f"===== {title} =====")
    for line in out.splitlines():
        if "[__fusion]" in line:
            msg = line.split("[__fusion]")[1].strip()
            if msg:
                print("  " + msg)
    kernels = []
    for m in re.finditer(r"(cpp_fused\w+) = async_compile", out):
        kernels.append(m.group(1))
    calls = re.findall(r"(cpp_fused\w+)\(|extern_kernels\.(\w+)\(", out)
    call_seq = [a or f"extern_kernels.{b}" for a, b in calls]
    seen = set()
    call_seq = [c for c in call_seq if not (c in seen or seen.add(c))]
    print(f"  compiled kernels: {kernels}")
    print(f"  call sequence: {call_seq}")
    loops = re.findall(r"for\(int64_t x0=.*x0\+=", out)
    print(f"  outer loop nests in cpp kernels: {len(loops)}")
    print()


def main():
    import torch

    print(f"torch {torch.__version__} | device: cpu")
    print()
    for title, code in CASES:
        run_case(title, code)


if __name__ == "__main__":
    main()
