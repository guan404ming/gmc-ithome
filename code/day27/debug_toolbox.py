import glob
import os
import re
import shutil
import subprocess
import sys

import depyf
import torch
from torch._dynamo import register_backend

CACHE_DIR = "/tmp/torchinductor_day27"
DBG_CACHE_DIR = "/tmp/torchinductor_day27_dbg"
DEPYF_DIR = "/tmp/day27_depyf"
DEBUG_DIR = "/tmp/day27_compile_debug"
MINI_DIR = "/tmp/day27_minifier"


def f(x, n):
    y = torch.sin(x) + 1
    if y.sum() > 0:
        y = y * 2
    return torch.relu(y) * n


def g(x):
    y = torch.sin(x) + 1
    z = torch.relu(y)
    return torch.cos(z) * 2


def register_bad_backend():
    @register_backend
    def day27_bad(gm, example_inputs):
        for node in gm.graph.nodes:
            if "relu" in str(node.target):
                raise RuntimeError("day27_bad backend cannot handle relu")
        return gm.forward


def child_compile():
    cf = torch.compile(f)
    cf(torch.randn(32, 32), 10)


def child_recompile():
    cf = torch.compile(f)
    cf(torch.randn(32, 32), 10)
    cf(torch.randn(48, 48), 10)
    cf(torch.randn(64, 64), 10)


def child_explain():
    print(torch._dynamo.explain(f)(torch.randn(32, 32), 10))


def child_depyf():
    with depyf.prepare_debug(DEPYF_DIR):
        cf = torch.compile(f)
        cf(torch.randn(32, 32), 10)


def child_crash():
    register_bad_backend()
    cf = torch.compile(g, backend="day27_bad")
    try:
        cf(torch.randn(8, 8))
    except Exception as e:
        print(f"  {type(e).__name__}: day27_bad backend cannot handle relu")


def child_minify():
    register_bad_backend()
    launcher = glob.glob(MINI_DIR + "/torch_compile_debug/run_*/minifier/minifier_launcher.py")[0]
    src = open(launcher).read().replace("if __name__ == '__main__':", "if True:")
    exec(compile(src, launcher, "exec"), {})


def run(tag, mode, env=None, cwd=None):
    print(f"[{tag}]")
    e = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=CACHE_DIR)
    if env:
        e.update(env)
    r = subprocess.run([sys.executable, os.path.abspath(__file__), mode], env=e, capture_output=True, text=True, cwd=cwd)
    sys.stdout.write(r.stdout)
    return r.stderr


def emit(stderr, tag, pat=None, limit=200):
    for line in stderr.splitlines():
        if f"[{tag}]" in line:
            msg = line.split(f"[{tag}]", 1)[1].strip()
            if pat is None or re.search(pat, msg):
                print("  " + msg[:limit])


def main():
    for d in (CACHE_DIR, DBG_CACHE_DIR, DEPYF_DIR, DEBUG_DIR, MINI_DIR):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(MINI_DIR)

    err = run("tool 1: TORCH_LOGS=graph_breaks", "compile", env={"TORCH_LOGS": "graph_breaks"})
    emit(err, "__graph_breaks")

    err = run("tool 2: TORCH_LOGS=recompiles", "recompile", env={"TORCH_LOGS": "recompiles"})
    emit(err, "__recompiles")

    err = run("tool 3: TORCH_LOGS=output_code", "compile", env={"TORCH_LOGS": "output_code"})
    emit(err, "__output_code", pat=r"cpp_fused|Output code written")

    run("tool 4: torch._dynamo.explain", "explain")

    run("tool 5: depyf.prepare_debug", "depyf")
    print(f"[depyf dump dir: {DEPYF_DIR}]")
    for fn in sorted(os.listdir(DEPYF_DIR)):
        print("  " + fn)
    print("[__transformed_code_0_for_f.py]")
    print(open(DEPYF_DIR + "/__transformed_code_0_for_f.py").read())
    resume = glob.glob(DEPYF_DIR + "/__transformed_code_0_for_torch_dynamo_resume_in_f_*.py")[0]
    print(f"[{os.path.basename(resume)}]")
    print(open(resume).read())

    run("tool 6: TORCH_COMPILE_DEBUG=1", "compile", env={"TORCH_COMPILE_DEBUG": "1", "TORCH_COMPILE_DEBUG_DIR": DEBUG_DIR, "TORCHINDUCTOR_CACHE_DIR": DBG_CACHE_DIR})
    for root, dirs, files in sorted(os.walk(DEBUG_DIR)):
        dirs.sort()
        rel = os.path.relpath(root, DEBUG_DIR)
        for fn in sorted(files):
            kb = os.path.getsize(os.path.join(root, fn)) / 1024
            print(f"  {rel}/{fn}  ({kb:.0f} KB)")

    run("tool 7a: TORCHDYNAMO_REPRO_AFTER=dynamo", "crash", env={"TORCHDYNAMO_REPRO_AFTER": "dynamo", "TORCH_COMPILE_DEBUG_DIR": MINI_DIR})
    launcher = glob.glob(MINI_DIR + "/torch_compile_debug/run_*/minifier/minifier_launcher.py")
    print(f"  minifier_launcher.py written: {bool(launcher)}")

    out = run("tool 7b: run minifier_launcher", "minify", env={"TORCH_COMPILE_DEBUG_DIR": MINI_DIR}, cwd=MINI_DIR)
    for line in out.splitlines():
        if "Went from" in line or "minimal repro" in line:
            print("  " + line.split("] ", 1)[-1].strip())
    print("[minified repro.py forward]")
    src = open(MINI_DIR + "/repro.py").read()
    body = src[src.index("class Repro") : src.index("mod = Repro()")]
    print("\n".join(l for l in body.splitlines() if l.strip()))


if __name__ == "__main__":
    modes = {"compile": child_compile, "recompile": child_recompile, "explain": child_explain, "depyf": child_depyf, "crash": child_crash, "minify": child_minify}
    if len(sys.argv) > 1:
        modes[sys.argv[1]]()
    else:
        main()
