import os
import shutil
import subprocess
import sys
import time

CACHE_DIR = "/tmp/torchinductor_day24"


def child(n):
    import torch

    def f(x, y):
        return torch.nn.functional.gelu(x @ y + 1).sum(dim=1)

    x = torch.randn(n, n)
    y = torch.randn(n, n)
    cf = torch.compile(f)
    t0 = time.perf_counter()
    cf(x, y)
    t1 = time.perf_counter()
    cf(x, y)
    t2 = time.perf_counter()
    c = torch._dynamo.utils.counters["inductor"]
    print(f"  first call (compile): {t1 - t0:6.2f} s")
    print(f"  second call (cached): {(t2 - t1) * 1000:6.2f} ms")
    print(f"  fxgraph_cache_miss={c['fxgraph_cache_miss']} fxgraph_cache_hit={c['fxgraph_cache_hit']}")


def run(tag, n=512, env=None):
    print(f"[{tag}]")
    e = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=CACHE_DIR)
    if env:
        e.update(env)
    r = subprocess.run([sys.executable, __file__, "child", str(n)], env=e, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    return r.stderr


def main():
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    run("run 1: fresh cache dir, n=512")
    run("run 2: same cache dir, n=512")

    print("[cache dir contents]")
    for root, dirs, files in sorted(os.walk(CACHE_DIR)):
        dirs.sort()
        rel = os.path.relpath(root, CACHE_DIR)
        for fn in sorted(files):
            kb = os.path.getsize(os.path.join(root, fn)) / 1024
            print(f"  {rel}/{fn}  ({kb:.0f} KB)")

    run("run 3: same cache dir, n=768", n=768)
    run("run 4: same cache dir, n=512, TORCHINDUCTOR_FX_GRAPH_CACHE=0", env={"TORCHINDUCTOR_FX_GRAPH_CACHE": "0"})
    run("run 5: same cache dir, n=512, TORCHINDUCTOR_CPP_WRAPPER=1", env={"TORCHINDUCTOR_CPP_WRAPPER": "1"})

    print("[codecache log, n=512 again]")
    err = run("run 6: TORCH_LOGS=+torch._inductor.codecache", env={"TORCH_LOGS": "+torch._inductor.codecache"})
    for line in err.splitlines():
        if "cache" in line and ("hit" in line or "miss" in line or "key" in line):
            print("  " + line.split("] ", 1)[-1][:200])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(int(sys.argv[2]))
    else:
        main()
