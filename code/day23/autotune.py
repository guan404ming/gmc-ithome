import modal

app = modal.App("ironman-day23")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "numpy")


@app.function(image=image, gpu="L40S", timeout=1200)
def run():
    import os
    import time

    os.environ["TORCH_LOGS"] = "+torch._inductor.runtime.coordinate_descent_tuner"

    import torch
    from torch._inductor.utils import run_and_get_code

    print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
    print("max-autotune options:", torch._inductor.list_mode_options("max-autotune"))

    def mm(x, y):
        return x @ y

    def mm_relu(x, y):
        return torch.relu(x @ y)

    def kernel_calls(code):
        for ln in code.splitlines():
            s = ln.strip()
            if s.startswith("extern_kernels.") or ".run(" in s:
                print("   ", s.split("(")[0] + "(...)")

    def compile_and_report(fn, tag, args, **kw):
        torch._dynamo.reset()
        g = torch.compile(fn, **kw)
        t0 = time.perf_counter()
        _, codes = run_and_get_code(g, *args)
        print(f"[{tag}] compile time: {time.perf_counter() - t0:.1f} s")
        kernel_calls(codes[0])
        return g

    def bench(fn, args, iters=200):
        for _ in range(10):
            fn(*args)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn(*args)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    def compare(fn, args):
        fd = compile_and_report(fn, "default", args)
        fm = compile_and_report(fn, "max-autotune", args, mode="max-autotune-no-cudagraphs")
        print(f"eager           {bench(fn, args):.4f} ms")
        print(f"default         {bench(fd, args):.4f} ms")
        print(f"max-autotune    {bench(fm, args):.4f} ms")

    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    print("\n=== mm 2048x2048x2048 fp16 ===")
    compare(mm, (a, b))

    s1 = torch.randn(16, 4096, device="cuda", dtype=torch.float16)
    s2 = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    print("\n=== mm 16x4096x4096 fp16 ===")
    compare(mm, (s1, s2))

    print("\n=== mm + relu epilogue 2048x2048x2048 ===")
    compare(mm_relu, (a, b))

    print("\n=== coordinate descent tuning: softmax ===")

    def soft(x, y):
        return torch.softmax(x + y, dim=-1)

    torch._dynamo.reset()
    fs = torch.compile(soft, mode="max-autotune-no-cudagraphs")
    t0 = time.perf_counter()
    fs(a, b)
    torch.cuda.synchronize()
    print(f"compile time: {time.perf_counter() - t0:.1f} s")


@app.local_entrypoint()
def main():
    run.remote()
