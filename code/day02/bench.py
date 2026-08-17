import modal

app = modal.App("ironman-day2")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "numpy")


@app.function(image=image, gpu="L40S", timeout=600)
def run():
    import torch
    import torch._dynamo as dynamo

    print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))

    def f(x):
        return torch.sin(x) * torch.cos(x) + torch.tanh(x)

    x = torch.randn(4096, 4096, device="cuda")
    f(x)

    compiled = torch.compile(f)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    compiled(x)
    t1.record()
    torch.cuda.synchronize()
    print(f"first call (compile + run): {t0.elapsed_time(t1):.1f} ms")

    def bench(fn, iters=100):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(x)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    import statistics

    def bench10(fn):
        r = [bench(fn) for _ in range(10)]
        return statistics.mean(r), statistics.stdev(r)

    e, es = bench10(f)
    c, cs = bench10(compiled)
    print(f"eager    {e:.3f} ms (+/- {es:.3f})")
    print(f"compiled {c:.3f} ms (+/- {cs:.3f})")
    print(f"speedup  {e / c:.2f}x")

    print("\n=== backend variants ===")
    for b in ["eager", "aot_eager", "inductor"]:
        g = torch.compile(f, backend=b)
        g(x)
        m, sd = bench10(g)
        print(f"backend={b:10s} {m:.3f} ms (+/- {sd:.3f})")

    print("\n=== dynamo.explain ===")
    print(dynamo.explain(f)(x))

    print("\n=== dump: dynamo graph / aot graphs / inductor output code ===")
    dynamo.reset()
    torch._inductor.config.force_disable_caches = True
    torch._logging.set_logs(graph_code=True, aot_graphs=True, output_code=True)
    torch.compile(f)(x)


@app.local_entrypoint()
def main():
    run.remote()
