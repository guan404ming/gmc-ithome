import modal

app = modal.App("ironman-day25")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "numpy")


@app.function(image=image, gpu="L40S", timeout=900)
def run():
    import statistics
    import torch
    import torch.nn as nn

    print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))

    torch.manual_seed(0)
    layers = []
    for _ in range(32):
        layers.append(nn.Linear(256, 256))
        layers.append(nn.ReLU())
    model = nn.Sequential(*layers).cuda()

    def bench(fn, x, iters=100):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(x)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    def bench10(fn, x):
        r = [bench(fn, x) for _ in range(10)]
        return statistics.mean(r), statistics.stdev(r)

    compiled = torch.compile(model)
    reduced = torch.compile(model, mode="reduce-overhead")

    print("\n=== 32 x Linear(256, 256) + ReLU, batch=8 ===")
    x_small = torch.randn(8, 256, device="cuda")
    with torch.no_grad():
        for _ in range(5):
            model(x_small)
            compiled(x_small)
            reduced(x_small)
        e, es = bench10(model, x_small)
        c, cs = bench10(compiled, x_small)
        r, rs = bench10(reduced, x_small)
    print(f"eager            {e:.3f} ms (+/- {es:.3f})")
    print(f"compile default  {c:.3f} ms (+/- {cs:.3f})")
    print(f"reduce-overhead  {r:.3f} ms (+/- {rs:.3f})")
    print(f"default vs eager          {e / c:.2f}x")
    print(f"reduce-overhead vs eager  {e / r:.2f}x")
    print(f"reduce-overhead vs default {c / r:.2f}x")

    print("\n=== same model, batch=8192 ===")
    x_big = torch.randn(8192, 256, device="cuda")
    with torch.no_grad():
        for _ in range(5):
            model(x_big)
            compiled(x_big)
            reduced(x_big)
        e, es = bench10(model, x_big)
        c, cs = bench10(compiled, x_big)
        r, rs = bench10(reduced, x_big)
    print(f"eager            {e:.3f} ms (+/- {es:.3f})")
    print(f"compile default  {c:.3f} ms (+/- {cs:.3f})")
    print(f"reduce-overhead  {r:.3f} ms (+/- {rs:.3f})")
    print(f"reduce-overhead vs default {c / r:.2f}x")

    print("\n=== reduce-overhead outputs live in the graph pool ===")
    with torch.no_grad():
        y1 = reduced(x_small)
        p1 = y1.data_ptr()
        del y1
        y2 = reduced(x_small)
        p2 = y2.data_ptr()
        print(f"release output, call again: ptr {p1} -> {p2}, same: {p1 == p2}")
        y3 = reduced(x_small)
        p3 = y3.data_ptr()
        y4 = reduced(x_small)
        p4 = y4.data_ptr()
        print(f"hold output alive, call again: ptr {p3} -> {p4}, same: {p3 == p4}")
        del y3, y4

    print("\n=== skip reason: input mutation ===")
    torch._logging.set_logs(perf_hints=True)

    def mutate(x):
        x.add_(1)
        return x * 2

    with torch.no_grad():
        torch.compile(mutate, mode="reduce-overhead")(torch.randn(8, 256, device="cuda"))
    torch._logging.set_logs()

    print("\n=== manual torch.cuda.CUDAGraph capture / replay ===")
    static_x = torch.randn(8, 256, device="cuda")
    with torch.no_grad():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                model(static_x)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_y = model(static_x)

        new_x = torch.randn(8, 256, device="cuda")
        static_x.copy_(new_x)
        g.replay()
        torch.cuda.synchronize()
        ref = model(new_x)
        print(f"replay result matches eager: {torch.allclose(static_y, ref)}")

        def eager_loop(x):
            model(x)

        def graph_replay(x):
            static_x.copy_(x)
            g.replay()

        el, _ = bench10(eager_loop, new_x)
        gl, _ = bench10(graph_replay, new_x)
    print(f"eager 64 launches  {el:.3f} ms")
    print(f"graph replay       {gl:.3f} ms")
    print(f"replay vs eager    {el / gl:.2f}x")


@app.local_entrypoint()
def main():
    run.remote()
