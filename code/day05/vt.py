import modal

app = modal.App("ironman-day05")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "numpy")


@app.function(image=image, gpu="L40S", timeout=600)
def run():
    import torch
    import torch._dynamo as dynamo

    print("torch", torch.__version__)
    x = torch.randn(8, device="cuda")

    class Config:
        scale = 2

    cfg = Config()

    def helper(t):
        return t * 3

    def f(x, n, items, cfg=cfg):
        y = helper(x)
        return y * n + items[0] + cfg.scale

    print("\n=== 1. trace_bytecode: what is on the stack ===")
    torch._logging.set_logs(trace_bytecode=True)
    torch.compile(f)(x, 3, [1, 2])
    dynamo.reset()

    print("\n=== 2. graph_code: ints baked, helper inlined ===")
    torch._logging.set_logs(trace_bytecode=False, graph_code=True)
    torch.compile(f)(x, 3, [1, 2])
    dynamo.reset()

    print("\n=== 3. guards: Source names ===")
    torch._logging.set_logs(graph_code=False, guards=True)
    torch.compile(f)(x, 3, [1, 2])
    dynamo.reset()

    print("\n=== 4. recompile when a baked constant changes ===")
    torch._logging.set_logs(guards=False, recompiles=True)
    g = torch.compile(f)
    g(x, 3, [1, 2])
    g(x, 4, [1, 2])
    cfg.scale = 5
    g(x, 4, [1, 2])
    dynamo.reset()

    print("\n=== 5. Source objects ===")
    from torch._dynamo.source import AttrSource, GetItemSource, GlobalSource, LocalSource

    s1 = LocalSource("x")
    s2 = AttrSource(GlobalSource("cfg"), "scale")
    s3 = GetItemSource(LocalSource("items"), 0)
    for s in (s1, s2, s3):
        print(f"{type(s).__name__:14s} name() = {s.name()}")


@app.local_entrypoint()
def main():
    run.remote()
