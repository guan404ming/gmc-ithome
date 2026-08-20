import modal

app = modal.App("ironman-day11")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "numpy")


@app.function(image=image, gpu="L40S", timeout=600)
def run():
    import torch
    import torch._dynamo as dynamo

    print("torch", torch.__version__)

    def f(x, y):
        return x @ y

    g = torch.compile(f)
    y = torch.randn(4, 8, device="cuda")
    print("=== call 1: (4,4) static ===")
    torch._logging.set_logs(guards=True)
    g(torch.randn(4, 4, device="cuda"), y)
    torch._logging.set_logs(guards=False)

    print("\n=== call 2: (8,4) -> recompile with SymInt ===")
    torch._logging.set_logs(recompiles=True, guards=True)
    g(torch.randn(8, 4, device="cuda"), y)
    torch._logging.set_logs(recompiles=False, guards=False)

    print("\n=== call 3/4: (16,4) (100,4) -> no recompile ===")
    torch._logging.set_logs(recompiles=True)
    g(torch.randn(16, 4, device="cuda"), y)
    g(torch.randn(100, 4, device="cuda"), y)
    torch._logging.set_logs(recompiles=False)

    print("\n=== SymInt expression propagation ===")
    dynamo.reset()

    def h(x):
        b = x.shape[0]
        return x.reshape(b * 2, -1), b * 2

    torch._logging.set_logs(graph_code=True)
    hh = torch.compile(h, dynamic=True)
    hh(torch.randn(4, 6, device="cuda"))
    torch._logging.set_logs(graph_code=False)

    print("\n=== shape if: guard on s0 ===")
    dynamo.reset()

    def k(x):
        if x.shape[0] > 10:
            return x * 2
        return x + 1

    kk = torch.compile(k, dynamic=True)
    torch._logging.set_logs(recompiles=True)
    kk(torch.randn(4, device="cuda"))
    kk(torch.randn(20, device="cuda"))
    torch._logging.set_logs(recompiles=False)

    print("\n=== mark_dynamic + shape==4 -> ConstraintViolation ===")
    dynamo.reset()

    def bad(x):
        if x.shape[0] == 4:
            return x * 2
        return x + 1

    x4 = torch.randn(4, 5, device="cuda")
    torch._dynamo.mark_dynamic(x4, 0)
    try:
        torch.compile(bad)(x4)
    except Exception as e:
        print(type(e).__name__, ":", str(e)[:500])

    print("\n=== unbacked: item() ===")
    dynamo.reset()

    def ub(x):
        n = int(x.sum().item())
        return torch.zeros(n, device="cuda")

    print(dynamo.explain(ub)(torch.tensor([3.0], device="cuda")))


@app.local_entrypoint()
def main():
    run.remote()
