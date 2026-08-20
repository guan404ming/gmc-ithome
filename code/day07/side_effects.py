import modal

app = modal.App("ironman-day07")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "numpy")


@app.function(image=image, gpu="L40S", timeout=600)
def run():
    import torch

    print("torch", torch.__version__)
    log = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            log.append(self.calls)
            return x * 2

    m = Model()
    cm = torch.compile(m)
    torch._logging.set_logs(graph_code=True, bytecode=True)
    cm(torch.randn(4, device="cuda"))
    torch._logging.set_logs(graph_code=False, bytecode=False)
    cm(torch.randn(4, device="cuda"))
    print("calls =", m.calls, "| log =", log)

    print("\n=== read-your-own-write during tracing ===")

    def g(x, obj):
        obj.k = 7
        return x + obj.k

    class O:
        k = 0

    o = O()
    torch._logging.set_logs(graph_code=True)
    torch.compile(g)(torch.randn(4, device="cuda"), o)
    torch._logging.set_logs(graph_code=False)
    print("obj.k after call =", o.k)

    print("\n=== dead new object is pruned ===")

    def h(x):
        tmp = []
        tmp.append(1)
        return x * 2

    torch._logging.set_logs(bytecode=True)
    torch.compile(h)(torch.randn(4, device="cuda"))


@app.local_entrypoint()
def main():
    run.remote()
