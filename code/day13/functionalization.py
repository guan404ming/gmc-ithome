import modal

app = modal.App("ironman-day13")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "numpy")


@app.function(image=image, gpu="L40S", timeout=600)
def run():
    import torch

    print("torch", torch.__version__)

    def f(x):
        y = x.view(2, 8)
        y.add_(1)
        y.relu_()
        return x * 2

    x = torch.randn(4, 4, device="cuda")

    print("=== dynamo graph: in-place ops kept ===")
    torch._logging.set_logs(graph_code=True)
    torch.compile(f)(x)
    torch._logging.set_logs(graph_code=False)

    print("\n=== aot graph: functionalized ===")
    torch._dynamo.reset()
    torch._inductor.config.force_disable_caches = True
    torch._logging.set_logs(aot_graphs=True)
    torch.compile(f)(x)
    torch._logging.set_logs(aot_graphs=False)

    print("\n=== semantics preserved ===")
    x1 = torch.randn(4, 4, device="cuda")
    x2 = x1.clone()
    r1 = f(x1)
    r2 = torch.compile(f)(x2)
    print("outputs equal:", torch.equal(r1, r2), "| inputs equal after mutation:", torch.equal(x1, x2))

    print("\n=== input mutation becomes copy_ at the end ===")
    torch._dynamo.reset()

    def g(x):
        x.mul_(2)
        return x + 1

    torch._logging.set_logs(aot_graphs=True)
    torch.compile(g)(torch.randn(4, device="cuda"))


@app.local_entrypoint()
def main():
    run.remote()
