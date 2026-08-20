import modal

app = modal.App("ironman-day12")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "numpy")


@app.function(image=image, gpu="L40S", timeout=600)
def run():
    import torch

    print("torch", torch.__version__)

    def f(x, w):
        return (x @ w).relu().sum()

    x = torch.randn(4, 4, device="cuda")
    w = torch.randn(4, 4, device="cuda", requires_grad=True)

    print("=== inference: only forward graph ===")
    torch._logging.set_logs(aot_graphs=True)
    with torch.no_grad():
        torch.compile(f)(x, w)

    print("\n=== training: forward + backward graphs ===")
    torch._dynamo.reset()
    out = torch.compile(f)(x, w)
    out.backward()
    print("w.grad shape:", w.grad.shape)
    torch._logging.set_logs(aot_graphs=False)

    print("\n=== dynamo graph vs aot graph op names ===")
    torch._dynamo.reset()
    torch._logging.set_logs(graph_code=True)
    with torch.no_grad():
        torch.compile(f)(x, w)


@app.local_entrypoint()
def main():
    run.remote()
