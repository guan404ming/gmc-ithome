import modal

app = modal.App("ironman-day15")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "numpy")


@app.function(image=image, gpu="L40S", timeout=600)
def run():
    import torch

    print("torch", torch.__version__)

    def f(x, w):
        return torch.tanh(x @ w).sum()

    x = torch.randn(64, 64, device="cuda")
    w = torch.randn(64, 64, device="cuda", requires_grad=True)

    print("=== joint graph then partitioned fw/bw ===")
    torch._logging.set_logs(aot_joint_graph=True, aot_graphs=True)
    out = torch.compile(f)(x, w)
    out.backward()
    torch._logging.set_logs(aot_joint_graph=False, aot_graphs=False)

    print("\n=== recompute (activation checkpoint style) ===")
    torch._dynamo.reset()
    import torch.utils.checkpoint as cp

    def g(x, w):
        return cp.checkpoint(lambda a: torch.tanh(a @ w).sum(), x, use_reentrant=False)

    torch._logging.set_logs(aot_graphs=True)
    out = torch.compile(g)(x, w)
    out.backward()


@app.local_entrypoint()
def main():
    run.remote()
