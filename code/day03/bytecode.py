import modal

app = modal.App("ironman-day03")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "numpy")


@app.function(image=image, gpu="L40S", timeout=600)
def run():
    import dis
    import sys

    import torch
    import torch._dynamo as dynamo

    print("python", sys.version.split()[0], "| torch", torch.__version__)

    def f(x):
        return torch.sin(x) + 1

    print("\n=== dis.dis(f) ===")
    dis.dis(f)

    print("\n=== eval_frame hook ===")
    print("set_eval_frame:", torch._C._dynamo.eval_frame.set_eval_frame)

    print("\n=== TORCH_LOGS=bytecode ===")
    torch._logging.set_logs(bytecode=True)
    x = torch.randn(8, device="cuda")
    torch.compile(f)(x)

    print("\n=== graph break: data-dependent if ===")
    torch._logging.set_logs(bytecode=False, graph_breaks=True)

    def g(x):
        if x.sum() > 0:
            return x * 2
        return x + 1

    print(dynamo.explain(g)(x))


@app.local_entrypoint()
def main():
    run.remote()
