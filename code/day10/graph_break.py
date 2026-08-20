import modal

app = modal.App("ironman-day10")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "numpy")


@app.function(image=image, gpu="L40S", timeout=600)
def run():
    import torch
    import torch._dynamo as dynamo

    print("torch", torch.__version__)

    def f(x):
        x = x * 2
        print("mid")
        return x + 1

    x = torch.randn(4, device="cuda")

    print("\n=== 1. graph_breaks + graph_code + bytecode ===")
    torch._logging.set_logs(graph_breaks=True, graph_code=True, bytecode=True)
    torch.compile(f)(x)
    torch._logging.set_logs(graph_breaks=False, graph_code=False, bytecode=False)

    print("\n=== 2. explain ===")
    dynamo.reset()
    print(dynamo.explain(f)(x))

    print("\n=== 3. break inside inlined function ===")
    dynamo.reset()

    def util(t):
        print("log")
        return t + 1

    def big(x):
        y = x * 2
        z = util(y)
        return z * 3

    print(dynamo.explain(big)(x))

    print("\n=== 4. fullgraph=True ===")
    dynamo.reset()
    try:
        torch.compile(f, fullgraph=True)(x)
    except Exception as e:
        print(type(e).__name__, ":", str(e)[:400])


@app.local_entrypoint()
def main():
    run.remote()
