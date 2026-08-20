import modal

app = modal.App("ironman-day08")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "numpy")


@app.function(image=image, gpu="L40S", timeout=600)
def run():
    import torch

    print("torch", torch.__version__)
    bias = torch.randn(4, device="cuda")

    def f(x, y, unused):
        return (x @ y + bias).relu()

    torch._logging.set_logs(graph_code=True)
    torch.compile(f)(torch.randn(4, 4, device="cuda"), torch.randn(4, 4, device="cuda"), torch.randn(9, device="cuda"))
    torch._logging.set_logs(graph_code=False)

    print("\n=== empty graph is skipped ===")
    compile_count = 0

    def counting_backend(gm, inputs):
        nonlocal compile_count
        compile_count += 1
        return gm.forward

    def no_tensor(x):
        return len([1, 2, 3])

    torch.compile(no_tensor, backend=counting_backend)(torch.randn(4))
    print("backend called:", compile_count, "times")

    print("\n=== install_global ===")

    def g(x):
        return x + 1

    cg = torch.compile(g)
    cg(torch.randn(4, device="cuda"))
    names = [k for k in g.__globals__ if k.startswith("__compiled_fn")]
    print("in f.__globals__:", names)


@app.local_entrypoint()
def main():
    run.remote()
