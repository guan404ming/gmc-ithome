import modal

app = modal.App("ironman-day14")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "numpy")


@app.function(image=image, gpu="L40S", timeout=600)
def run():
    import torch

    print("torch", torch.__version__)
    print("aten ops registered:", len(list(torch.ops.aten)))
    from torch._decomp import decomposition_table
    print("entries in torch._decomp.decomposition_table:", len(decomposition_table))
    from torch._inductor.decomposition import decompositions as ind
    print("inductor decomposition table:", len(ind))

    print("\n=== gelu + layer_norm through aot_graphs ===")

    ln = torch.nn.LayerNorm(8).cuda()

    def f(x):
        return torch.nn.functional.gelu(ln(x))

    torch._logging.set_logs(aot_graphs=True)
    with torch.no_grad():
        torch.compile(f)(torch.randn(4, 8, device="cuda"))
    torch._logging.set_logs(aot_graphs=False)

    print("\n=== post-grad graph (after inductor decomps) ===")
    torch._dynamo.reset()
    torch._logging.set_logs(post_grad_graphs=True)
    with torch.no_grad():
        torch.compile(f)(torch.randn(4, 8, device="cuda"))


@app.local_entrypoint()
def main():
    run.remote()
