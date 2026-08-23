import modal

app = modal.App("ironman-day21")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "numpy")


@app.function(image=image, gpu="L40S", timeout=900)
def run():
    import gc
    import math
    import torch
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))

    torch._logging.set_logs(output_code=True)

    def pointwise(x, y):
        return torch.relu(x + y) * 2

    def rowsum(x):
        return torch.relu(x + 1).sum(dim=1)

    def total(x):
        return x.sum()

    print("\n=== case 1: pointwise relu(x + y) * 2, shape (1000000,) ===")
    x1 = torch.randn(1000000, device="cuda")
    y1 = torch.randn(1000000, device="cuda")
    torch.compile(pointwise)(x1, y1)

    print("\n=== case 2: rowsum relu(x + 1).sum(dim=1), shape (1024, 1024) ===")
    torch._dynamo.reset()
    x2 = torch.randn(1024, 1024, device="cuda")
    torch.compile(rowsum)(x2)

    print("\n=== case 3: global sum x.sum(), shape (4096, 4096) ===")
    torch._dynamo.reset()
    x3 = torch.randn(4096, 4096, device="cuda")
    torch.compile(total)(x3)

    print("\n=== compiled kernel launch configs ===")
    for obj in gc.get_objects():
        if not isinstance(obj, CachingAutotuner):
            continue
        name = obj.inductor_meta.get("kernel_name", "?")
        for cfg in obj.configs or []:
            print(f"{name}: candidate {cfg.kwargs} num_warps={cfg.num_warps} num_stages={cfg.num_stages}")
        launchers = getattr(obj, "launchers", None)
        best = getattr(launchers[0], "config", None) if launchers else None
        if best is not None:
            print(f"{name}: picked {best.kwargs} num_warps={best.num_warps} num_stages={best.num_stages}")
            xblock = best.kwargs.get("XBLOCK")
            if name.startswith("triton_poi") and xblock:
                print(f"{name}: grid = (ceil(1000000 / {xblock}), 1, 1) = ({math.ceil(1000000 / xblock)}, 1, 1)")


@app.local_entrypoint()
def main():
    run.remote()
