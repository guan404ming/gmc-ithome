import modal

app = modal.App("ironman-day09")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "numpy")


@app.function(image=image, gpu="L40S", timeout=600)
def run():
    import torch

    print("torch", torch.__version__)

    def f(x, n):
        return x * n + 1

    torch._logging.set_logs(bytecode=True)
    torch.compile(f)(torch.randn(4, device="cuda"), 3)
    torch._logging.set_logs(bytecode=False)

    print("\n=== instruction objects: jump targets are references ===")
    import dis

    from torch._dynamo.bytecode_transformation import cleaned_instructions

    def g(x):
        if x is None:
            return 1
        return 2

    insts = cleaned_instructions(g.__code__)
    for i in insts[:8]:
        tgt = f" -> target={i.target.opname}@{i.target.offset}" if i.target else ""
        print(f"  {i.opname:22s} arg={i.arg}{tgt}")


@app.local_entrypoint()
def main():
    run.remote()
