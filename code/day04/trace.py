import modal

app = modal.App("ironman-day04")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "numpy")


@app.function(image=image, gpu="L40S", timeout=600)
def run():
    import dis

    import torch
    from torch._dynamo.symbolic_convert import InstructionTranslator, InstructionTranslatorBase

    print("torch", torch.__version__)

    def f(x, y):
        z = x * y
        return z + 1

    print("\n=== dis.dis(f) ===")
    dis.dis(f)

    print("\n=== dispatch_table ===")
    table = InstructionTranslator.dispatch_table
    print("entries:", len(table), "| handlers:", sum(h is not None for h in table))
    for name in ["LOAD_FAST", "STORE_FAST", "LOAD_CONST", "BINARY_OP", "CALL", "RETURN_VALUE", "POP_JUMP_IF_FALSE"]:
        h = table[dis.opmap[name]]
        print(f"  {name:18s} -> {getattr(h, '__qualname__', h)}")

    print("\n=== TORCH_LOGS=trace_bytecode ===")
    torch._logging.set_logs(trace_bytecode=True, graph_code=True)
    x = torch.randn(8, device="cuda")
    y = torch.randn(8, device="cuda")
    torch.compile(f)(x, y)

    print("\n=== python-only work is absorbed ===")
    torch._logging.set_logs(trace_bytecode=False, graph_code=True)

    def g(x):
        scale = 2
        parts = []
        for i in range(3):
            parts.append(x * (scale + i))
        return sum(parts)

    torch.compile(g)(x)


@app.local_entrypoint()
def main():
    run.remote()
