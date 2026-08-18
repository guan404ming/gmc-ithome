import modal

app = modal.App("ironman-day06")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "numpy")


@app.function(image=image, gpu="L40S", timeout=600)
def run():
    import torch
    import torch._dynamo as dynamo

    print("torch", torch.__version__)
    cfg_scale = 2

    def f(x, n):
        return x * n * cfg_scale

    x = torch.randn(4, 4, device="cuda")

    print("\n=== 1. TORCH_LOGS=guards ===")
    torch._logging.set_logs(guards=True)
    g = torch.compile(f)
    g(x, 3)
    torch._logging.set_logs(guards=False)

    print("\n=== 2. what fails, what does not ===")
    torch._logging.set_logs(recompiles=True)
    g(torch.randn(4, 4, device="cuda"), 3)      # same shape/dtype, new values: no recompile
    g(torch.randn(8, 4, device="cuda"), 3)      # shape changed
    g(x, 4)                                     # baked int changed
    g(x.double(), 3)                            # dtype changed
    with torch.no_grad():
        g(x, 3)                                 # grad mode changed
    torch._logging.set_logs(recompiles=False)

    print("\n=== 3. cache entries on f.__code__ ===")
    entries = torch._C._dynamo.eval_frame._debug_get_cache_entry_list(f.__code__)
    print("entries:", len(entries))
    for i, e in enumerate(entries):
        print(f"  entry {i}: {e.guard_manager}")

    print("\n=== 4. hit different entries without recompiling ===")
    torch._logging.set_logs(recompiles=True)
    g(x, 3)
    g(x, 4)
    g(torch.randn(8, 4, device="cuda"), 3)
    torch._logging.set_logs(recompiles=False)
    print("still", len(torch._C._dynamo.eval_frame._debug_get_cache_entry_list(f.__code__)), "entries")

    print("\n=== 5. recompile_limit ===")
    print("recompile_limit =", torch._dynamo.config.recompile_limit)
    dynamo.reset()
    torch._dynamo.config.recompile_limit = 2

    def h(x, n):
        return x * n

    hh = torch.compile(h)
    torch._logging.set_logs(recompiles=True)
    for n in range(4):
        hh(x, n)
    torch._logging.set_logs(recompiles=False)
    torch._dynamo.config.recompile_limit = 8


@app.local_entrypoint()
def main():
    run.remote()
