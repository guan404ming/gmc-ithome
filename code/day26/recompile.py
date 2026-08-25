import time

import torch
import torch._dynamo

step = 0


def poly(x):
    return ((x * step).sin() + x.cos() * 0.5).relu().sum()


def poly_t(x, s):
    return ((x * s).sin() + x.cos() * 0.5).relu().sum()


def scale(x, s):
    return ((x * s).sin() + x.cos()).sum()


def net(x):
    return (x @ x.T).relu().sum()


def apply_fn(x, fn):
    return fn(x)


def graphs():
    return torch._dynamo.utils.counters["stats"]["unique_graphs"]


def section(name):
    torch._dynamo.reset()
    print(f"\n[{name}]", flush=True)
    return graphs()


def done(base):
    print(f"  graphs compiled this section: {graphs() - base}", flush=True)


def main():
    global step
    x = torch.randn(64, 64)

    base = section("case 1: global constant changes every call")
    cp = torch.compile(poly)
    times = []
    for i in range(10):
        step = i
        t0 = time.perf_counter()
        cp(x)
        times.append(time.perf_counter() - t0)
    for i, t in enumerate(times):
        print(f"  call {i} (step={i}): {t:7.3f} s", flush=True)
    print(f"  first 8 calls total: {sum(times[:8]):.2f} s", flush=True)
    done(base)
    step = 999
    t0 = time.perf_counter()
    r = cp(x)
    dt = (time.perf_counter() - t0) * 1000
    print(f"  step=999 after limit: {dt:.2f} ms, matches eager: {torch.allclose(r, poly(x))}", flush=True)

    base = section("case 2: same constant passed as argument")
    cs = torch.compile(scale)
    for i in range(10):
        cs(x, float(i))
    done(base)

    base = section("case 3a: batch size jumps")
    cn = torch.compile(net)
    for b in [8, 16, 24, 40, 56, 96]:
        cn(torch.randn(b, 32))
    done(base)

    base = section("case 3b: rank jumps")
    cf = torch.compile(poly_t)
    v = torch.randn(16)
    s = torch.tensor(2.0)
    for shp in [(16,), (4, 4), (2, 2, 4), (2, 2, 2, 2)]:
        cf(v.view(shp), s)
    done(base)

    base = section("case 3c: batch size 1 in the mix")
    cn1 = torch.compile(net)
    for b in [8, 16, 1, 24, 1, 40]:
        cn1(torch.randn(b, 32))
    done(base)

    base = section("case 4: grad mode flips")
    cg = torch.compile(scale)
    xr = torch.randn(64, 64, requires_grad=True)
    for i in range(4):
        if i % 2 == 0:
            with torch.no_grad():
                cg(xr, 1.0)
        else:
            cg(xr, 1.0)
    done(base)

    base = section("case 5: a different function object every call")
    ca = torch.compile(apply_fn)
    for fn in [lambda t: t + 1, lambda t: t * 2, lambda t: t.sin(), lambda t: t.relu()]:
        ca(x, fn)
    done(base)

    base = section("fix 1: move the constant into a tensor")
    ct = torch.compile(poly_t)
    times = []
    for i in range(10):
        t0 = time.perf_counter()
        ct(x, torch.tensor(float(i)))
        times.append(time.perf_counter() - t0)
    print(f"  call 0: {times[0]:.3f} s, calls 1-9 max: {max(times[1:]) * 1000:.2f} ms", flush=True)
    done(base)

    base = section("fix 2: mark_dynamic on the batch dim")
    cm = torch.compile(net)
    for b in [8, 16, 24, 40, 56, 96]:
        t = torch.randn(b, 32)
        torch._dynamo.mark_dynamic(t, 0)
        cm(t)
    done(base)

    base = section("fix 3: dynamic=True")
    cd = torch.compile(net, dynamic=True)
    for b in [8, 16, 24, 40, 56, 96]:
        cd(torch.randn(b, 32))
    done(base)


if __name__ == "__main__":
    main()
