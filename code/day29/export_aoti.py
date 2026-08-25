import os
import zipfile

import torch
from torch.export import Dim, export


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 8)

    def forward(self, x):
        return torch.relu(self.fc(x)) + 1


class Branchy(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x + 1
        return x - 1


def main():
    torch.manual_seed(0)
    m = Tiny().eval()
    x = torch.randn(4, 16)

    batch = Dim("batch", min=1, max=1024)
    ep = export(m, (x,), dynamic_shapes={"x": {0: batch}})
    print("[exported program]")
    print(ep)

    print("[torch.compile on data-dependent branch]")
    exp = torch._dynamo.explain(Branchy())(x)
    print(f"  graph_count={exp.graph_count} graph_break_count={exp.graph_break_count}")
    for r in exp.break_reasons:
        print(f"  break reason: {r.reason}")

    print("[torch.export on the same module]")
    try:
        export(Branchy(), (x,))
    except Exception as e:
        msg = " ".join(str(e).split())
        print(f"  {type(e).__name__}: {msg[:220]}")

    pt2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny.pt2")
    torch._inductor.aoti_compile_and_package(ep, package_path=pt2)
    print("[pt2 package contents]")
    with zipfile.ZipFile(pt2) as z:
        for info in z.infolist():
            if not info.is_dir():
                print(f"  {info.filename}  ({info.file_size / 1024:.0f} KB)")

    print("[load and run]")
    runner = torch._inductor.aoti_load_package(pt2)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    y_aoti = runner(x)
    y_eager = m(x)
    print(f"  allclose(aoti, eager) = {torch.allclose(y_aoti, y_eager, atol=1e-5)}")
    big = torch.randn(512, 16)
    print(f"  batch=512 output shape = {tuple(runner(big).shape)}")
    frames = sum(torch._dynamo.utils.counters["frames"].values())
    print(f"  dynamo frames traced while running .pt2 = {frames}")

    cf = torch.compile(m)
    cf(x)
    frames = sum(torch._dynamo.utils.counters["frames"].values())
    print(f"  dynamo frames traced after one torch.compile call = {frames}")


if __name__ == "__main__":
    main()
