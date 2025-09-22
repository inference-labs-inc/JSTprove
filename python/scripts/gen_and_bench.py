# python\scripts\gen_and_bench.py
# ruff: noqa: S603
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- helpers for planning ----------


def _max_pools_allowed(input_hw: int, stop_at_hw: int) -> int:
    """
    Given input spatial size H=W=input_hw, how many 2x2 stride-2 pools
    can we apply before H would drop below stop_at_hw?
    """
    if input_hw <= 0 or stop_at_hw <= 0:
        return 0
    pools = 0
    h = input_hw
    while h >= 2 and (h // 2) >= stop_at_hw:
        pools += 1
        h //= 2
    return pools


def plan_for_depth(
    d: int,
    *,
    input_hw: int = 28,
    base_fc: int = 1,
    pool_cap: Optional[int] = None,
    stop_at_hw: Optional[int] = 7,
) -> list[str]:
    """
    Build a plan of length `d` conv blocks.
      - For the first K blocks (K = min(d, allowed_pools)), use: conv -> relu -> maxpool2d_k2_s2
      - For the remaining blocks (if any), use:              conv -> relu    (no further pooling)
      - Then add: reshape -> (fc1 -> relu) x base_fc -> final

    Pooling policy:
      - If pool_cap is provided, cap pooling at that many blocks (independent of input size).
      - Else, if stop_at_hw is provided, allow pooling while H >= stop_at_hw.
      - Else, fallback to floor(log2(H)) legacy behavior (pool down to 1×1).
    """
    if pool_cap is not None:
        allowed_pools = max(0, int(pool_cap))
    elif stop_at_hw is not None:
        allowed_pools = _max_pools_allowed(input_hw, stop_at_hw)
    else:
        allowed_pools = int(math.log2(max(1, input_hw)))

    pools = min(d, allowed_pools)
    conv_only = max(0, d - pools)

    plan: list[str] = []
    # pool-enabled blocks
    for i in range(1, pools + 1):
        plan += [f"conv{i}", "relu", "maxpool2d_k2_s2"]
    # conv-only blocks
    for j in range(pools + 1, pools + conv_only + 1):
        plan += [f"conv{j}", "relu"]

    # FC tail
    plan += ["reshape"]
    for k in range(1, base_fc + 1):
        plan += [f"fc{k}", "relu"]
    plan += ["final"]
    return plan


def count_layers(plan: Sequence[str]) -> tuple[int, int, int, int]:
    """Return (C, P, F, R) counts from the symbolic plan."""
    c = sum(1 for t in plan if t.startswith("conv"))
    p = sum(1 for t in plan if t.startswith("maxpool"))
    f = sum(1 for t in plan if t.startswith("fc"))
    r = sum(1 for t in plan if t == "relu")
    return c, p, f, r


# ---------- Torch model that consumes the plan ----------


class CNNDemo(nn.Module):
    def __init__(
        self,
        layers: Sequence[str],
        *,
        in_ch: int = 4,
        conv_out_ch: int = 16,
        conv_kernel: int = 3,
        conv_stride: int = 1,
        conv_pad: int = 1,
        fc_hidden: int = 256,
        n_actions: int = 10,
        input_shape: Tuple[int, int, int, int] = (1, 4, 28, 28),
    ) -> None:
        super().__init__()
        self.layers_plan = list(layers)

        _, C, H, W = input_shape
        cur_c, cur_h, cur_w = C, H, W

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.pools = nn.ModuleList()

        next_fc_in = None
        for tok in self.layers_plan:
            if tok.startswith("conv"):
                conv = nn.Conv2d(
                    in_channels=cur_c,
                    out_channels=conv_out_ch,
                    kernel_size=conv_kernel,
                    stride=conv_stride,
                    padding=conv_pad,
                )
                self.convs.append(conv)
                cur_c = conv_out_ch
                cur_h = (cur_h + 2 * conv_pad - conv_kernel) // conv_stride + 1
                cur_w = (cur_w + 2 * conv_pad - conv_kernel) // conv_stride + 1
            elif tok == "relu":
                pass
            elif tok.startswith("maxpool"):
                pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                self.pools.append(pool)
                cur_h = (cur_h - 2) // 2 + 1
                cur_w = (cur_w - 2) // 2 + 1
            elif tok == "reshape":
                next_fc_in = cur_c * cur_h * cur_w
            elif tok.startswith("fc") or tok == "final":
                if next_fc_in is None:
                    next_fc_in = cur_c * cur_h * cur_w
                out_features = n_actions if tok == "final" else fc_hidden
                self.fcs.append(nn.Linear(next_fc_in, out_features))
                next_fc_in = out_features
            else:
                raise ValueError(f"Unknown token: {tok}")

        self._ci = self._fi = self._pi = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ci = self._fi = self._pi = 0
        for tok in self.layers_plan:
            if tok.startswith("conv"):
                x = self.convs[self._ci](x)
                self._ci += 1
            elif tok == "relu":
                x = F.relu(x)
            elif tok.startswith("maxpool"):
                x = self.pools[self._pi](x)
                self._pi += 1
            elif tok == "reshape":
                x = x.reshape(x.shape[0], -1)
            elif tok.startswith("fc") or tok == "final":
                x = self.fcs[self._fi](x)
                self._fi += 1
            else:
                raise ValueError(f"Unknown token: {tok}")
        return x


# ---------- export / inputs / bench ----------


def export_onnx(model: nn.Module, onnx_path: Path, input_shape=(1, 4, 28, 28)) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(*input_shape)
    torch.onnx.export(
        model,
        dummy,
        onnx_path.as_posix(),
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes=None,
    )


def write_input_json(json_path: Path, input_shape=(1, 4, 28, 28)) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    n, c, h, w = input_shape
    arr = [0.0] * (n * c * h * w)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"input": arr, "shape": [n, c, h, w]}, f)


def run_bench(
    onnx_path: Path, input_json: Path, iterations: int, results_jsonl: Path
) -> int:
    cmd = [
        "python",
        "-m",
        "python.scripts.benchmark_runner",
        "--model",
        onnx_path.as_posix(),
        "--input",
        input_json.as_posix(),
        "--iterations",
        str(iterations),
        "--output",
        results_jsonl.as_posix(),
        "--summarize",
    ]
    return subprocess.run(cmd, check=False, shell=False).returncode  # noqa: S603


# ---------- CLI / main ----------


def _parse_int_list(s: str) -> List[int]:
    # supports "28,56,84" or "28:112:28" (start:stop:step inclusive)
    s = s.strip()
    if ":" in s:
        parts = [int(x) for x in s.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError("range syntax must be start:stop[:step]")
        start, stop = parts[0], parts[1]
        step = parts[2] if len(parts) == 3 else 1
        if step == 0:
            raise ValueError("step must be nonzero")
        out = list(range(start, stop + (1 if step > 0 else -1), step))
        return [x for x in out if (x > 0)]
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Depth or breadth sweep for simple LeNet-like CNNs."
    )
    # existing depth controls (unchanged defaults)
    ap.add_argument("--depth-min", type=int, default=1)
    ap.add_argument("--depth-max", type=int, default=12)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--results", default="depth_sweep.jsonl")
    ap.add_argument("--onnx-dir", default="python/models/models_onnx/depth")
    ap.add_argument("--inputs-dir", default="python_testing/models/inputs/depth")
    ap.add_argument("--n-actions", type=int, default=10)

    # NEW: sweep mode + breadth options (do not break old usage)
    ap.add_argument(
        "--sweep",
        choices=["depth", "breadth"],
        default="depth",
        help="depth: vary number of conv blocks (default); breadth: vary input size with fixed architecture",
    )
    ap.add_argument(
        "--arch-depth",
        type=int,
        default=5,
        help="(breadth) conv blocks to use when sweeping input sizes",
    )
    ap.add_argument(
        "--input-hw",
        type=int,
        default=28,
        help="(depth) input H=W when sweeping depth (kept for backwards compatibility)",
    )
    ap.add_argument(
        "--input-hw-list",
        type=str,
        default="28,56,84,112",
        help="(breadth) comma list or start:stop[:step], e.g. '28,56,84' or '32:160:32'",
    )
    ap.add_argument(
        "--pool-cap",
        type=int,
        default=2,
        help="cap the number of maxpool blocks at the start (Lenet-like: 2)",
    )
    ap.add_argument(
        "--stop-at-hw",
        type=int,
        default=None,
        help="if set (and pool-cap not set), allow pooling while H >= stop_at_hw",
    )
    ap.add_argument("--conv-out-ch", type=int, default=16)
    ap.add_argument("--fc-hidden", type=int, default=256)
    ap.add_argument("--tag", type=str, default="", help="optional tag in filenames")

    args = ap.parse_args()

    onnx_dir = Path(args.onnx_dir)
    in_dir = Path(args.inputs_dir)
    results = Path(args.results)

    if args.sweep == "depth":
        # legacy behavior preserved (now you can change input-hw too)
        input_shape = (1, 4, args.input_hw, args.input_hw)
        for d in range(args.depth_min, args.depth_max + 1):
            plan = plan_for_depth(
                d=d,
                input_hw=args.input_hw,
                base_fc=1,
                pool_cap=args.pool_cap,
                stop_at_hw=args.stop_at_hw,
            )
            C, P, F, R = count_layers(plan)
            uid = f"depth_d{d}_c{C}_p{P}_f{F}_r{R}"
            if args.tag:
                uid = f"{uid}_{args.tag}"

            onnx_path = onnx_dir / f"{uid}.onnx"
            input_json = in_dir / f"{uid}_input.json"

            model = CNNDemo(
                plan,
                input_shape=input_shape,
                n_actions=args.n_actions,
                conv_out_ch=args.conv_out_ch,
                fc_hidden=args.fc_hidden,
            )
            export_onnx(model, onnx_path, input_shape=input_shape)
            write_input_json(input_json, input_shape=input_shape)
            print(f"[gen] d={d} :: C={C}, P={P}, F={F}, R={R} -> {onnx_path.name}")

            rc = run_bench(onnx_path, input_json, args.iterations, results)
            if rc != 0:
                print(f"[warn] benchmark rc={rc} for depth={d}")

    else:
        # breadth sweep: fix architecture depth; vary input size(s)
        sizes = _parse_int_list(args.input_hw_list)
        d = int(args.arch_depth)
        for hw in sizes:
            input_shape = (1, 4, hw, hw)
            plan = plan_for_depth(
                d=d,
                input_hw=hw,
                base_fc=1,
                pool_cap=args.pool_cap,  # enforce Lenet-like initial pooling
                stop_at_hw=args.stop_at_hw,  # or use stop_at_hw if you prefer
            )
            C, P, F, R = count_layers(plan)
            uid = f"breadth_h{hw}_d{d}_c{C}_p{P}_f{F}_r{R}"
            if args.tag:
                uid = f"{uid}_{args.tag}"

            onnx_path = onnx_dir / f"{uid}.onnx"
            input_json = in_dir / f"{uid}_input.json"

            model = CNNDemo(
                plan,
                input_shape=input_shape,
                n_actions=args.n_actions,
                conv_out_ch=args.conv_out_ch,
                fc_hidden=args.fc_hidden,
            )
            export_onnx(model, onnx_path, input_shape=input_shape)
            write_input_json(input_json, input_shape=input_shape)
            print(
                f"[gen] H=W={hw} :: d={d} | C={C}, P={P}, F={F}, R={R} -> {onnx_path.name}"
            )

            rc = run_bench(onnx_path, input_json, args.iterations, results)
            if rc != 0:
                print(f"[warn] benchmark rc={rc} for hw={hw}")


if __name__ == "__main__":
    main()
