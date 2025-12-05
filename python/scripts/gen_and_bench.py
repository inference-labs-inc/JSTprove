# python/scripts/gen_and_bench.py
# ruff: noqa: S603, T201, RUF002

"""
Generate simple CNN ONNX models and benchmark JSTprove.

Typical usages:

Depth sweep (vary conv depth, fixed input size):
    jst bench \
      --sweep depth \
      --depth-min 1 \
      --depth-max 16 \
      --input-hw 56 \
      --iterations 3 \
      --results benchmarking/depth_sweep.jsonl

Breadth sweep (vary input resolution, fixed conv depth):
    jst bench \
      --sweep breadth \
      --arch-depth 5 \
      --input-hw-list 28,56,84,112 \
      --iterations 3 \
      --results benchmarking/breadth_sweep.jsonl \
      --pool-cap 2 --conv-out-ch 16 --fc-hidden 256

Output locations (unless overridden via --onnx-dir / --inputs-dir):
    depth   → python/models/models_onnx/depth   ; python/models/inputs/depth
    breadth → python/models/models_onnx/breadth ; python/models/inputs/breadth

Each model is benchmarked by python.scripts.benchmark_runner and a row is
appended to the JSONL file passed via --results.
"""

from __future__ import annotations

# --- Standard library --------------------------------------------------------
import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


# --- Third-party -------------------------------------------------------------
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

# -----------------------------------------------------------------------------
# Planning helpers
# -----------------------------------------------------------------------------


def _max_pools_allowed(input_hw: int, stop_at_hw: int) -> int:
    """
    Given an input size H=W=input_hw, return how many 2×2/stride-2 pools
    can be applied while keeping H >= stop_at_hw.
    """
    two = 2
    if input_hw <= 0 or stop_at_hw <= 0:
        return 0
    pools = 0
    h = input_hw
    while h >= two and (h // two) >= stop_at_hw:
        pools += 1
        h //= two
    return pools


def plan_for_depth(
    d: int,
    *,
    input_hw: int = 56,
    base_fc: int = 1,
    pool_cap: int | None = None,
    stop_at_hw: int | None = 7,
) -> list[str]:
    """
    Build a symbolic plan for `d` conv blocks followed by FC layers.

    Layout:
      - First K blocks: conv → relu → maxpool2d_k2_s2  (K = min(d, allowed_pools))
      - Remaining blocks: conv → relu
      - Tail: reshape → (fc → relu) × base_fc → final

    Pooling policy:
      - If pool_cap is provided, cap pooling at that many initial blocks.
      - Else if stop_at_hw is provided, allow pooling while H >= stop_at_hw.
      - Else fall back to floor(log2(H)).
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
    # pooled blocks
    for i in range(1, pools + 1):
        plan += [f"conv{i}", "relu", "maxpool2d_k2_s2"]
    # non-pooled blocks
    for j in range(pools + 1, pools + conv_only + 1):
        plan += [f"conv{j}", "relu"]

    # FC tail
    plan += ["reshape"]
    for k in range(1, base_fc + 1):
        plan += [f"fc{k}", "relu"]
    plan += ["final"]
    return plan


def count_layers(plan: Sequence[str]) -> tuple[int, int, int, int]:
    """Return a summary tuple (num_convs, num_pools, num_fcs, num_relus)."""
    c = sum(1 for t in plan if t.startswith("conv"))
    p = sum(1 for t in plan if t.startswith("maxpool"))
    f = sum(1 for t in plan if t.startswith("fc"))
    r = sum(1 for t in plan if t == "relu")
    return c, p, f, r


# -----------------------------------------------------------------------------
# Torch model that consumes the plan
# -----------------------------------------------------------------------------


class CNNDemo(nn.Module):
    """
    Minimal CNN whose structure is defined by a symbolic plan.
    Uses fixed conv hyperparameters and a configurable FC head.
    """

    def __init__(  # noqa: PLR0913
        self: CNNDemo,
        layers: Sequence[str],
        *,
        in_ch: int = 4,
        conv_out_ch: int = 16,
        conv_kernel: int = 3,
        conv_stride: int = 1,
        conv_pad: int = 1,
        fc_hidden: int = 128,
        n_actions: int = 10,
        input_shape: tuple[int, int, int, int] = (1, 4, 56, 56),
    ) -> None:
        super().__init__()
        self.layers_plan = list(layers)
        _ = in_ch
        _, C, H, W = input_shape  # noqa: N806
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
                msg = f"Unknown token: {tok}"
                raise ValueError(msg)

        self._ci = self._fi = self._pi = 0

    def forward(self: CNNDemo, x: torch.Tensor) -> torch.Tensor:
        """Execute the plan in order."""
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
                msg = f"Unknown token: {tok}"
                raise ValueError(msg)
        return x


# -----------------------------------------------------------------------------
# Export / inputs / benchmark shim
# -----------------------------------------------------------------------------


def export_onnx(
    model: nn.Module,
    onnx_path: Path,
    input_shape: tuple[int] = (1, 4, 56, 56),
) -> None:
    """Export a Torch model to ONNX and ensure the directory exists."""
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


def write_input_json(json_path: Path, input_shape: tuple[int] = (1, 4, 28, 28)) -> None:
    """Write a zero-valued input tensor to JSON without shape information."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    n, c, h, w = input_shape
    arr = [0.0] * (n * c * h * w)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"input": arr}, f)


def run_bench(
    onnx_path: Path,
    input_json: Path,
    iterations: int,
    results_jsonl: Path,
) -> int:
    """
    Invoke the benchmark runner module as a subprocess.
    Returns the exit code (0 on success).
    """
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
    return subprocess.run(cmd, check=False, shell=False).returncode


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_int_list(s: str) -> list[int]:
    """
    Parse either a comma list "28,56,84" or a range "start:stop[:step]".
    The range is inclusive of stop. Non-positive values are filtered out.
    """
    three = 3
    s = s.strip()
    if ":" in s:
        parts = [int(x) for x in s.split(":")]
        if len(parts) not in (2, 3):
            msg = "range syntax must be start:stop[:step]"
            raise ValueError(msg)
        start, stop = parts[0], parts[1]
        step = parts[2] if len(parts) == three else 1
        if step == 0:
            msg = "step must be nonzero"
            raise ValueError(msg)
        out = list(range(start, stop + (1 if step > 0 else -1), step))
        return [x for x in out if x > 0]
    return [int(x) for x in s.split(",") if x.strip()]


def _resolve_output_dirs(
    sweep: str,
    onnx_dir_arg: str | None,
    inputs_dir_arg: str | None,
) -> tuple[Path, Path]:
    """
    Choose output directories from the sweep type unless explicitly overridden.

    depth   → python/models/models_onnx/depth   ; python/models/inputs/depth
    breadth → python/models/models_onnx/breadth ; python/models/inputs/breadth
    """
    sub = sweep if sweep in ("depth", "breadth") else "depth"
    default_onnx = Path(f"python/models/models_onnx/{sub}")
    default_inputs = Path(f"python/models/inputs/{sub}")
    onnx_dir = Path(onnx_dir_arg) if onnx_dir_arg else default_onnx
    inputs_dir = Path(inputs_dir_arg) if inputs_dir_arg else default_inputs
    return onnx_dir, inputs_dir


def main() -> None:  # noqa: PLR0915
    """Argument parsing and sweep orchestration."""
    ap = argparse.ArgumentParser(
        description="Depth or breadth sweep for simple LeNet-like CNNs.",
    )
    # depth controls
    ap.add_argument("--depth-min", type=int, default=1)
    ap.add_argument("--depth-max", type=int, default=12)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--results", default=None)
    ap.add_argument("--onnx-dir", default=None)
    ap.add_argument("--inputs-dir", default=None)
    ap.add_argument("--n-actions", type=int, default=10)

    # sweep mode + breadth options
    ap.add_argument(
        "--sweep",
        choices=["depth", "breadth"],
        default="depth",
        help="depth: vary number of conv blocks; "
        "breadth: vary input size at fixed depth",
    )
    ap.add_argument(
        "--arch-depth",
        type=int,
        default=5,
        help="(breadth) conv blocks used for all inputs",
    )
    ap.add_argument(
        "--input-hw",
        type=int,
        default=56,
        help="(depth) input H=W when varying depth",
    )
    ap.add_argument(
        "--input-hw-list",
        type=str,
        default="28,56,84,112",
        help="(breadth) comma list or start:stop[:step], e.g. "
        "'28,56,84' or '32:160:32'",
    )
    ap.add_argument(
        "--pool-cap",
        type=int,
        default=2,
        help="cap the number of initial maxpool blocks",
    )
    ap.add_argument(
        "--stop-at-hw",
        type=int,
        default=None,
        help="allow pooling while H >= this (if pool-cap unset)",
    )
    ap.add_argument("--conv-out-ch", type=int, default=16)
    ap.add_argument("--fc-hidden", type=int, default=128)
    ap.add_argument(
        "--tag",
        type=str,
        default="",
        help="optional tag added to filenames",
    )

    args = ap.parse_args()

    # Ensure output dirs and a robust default results path
    onnx_dir, in_dir = _resolve_output_dirs(args.sweep, args.onnx_dir, args.inputs_dir)

    # Dynamic default for results (also handles empty string)
    if args.results is None or not str(args.results).strip():
        # default results path based on sweep type
        results = Path("benchmarking") / f"{args.sweep}_sweep.jsonl"
    else:
        results = Path(args.results)

    results.parent.mkdir(parents=True, exist_ok=True)

    onnx_dir, in_dir = _resolve_output_dirs(args.sweep, args.onnx_dir, args.inputs_dir)

    if args.sweep == "depth":
        input_shape = (1, 4, args.input_hw, args.input_hw)
        for d in range(args.depth_min, args.depth_max + 1):
            plan = plan_for_depth(
                d=d,
                input_hw=args.input_hw,
                base_fc=1,
                pool_cap=args.pool_cap,
                stop_at_hw=args.stop_at_hw,
            )
            C, P, Fc, R = count_layers(plan)  # noqa: N806
            uid = f"depth_d{d}_c{C}_p{P}_f{Fc}_r{R}"
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
            print(f"[gen] d={d} :: C={C}, P={P}, F={Fc}, R={R} -> {onnx_path.name}")

            rc = run_bench(onnx_path, input_json, args.iterations, results)
            if rc != 0:
                print(f"[warn] benchmark rc={rc} for depth={d}")
    else:
        # breadth sweep: fixed architecture depth; vary input sizes
        sizes = _parse_int_list(args.input_hw_list)
        d = int(args.arch_depth)
        for hw in sizes:
            input_shape = (1, 4, hw, hw)
            plan = plan_for_depth(
                d=d,
                input_hw=hw,
                base_fc=1,
                pool_cap=args.pool_cap,
                stop_at_hw=args.stop_at_hw,
            )
            C, P, Fc, R = count_layers(plan)  # noqa: N806
            uid = f"breadth_h{hw}_d{d}_c{C}_p{P}_f{Fc}_r{R}"
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
                f"[gen] H=W={hw} :: d={d} | C={C}, P={P}, F={Fc}, R={R} "
                f"-> {onnx_path.name}",
            )

            rc = run_bench(onnx_path, input_json, args.iterations, results)
            if rc != 0:
                print(f"[warn] benchmark rc={rc} for hw={hw}")


if __name__ == "__main__":
    main()
