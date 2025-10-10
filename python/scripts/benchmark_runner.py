# python/scripts/benchmark_runner.py
# ruff: noqa: S603, RUF003, RUF002, T201

"""
Benchmark JSTProve by invoking the CLI phases (compile → witness → prove → verify),
streaming live output with a spinner/HUD, and logging per-phase timing/memory.

Writes one JSON object per phase per iteration to a JSONL file so you can analyze later.
"""

from __future__ import annotations

# --- Standard library --------------------------------------------------------
import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

# --- Third-party -------------------------------------------------------------
import psutil

# --- Local -------------------------------------------------------------------
from python.core.utils.benchmarking_helpers import (
    end_memory_collection,
    start_memory_collection,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("benchmark_runner")

# -----------------------------------------------------------------------------
# Parsing helpers / patterns
# -----------------------------------------------------------------------------
TIME_PATTERNS = [
    re.compile(r"Rust time taken:\s*([0-9.]+)"),
    re.compile(r"Time elapsed:\s*([0-9.]+)\s*seconds"),
]
MEM_PATTERNS = [
    re.compile(r"Peak Memory used Overall\s*:\s*([0-9.]+)"),
    re.compile(r"Rust subprocess memory\s*:\s*([0-9.]+)"),
]

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
ECC_HINT_RE = re.compile(r"built\s+hint\s+normalized\s+ir\b.*", re.IGNORECASE)
ECC_LAYERED_RE = re.compile(r"built\s+layered\s+circuit\b.*", re.IGNORECASE)
ECC_LINE_PATTERNS = [
    re.compile(r"built layered circuit\b.*", re.IGNORECASE),
    re.compile(r"built hint normalized ir\b.*", re.IGNORECASE),
]

ECC_KEYS = {
    "numInputs",
    "numConstraints",
    "numInsns",
    "numVars",
    "numTerms",
    "numSegment",
    "numLayer",
    "numUsedInputs",
    "numUsedVariables",
    "numVariables",
    "numAdd",
    "numCst",
    "numMul",
    "totalCost",
}

# Accept optional spaces and thousands separators in values
KV_PAIR = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9,]+)\b")

# Spinner glyphs (ASCII by default; set JSTPROVE_UNICODE=1 to switch)
_SPINNER_ASCII = "-\\|/"
_SPINNER_UNICODE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _term_width(default: int = 100) -> int:
    """Best-effort terminal width (columns)."""
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default


# NOTE: currently unused; retained only because the (deprecated) compile card used it.
def _human_bytes(n: int | None) -> str:
    """Pretty-print bytes as B/KB/MB/GB/TB (unused in the current flow)."""
    conversion_value = 1024.0
    if n is None:
        return "NA"
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < conversion_value or u == units[-1]:
            return f"{x:.1f} {u}" if u != "B" else f"{int(x)} B"
        x /= conversion_value
    msg = "Unreachable code: failed to format byte size."
    raise RuntimeError(msg)


# NOTE: currently unused; retained only because the (deprecated) compile card used it.
def _fmt_int(n: int | None) -> str:
    """Format an int with thousands separators (unused in the current flow)."""
    return f"{n:,}" if isinstance(n, int) else "NA"


def _bar(value: int, vmax: int, width: int = 24, char: str = "█") -> str:
    """Fixed-width bar proportional to value/vmax, using a solid block character."""
    if vmax <= 0 or value <= 0:
        return " " * width
    fill = max(1, int(width * min(value, vmax) / vmax))
    return char * fill + " " * (width - fill)


def _marquee(t: float, width: int = 24, char: str = "█") -> str:
    """Bouncing 8-char block to suggest activity when total work is unknown."""
    w = max(8, min(width, 24))
    pos = int((abs(((t * 0.8) % 2) - 1)) * (w - 8))
    return " " * pos + char * 8 + " " * (w - 8 - pos)


def _sum_child_rss_mb(parent_pid: int) -> float:
    """Approximate current total RSS of all child processes, in MB."""
    try:
        parent = psutil.Process(parent_pid)
    except psutil.Error:
        return 0.0
    total = 0
    for c in parent.children(recursive=True):
        with suppress(psutil.Error):
            total += c.memory_info().rss
    return total / (1024.0 * 1024.0)


def parse_ecc_stats(text: str) -> dict[str, int]:
    """Scan the whole blob for k=v pairs and keep only ECC keys."""
    clean = ANSI_RE.sub("", text).replace("\r", "\n")
    stats: dict[str, int] = {}
    for k, v in KV_PAIR.findall(clean):
        if k in ECC_KEYS:
            with suppress(ValueError):
                stats[k] = int(v.replace(",", ""))
    return stats


def strip_ansi(s: str) -> str:
    """Remove ANSI color/escape sequences from a string."""
    return ANSI_RE.sub("", s)


def count_onnx_parameters(model_path: Path) -> int:
    """
    Sum element counts of ONNX initializers (trainable weights).
    Returns -1 if the `onnx` dependency is unavailable.
    """
    try:
        import onnx  # noqa: PLC0415 # type: ignore[import]
    except Exception:
        return -1

    model = onnx.load(str(model_path))
    total = 0
    for init in model.graph.initializer:
        n = 1
        for d in init.dims:
            n *= int(d)
        total += n
    return int(total)


def file_size_bytes(path: str | Path) -> int | None:
    """Return file size in bytes (or None if the path does not exist)."""
    try:
        p = Path(path)
        return p.stat().st_size if p.exists() else None
    except OSError:
        return None


def parse_metrics(text: str) -> tuple[float | None, float | None]:
    """Best-effort parse for time (seconds) and memory (MB) from CLI output."""
    time_s: float | None = None
    mem_mb: float | None = None
    for pat in TIME_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                time_s = float(m.group(1))
                break
            except ValueError:
                pass
    for pat in MEM_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                mem_mb = float(m.group(1))
                break
            except ValueError:
                pass
    return time_s, mem_mb


def now_utc() -> str:
    """
    UTC timestamp in RFC3339 format without subseconds
    (e.g., '2025-01-01T00:00:00Z').
    """
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class PhaseIO:
    """Per-phase file locations used to invoke the CLI."""

    model_path: Path
    circuit_path: Path
    input_path: Path | None
    output_path: Path
    witness_path: Path
    proof_path: Path


def _build_phase_cmd(phase: str, io: PhaseIO) -> list[str]:
    """Construct the exact `jst` CLI command for a phase."""
    base = ["jst", "--no-banner"]
    if phase == "compile":
        return [*base, "compile", "-m", str(io.model_path), "-c", str(io.circuit_path)]
    if phase == "witness":
        cmd = [
            *base,
            "witness",
            "-c",
            str(io.circuit_path),
            "-o",
            str(io.output_path),
            "-w",
            str(io.witness_path),
        ]
        if io.input_path:
            cmd += ["-i", str(io.input_path)]
        return cmd
    if phase == "prove":
        return [
            *base,
            "prove",
            "-c",
            str(io.circuit_path),
            "-w",
            str(io.witness_path),
            "-p",
            str(io.proof_path),
        ]
    if phase == "verify":
        cmd = [
            *base,
            "verify",
            "-c",
            str(io.circuit_path),
            "-o",
            str(io.output_path),
            "-w",
            str(io.witness_path),
            "-p",
            str(io.proof_path),
        ]
        if io.input_path:
            cmd += ["-i", str(io.input_path)]
        return cmd
    msg = f"unknown phase: {phase}"
    raise ValueError(msg)


def run_cli(  # noqa: PLR0915, PLR0912, C901
    phase: str,
    io: PhaseIO,
) -> tuple[
    int,
    str,
    float | None,
    float | None,
    list[str],
    float | None,
    float | None,
    dict[str, int],
]:
    """
    Execute one CLI phase, streaming stdout live with a spinner/HUD and
    tracking peak RSS via psutil.

    Returns:
        (returncode, combined_output, time_s, mem_mb_primary, cmd_list,
         mem_mb_rust, mem_mb_psutil, ecc_dict)
    """
    cmd = _build_phase_cmd(phase, io)

    env = os.environ.copy()
    env.setdefault("RUST_LOG", "info")
    env.setdefault("RUST_BACKTRACE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")  # help with child buffering

    stop_ev, mon_thread, mon_results = start_memory_collection("")
    start = time.time()
    combined_lines: list[str] = []
    ecc_live: dict[str, int] = {}

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )

    spinner = (
        _SPINNER_UNICODE
        if os.environ.get("JSTPROVE_UNICODE") == "1"
        else _SPINNER_ASCII
    )
    sp_i = 0
    peak_live_mb = 0.0
    tw = _term_width()
    bar_w = max(18, min(28, tw - 50))

    try:
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            elapsed = time.time() - start

            # live peak memory
            live_mb = _sum_child_rss_mb(proc.pid)
            peak_live_mb = max(peak_live_mb, live_mb)

            if line:
                combined_lines.append(line.rstrip("\n"))
                low = line.lower()

                # Echo milestone lines for user feedback (unchanged behavior)
                if ("built layered circuit" in low) or (
                    "built hint normalized ir" in low
                ):
                    print(line, end="")

                # Harvest ECC counters live from any k=v pairs we see
                for k, v in KV_PAIR.findall(ANSI_RE.sub("", line)):
                    if k in ECC_KEYS:
                        with suppress(ValueError):
                            ecc_live[k] = int(v.replace(",", ""))

            # refresh HUD ~10Hz
            if int(elapsed * 10) != int((elapsed - 0.09) * 10) or not line:
                spin = spinner[sp_i % len(spinner)]
                sp_i += 1
                hud_bar = _marquee(elapsed, width=bar_w)
                hud = (
                    f"\r[{spin}] {phase:<7} | {elapsed:6.1f}s | "
                    f"mem↑ {peak_live_mb:7.1f} MB | {hud_bar}"
                )
                print(hud[: tw - 1], end="", flush=True)

            if proc.poll() is not None:
                # Drain any remaining buffered output after the process exits.
                if proc.stdout:
                    tail = proc.stdout.read()
                    if tail:
                        combined_lines.extend(tail.splitlines())

                # final HUD line
                elapsed = time.time() - start
                hud = (
                    f"\r[✔] {phase:<7} | {elapsed:6.1f}s "
                    f"| mem↑ {peak_live_mb:7.1f} MB | " + " " * bar_w
                )
                print(hud[: tw - 1])
                break

            time.sleep(0.09)

    finally:
        # collect psutil-based peak (may differ from live sampler)
        collected_mb: float | None = None
        try:
            mem = end_memory_collection(stop_ev, mon_thread, mon_results)  # type: ignore[arg-type]
            if isinstance(mem, dict):
                collected_mb = float(mem.get("total", 0.0))
        except Exception:
            collected_mb = None

    combined = "\n".join(combined_lines)
    time_s, mem_mb_rust = parse_metrics(combined)
    if time_s is None:
        time_s = elapsed
    mem_mb_psutil = collected_mb if collected_mb is not None else peak_live_mb
    mem_mb_primary = mem_mb_psutil if mem_mb_psutil is not None else mem_mb_rust

    # If we missed something live, parse once more from the combined blob
    if not ecc_live:
        ecc_live = parse_ecc_stats(combined)

    # Also append a compact ECC block into the combined text for later eyeballing
    if ecc_live:
        kv = " ".join(f"{k}={v}" for k, v in sorted(ecc_live.items()))
        combined += f"\n[ECC]\n{kv}\n"

    return (
        proc.returncode or 0,
        combined,
        time_s,
        mem_mb_primary,
        cmd,
        mem_mb_rust,
        mem_mb_psutil,
        ecc_live,
    )


# NOTE: this card printer is currently unused (kept for reference).
def _print_compile_card(
    ecc: dict,
    circuit_bytes: int | None,
    quant_bytes: int | None,
) -> None:
    """Pretty compile stats block (unused in the current flow)."""
    _ = circuit_bytes, quant_bytes
    if not ecc:
        return
    keys = [
        "numAdd",
        "numMul",
        "numCst",
        "numVars",
        "numInsns",
        "numConstraints",
        "totalCost",
    ]
    data = {k: int(ecc[k]) for k in keys if k in ecc}
    if not data:
        return
    vmax = max(data.values())
    w = max(24, min(40, _term_width() - 50))

    print()
    print(
        "┌────────────────────────── Compile Stats ──────────────────────────┐",
    )
    for k in keys:
        if k in data:
            bar = _bar(data[k], vmax, width=w)
            # _fmt_int/_human_bytes are intentionally still present
            # but currently unused elsewhere.
            print(f"│ {k:<14} {data[k]:>12}  {bar} │")
    print(
        "└────────────────────────────────────────────────────────────────────┘",
    )


def _quantized_path_from_circuit(circuit_path: Path) -> Path:
    """Derive quantized ONNX path from circuit: <dir>/<stem>_quantized_model.onnx"""
    return circuit_path.with_name(f"{circuit_path.stem}_quantized_model.onnx")


def _fmt_mean_sd(vals: list[float]) -> tuple[str, float | None, float | None]:
    """Format a list as 'μ ± σ' (or single value), returning the label and μ,σ."""
    if not vals:
        return "NA", None, None
    if len(vals) == 1:
        v = vals[0]
        return f"{v:.3f}", v, None
    mu, sd = mean(vals), stdev(vals)
    return f"{mu:.3f} ± {sd:.3f}", mu, sd


def _summary_card(  # noqa: PLR0915, C901
    model_name: str,
    tmap: dict[str, list[float]],
    mmap: dict[str, list[float]],
) -> None:
    """
    Render a compact summary card with data-driven column widths so that
    multi-digit means (e.g., 46.741) align just as neatly as single-digit ones.
    Uses ASCII borders when JSTPROVE_ASCII=1.
    """
    _ = model_name
    phases = ("compile", "witness", "prove", "verify")

    # 1) Build labels and stats first (so we know true content widths)
    rows: list[tuple[str, str, str, str, str]] = []
    t_means: list[float] = []
    m_means: list[float] = []

    def fmt_mean_sd(vals: list[float]) -> tuple[str, float | None]:
        if not vals:
            return "NA", None
        if len(vals) == 1:
            v = float(vals[0])
            return f"{v:.3f}", v
        mu = float(mean(vals))
        sd = float(stdev(vals))
        return f"{mu:.3f} ± {sd:.3f}", mu

    for ph in phases:
        tlabel, tmean = fmt_mean_sd(tmap.get(ph, []))
        mlabel, mmean = fmt_mean_sd(mmap.get(ph, []))
        tbest = f"{min(tmap[ph]):.3f}" if tmap.get(ph) else "NA"
        mpeak = f"{max(mmap[ph]):.3f}" if mmap.get(ph) else "NA"
        rows.append((ph, tlabel, tbest, mlabel, mpeak))
        if tmean is not None:
            t_means.append(tmean)
        if mmean is not None:
            m_means.append(mmean)

    # When everything is NA, avoid div-by-zero in bar scaling
    tmax = max(t_means) if t_means else 1.0
    mmax = max(m_means) if m_means else 1.0

    # 2) Compute column widths from the actual content + headers
    hdr_phase = "phase"
    hdr_time = "time (s)"
    hdr_best = "best"
    hdr_mem = "mem (MB)"
    hdr_peak = "peak"

    phase_w = max(len(hdr_phase), *(len(ph) for ph, *_ in rows))
    time_w = max(len(hdr_time), *(len(t) for _, t, *_ in rows))
    best_w = max(len(hdr_best), *(len(b) for *_, b, _, _ in rows))
    mem_w = max(len(hdr_mem), *(len(m) for *_, m, _ in rows))
    peak_w = max(len(hdr_peak), *(len(p) for *_, p in rows))

    # pick a reasonable bar width; shrink only if terminal is very narrow
    # we’ll try to fit inside the terminal if possible, but we *don’t* rely on it.
    min_bar = 10
    max_bar = 24
    # Estimate available width from the terminal; keep a comfortable default.
    tw = _term_width(100)
    # Fixed chars per row besides the two bars (separators, spaces, borders)
    # Layout: │ {phase:<pw} │ {time:<tw} │ {best:>bw} │ {tbar} │ {mem:<mw} │ {peak:>pk} │ {mbar} │ # noqa: E501
    fixed = (
        1  # left border
        + 1
        + phase_w
        + 1
        + 1  # "│ " + phase + " │"
        + 1
        + time_w
        + 1
        + 1  # " " + time + " │"
        + 1
        + best_w
        + 1
        + 1  # " " + best + " │"
        + 1
        + 1  # " " + "│" before mem
        + 1
        + mem_w
        + 1
        + 1  # " " + mem + " │"
        + 1
        + peak_w
        + 1
        + 1  # " " + peak + " │"
        + 1  # space before mbar
        + 1  # right border (we’ll account for it at the end)
    )
    # two bars + the final right border
    # available width = tw - (fixed + 2 bars + right border). solve for bar size.
    # we’ll clamp into [min_bar, max_bar].
    avail_for_bars = max(0, tw - (fixed + 1))  # leave room for right border
    per_bar = max(min_bar, min(max_bar, avail_for_bars // 2)) or min_bar
    bar_w = per_bar

    # Optionally switch to pure-ASCII table and bar characters
    ascii_mode = os.environ.get("JSTPROVE_ASCII") == "1"
    V = "|" if ascii_mode else "│"  # noqa: N806
    H = "-" if ascii_mode else "─"  # noqa: N806
    TL = "+" if ascii_mode else "┌"  # noqa: N806
    TR = "+" if ascii_mode else "┐"  # noqa: N806
    BL = "+" if ascii_mode else "└"  # noqa: N806
    BR = "+" if ascii_mode else "┘"  # noqa: N806
    TJ = "+" if ascii_mode else "├"  # noqa: N806
    BJ = "+" if ascii_mode else "┴"  # noqa: N806,F841
    MJ = "+" if ascii_mode else "┼"  # noqa: N806
    BAR_CHAR = "#" if ascii_mode else "█"  # noqa: N806

    def bar(val: float, vmax: float) -> str:
        # scale relative to max mean; clamp; ensure non-empty when val>0
        if vmax <= 0 or val <= 0:
            return " " * bar_w
        filled = max(1, int(bar_w * min(val, vmax) / vmax))
        return BAR_CHAR * filled + " " * (bar_w - filled)

    # Make a header content row to measure the total width
    header_line = (
        f"{V} {hdr_phase:<{phase_w}} {V} {hdr_time:<{time_w}} {V} {hdr_best:>{best_w}} "
        f"{V} {'t-bar':<{bar_w}} {V} {hdr_mem:<{mem_w}} "
        f"{V} {hdr_peak:>{peak_w}} {V} {'m-bar':<{bar_w}} {V}"
    )
    # Draw a top border that exactly matches header width
    top = (
        TL + H * (len(header_line) - 2) + TR
    )  # -2 accounts for replacing first/last char with corners
    sep = (
        TJ
        + H * (2 + phase_w)
        + MJ
        + H * (2 + time_w)
        + MJ
        + H * (2 + best_w)
        + MJ
        + H * (2 + bar_w)
        + MJ
        + H * (2 + mem_w)
        + MJ
        + H * (2 + peak_w)
        + MJ
        + H * (2 + bar_w)
        + TR.replace(TR, MJ)  # match corner with a cross-joint
    )
    bottom = BL + H * (len(header_line) - 2) + BR

    print()
    print(top)
    print(header_line)
    print(sep)

    # body
    for ph, tlabel, tbest, mlabel, mpeak in rows:
        # pull means again for bar scaling; parse left side of "μ ± σ" if present
        def _to_mean(s: str) -> float:
            if s == "NA":
                return 0.0
            part = s.split("±")[0].strip()
            try:
                return float(part)
            except Exception:
                return 0.0

        tmean = _to_mean(tlabel)
        mmean = _to_mean(mlabel)

        tbar = bar(tmean, tmax)
        mbar = bar(mmean, mmax)

        line = (
            f"{V} {ph:<{phase_w}} {V} {tlabel:<{time_w}} {V} {tbest:>{best_w}} "
            f"{V} {tbar} {V} {mlabel:<{mem_w}} {V} {mpeak:>{peak_w}} {V} {mbar} {V}"
        )
        print(line)

    print(bottom)


def _fmt_stats(vals: list[float]) -> str:
    """Legacy one-liner stats; kept for compatibility with older callers."""
    if not vals:
        return "NA"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"mean={mean(vals):.3f}  stdev={stdev(vals):.3f}  n={len(vals)}"


def summarize(rows: list[dict], model_name: str) -> None:
    """
    Build per-phase arrays from JSONL rows and print the summary card.
    Only successful (return_code==0) rows for the given model are included.
    """
    phases = ("compile", "witness", "prove", "verify")
    tmap: dict[str, list[float]] = {p: [] for p in phases}
    mmap: dict[str, list[float]] = {p: [] for p in phases}
    for r in rows:
        if r.get("model") == model_name and r.get("return_code") == 0:
            if r.get("time_s") is not None:
                tmap[r["phase"]].append(float(r["time_s"]))
            if r.get("mem_mb") is not None:
                mmap[r["phase"]].append(float(r["mem_mb"]))
    _summary_card(model_name, tmap, mmap)


def main() -> int:  # noqa: PLR0915, C901, PLR0912
    """CLI entrypoint for the benchmark runner."""
    ap = argparse.ArgumentParser(
        description="Benchmark JSTProve by calling the CLI directly.",
    )
    ap.add_argument(
        "--model",
        required=True,
        help="ONNX model name (e.g., 'lenet', where path to model is "
        "python/models/models_onnx/lenet.onnx)",
    )
    ap.add_argument("--input", required=False, help="Path to input JSON (optional).")
    ap.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of E2E loops (default: 5).",
    )
    ap.add_argument(
        "--output",
        default="results.jsonl",
        help="JSONL to append per-run rows (default: results.jsonl)",
    )
    ap.add_argument(
        "--summarize",
        action="store_true",
        help="Print per-phase summary card at the end.",
    )
    args = ap.parse_args()

    model_path = Path(args.model).resolve()
    param_count = count_onnx_parameters(model_path)
    fixed_input = Path(args.input).resolve() if args.input else None
    out_path = Path(args.output).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    try:
        for it in range(1, args.iterations + 1):
            with tempfile.TemporaryDirectory() as tmp_s:
                tmp = Path(tmp_s)
                io = PhaseIO(
                    model_path=model_path,
                    circuit_path=tmp / "circuit.txt",
                    input_path=fixed_input or (tmp / "input.json"),
                    output_path=tmp / "output.json",
                    witness_path=tmp / "witness.bin",
                    proof_path=tmp / "proof.bin",
                )

                for phase in ("compile", "witness", "prove", "verify"):
                    ts = now_utc()
                    rc, out, t, m, cmd, m_rust, m_psutil, ecc_live = run_cli(phase, io)

                    # ECC and artifact sizes are collected into the JSONL
                    # (not printed live)
                    artifact_sizes: dict[str, int | None] = {}
                    if phase == "compile":
                        circuit_size = file_size_bytes(io.circuit_path)
                        quantized_path = io.circuit_path.with_name(
                            f"{io.circuit_path.stem}_quantized_model.onnx",
                        )
                        quant_size = file_size_bytes(quantized_path)
                        artifact_sizes["circuit_size_bytes"] = circuit_size
                        artifact_sizes["quantized_size_bytes"] = quant_size
                    elif phase == "witness":
                        artifact_sizes["witness_size_bytes"] = file_size_bytes(
                            io.witness_path,
                        )
                        artifact_sizes["output_size_bytes"] = file_size_bytes(
                            io.output_path,
                        )
                    elif phase in ("prove", "verify"):
                        artifact_sizes["proof_size_bytes"] = file_size_bytes(
                            io.proof_path,
                        )

                    row = {
                        "timestamp": ts,
                        "model": str(model_path),
                        "iteration": it,
                        "phase": phase,
                        "return_code": rc,
                        "time_s": t,
                        "mem_mb": m,
                        "mem_mb_rust": m_rust,
                        "mem_mb_psutil": m_psutil,
                        "ecc": (ecc_live if ecc_live else {}),
                        "cmd": cmd,
                        "tmpdir": str(tmp),
                        "param_count": param_count,
                        **artifact_sizes,
                    }
                    rows.append(row)
                    with out_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")

                    # Guard: compile claimed success but circuit missing
                    if phase == "compile" and rc == 0:
                        if not io.circuit_path.exists():
                            log.error(
                                "[compile] rc=0 but circuit file missing: "
                                "%s\n----- compile output -----\n%s",
                                io.circuit_path,
                                out,
                            )
                            return 1
                        # Quantized is expected; warn (do not fail) if missing.
                        qpath = _quantized_path_from_circuit(io.circuit_path)
                        if not qpath.exists():
                            log.warning(
                                "[compile] expected quantized ONNX missing: %s",
                                qpath,
                            )

                    if rc != 0:
                        log.error("[%s] rc=%s — see logs below\n%s\n", phase, rc, out)

                    if t is not None:
                        mem_str = f"{m:.2f}" if m is not None else "NA"
                        log.info("[%s] t=%.3fs, mem=%s MB", phase, t, mem_str)
                    else:
                        log.info("[%s] metrics not parsed; rc=%s", phase, rc)

    except KeyboardInterrupt:
        log.info("\nCancelled by user (Ctrl+C).")
        return 130
    else:
        log.info("")
        log.info("✔ Wrote %d rows to %s", len(rows), out_path)
        if args.summarize:
            summarize(rows, str(model_path))
        return 0


if __name__ == "__main__":
    sys.exit(main())
