# python\scripts\benchmark_runner.py
# ruff: noqa: S603

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Optional

import time
import shutil
import psutil

from python.core.utils.benchmarking_helpers import (
    end_memory_collection,
    start_memory_collection,
)

log = logging.getLogger(__name__)

# Configure logging (INFO by default; switch to DEBUG if needed)
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("benchmark_runner")

# Which child process name to track per phase (case-insensitive substring match).
# If your circuit binary name differs, adjust "onnx_generic_circuit" accordingly.
PROC_KEY_BY_PHASE = {
    "compile": "onnx_generic_circuit",
    "witness": "onnx_generic_circuit",
    "prove": "expander-exec",
    "verify": "expander-exec",
}

# Parse the "built layered circuit ..." and "built hint normalized ir ..." lines.
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

TIME_PATTERNS = [
    re.compile(r"Rust time taken:\s*([0-9.]+)"),
    re.compile(r"Time elapsed:\s*([0-9.]+)\s*seconds"),
]
MEM_PATTERNS = [
    re.compile(r"Peak Memory used Overall\s*:\s*([0-9.]+)"),
    re.compile(r"Rust subprocess memory\s*:\s*([0-9.]+)"),
]

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

# Match either message; allow arbitrary prefixes (timestamps/levels/etc.)
ECC_HINT_RE = re.compile(r"built\s+hint\s+normalized\s+ir\b.*", re.IGNORECASE)
ECC_LAYERED_RE = re.compile(r"built\s+layered\s+circuit\b.*", re.IGNORECASE)

KV_PAIR = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([0-9]+)")

# spinner: ASCII by default (fast, always visible).
# Set JSTPROVE_UNICODE=1 to use the braille spinner.
_SPINNER_ASCII = "-\\|/"
_SPINNER_UNICODE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _term_width(default: int = 100) -> int:
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default


def _human_bytes(n: int | None) -> str:
    if n is None:
        return "NA"
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.1f} {u}" if u != "B" else f"{int(x)} B"
        x /= 1024.0


def _fmt_int(n: int | None) -> str:
    return f"{n:,}" if isinstance(n, int) else "NA"


def _bar(value: int, vmax: int, width: int = 24, char: str = "█") -> str:
    if vmax <= 0 or value <= 0:
        return " " * width
    fill = max(1, int(width * min(value, vmax) / vmax))
    return char * fill + " " * (width - fill)


def _marquee(t: float, width: int = 24, char: str = "█") -> str:
    # a bouncing block to suggest progress when total unknown
    w = max(8, min(width, 24))
    pos = int((abs(((t * 0.8) % 2) - 1)) * (w - 8))
    return " " * pos + char * 8 + " " * (w - 8 - pos)


def _sum_child_rss_mb(parent_pid: int) -> float:
    """Approx current total RSS of children (MB). Best-effort."""
    try:
        parent = psutil.Process(parent_pid)
    except psutil.Error:
        return 0.0
    total = 0
    for c in parent.children(recursive=True):
        try:
            total += c.memory_info().rss
        except psutil.Error:
            pass
    return total / (1024.0 * 1024.0)


def parse_ecc_stats(text: str) -> Dict[str, int]:
    """
    Strip ANSI, then look for ECC summary lines and extract key=value ints.
    We search the entire blob (not per-line) to be robust to prefixes.
    """
    # 1) remove ANSI/color
    clean = ANSI_RE.sub("", text)

    stats: Dict[str, int] = {}

    # 2) find both possible ECC summary lines anywhere in the blob
    for pat in (ECC_HINT_RE, ECC_LAYERED_RE):
        for m in pat.finditer(clean):
            line = m.group(0)
            for k, v in KV_PAIR.findall(line):
                try:
                    stats[k] = int(v)
                except ValueError:
                    pass

    return stats


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


# --- ONNX parameter counting ---
def count_onnx_parameters(model_path: Path) -> int:
    """
    Return total number of parameters in an ONNX model by summing
    the element counts of all graph initializers (trainable weights).
    This does not require loading tensor data into RAM.
    """
    try:
        import onnx  # type: ignore
    except Exception:
        # onnx not installed; just return -1 sentinel so you can see it's missing
        return -1

    model = onnx.load(str(model_path))  # loads external data references if present
    total = 0
    for init in model.graph.initializer:
        # Use dims product; avoids materializing the tensor
        n = 1
        for d in init.dims:
            n *= int(d)
        total += n
    return int(total)


def file_size_bytes(path: str | Path) -> Optional[int]:
    """Return file size in bytes, or None if missing."""
    try:
        p = Path(path)
        return p.stat().st_size if p.exists() else None
    except OSError:
        return None


def parse_metrics(text: str) -> tuple[float | None, float | None]:
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
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class PhaseIO:
    model_path: Path
    circuit_path: Path
    input_path: Path | None
    output_path: Path
    witness_path: Path
    proof_path: Path


def _build_phase_cmd(phase: str, io: PhaseIO) -> list[str]:
    base = ["jstprove", "--no-banner"]
    if phase == "compile":
        return [
            *base,
            "compile",
            "-m",
            str(io.model_path),
            "-c",
            str(io.circuit_path),
        ]
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


def run_cli(
    phase: str,
    io: PhaseIO,
) -> tuple[int, str, float | None, float | None, list[str], float | None, float | None]:
    """
    Run the CLI, stream output live (so the terminal isn't idle), and show a spinner
    + elapsed + marquee + live peak-RSS. Returns:
      (returncode, combined_output, time_s, mem_mb_primary, cmd_list, mem_mb_rust, mem_mb_psutil)
    """
    cmd = _build_phase_cmd(phase, io)

    env = os.environ.copy()
    env.setdefault("RUST_LOG", "info")
    env.setdefault("RUST_BACKTRACE", "1")

    # We still keep psutil-based peak tracking across *all* children for consistency
    stop_ev, mon_thread, mon_results = start_memory_collection("")

    start = time.time()
    combined_lines: list[str] = []

    # Use Popen to read output as it arrives
    proc = subprocess.Popen(  # noqa: S603
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
            now = time.time()
            elapsed = now - start

            # Update live peak from direct sampling (fast)
            live_mb = _sum_child_rss_mb(proc.pid)
            if live_mb > peak_live_mb:
                peak_live_mb = live_mb

            # stream any output we just read
            if line:
                combined_lines.append(line.rstrip("\n"))
                # Also echo ECC milestones immediately so it feels responsive
                if (
                    "built layered circuit" in line.lower()
                    or "built hint normalized ir" in line.lower()
                ):
                    print(line, end="")  # noqa: T201
            else:
                # If no new line, still refresh the HUD while process runs
                pass

            # Draw HUD every ~100ms
            if int((elapsed * 10)) != int(((elapsed - 0.09) * 10)) or not line:
                spin = spinner[sp_i % len(spinner)]
                sp_i += 1
                if phase == "compile":
                    hud_bar = _marquee(elapsed, width=bar_w)
                else:
                    hud_bar = _marquee(elapsed, width=bar_w)  # reuse for all phases

                hud = f"\r[{spin}] {phase:<7} | {elapsed:6.1f}s | mem↑ {peak_live_mb:7.1f} MB | {hud_bar}"
                print(hud[: tw - 1], end="", flush=True)  # noqa: T201

            if proc.poll() is not None:
                # one last HUD update on exit
                elapsed = time.time() - start
                hud = (
                    f"\r[✔] {phase:<7} | {elapsed:6.1f}s | mem↑ {peak_live_mb:7.1f} MB | "
                    + " " * bar_w
                )
                print(hud[: tw - 1])  # newline  # noqa: T201
                break

            time.sleep(0.09)

    finally:
        # Stop background peak monitor and capture its result (may differ slightly)
        collected_mb: float | None = None
        try:
            mem = end_memory_collection(stop_ev, mon_thread, mon_results)  # type: ignore[arg-type]
            if isinstance(mem, dict):
                collected_mb = float(mem.get("total", 0.0))
        except Exception:
            collected_mb = None

    combined = "\n".join(combined_lines)
    # Primary metrics from runner prints (if any), else from our timers
    time_s, mem_mb_rust = parse_metrics(combined)
    if time_s is None:
        time_s = elapsed
    mem_mb_psutil = collected_mb if collected_mb is not None else peak_live_mb
    mem_mb_primary = mem_mb_psutil if mem_mb_psutil is not None else mem_mb_rust

    # ECC stats
    ecc = parse_ecc_stats(combined)
    if ecc:
        kv = " ".join(f"{k}={v}" for k, v in sorted(ecc.items()))
        combined += f"\n[ECC]\n{kv}\n"

    return (
        proc.returncode or 0,
        combined,
        time_s,
        mem_mb_primary,
        cmd,
        mem_mb_rust,
        mem_mb_psutil,
    )


def _print_compile_card(
    ecc: dict, circuit_bytes: int | None, quant_bytes: int | None
) -> None:
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
    vmax = max(v for v in data.values())
    w = max(24, min(40, _term_width() - 50))

    print("")  # noqa: T201
    print(
        "┌────────────────────────── Compile Stats ──────────────────────────┐"
    )  # noqa: T201
    for k in keys:
        if k in data:
            bar = _bar(data[k], vmax, width=w)
            print(f"│ {k:<14} {_fmt_int(data[k]):>12}  {bar} │")  # noqa: T201
    print(
        "├────────────────────────────────────────────────────────────────────┤"
    )  # noqa: T201
    print(
        f"│ circuit.txt        {(_human_bytes(circuit_bytes) if circuit_bytes is not None else 'NA'):>12}                              │"
    )  # noqa: T201
    print(
        f"│ quantized_model    {(_human_bytes(quant_bytes) if quant_bytes is not None else 'NA'):>12}                              │"
    )  # noqa: T201
    print(
        "└────────────────────────────────────────────────────────────────────┘"
    )  # noqa: T201


def run_cli_old(
    phase: str,
    io: PhaseIO,
) -> tuple[int, str, float | None, float | None, list[str], float | None, float | None]:
    """
    Run your CLI exactly as you do by hand, capture stdout/stderr, and return:
    (returncode, combined_output, time_s, mem_mb_primary, cmd_list, mem_mb_rust, mem_mb_psutil).
    """
    cmd = _build_phase_cmd(phase, io)

    # Ensure info-level Rust logs are actually emitted
    env = os.environ.copy()
    env.setdefault("RUST_LOG", "info")
    env.setdefault("RUST_BACKTRACE", "1")

    # psutil collection for all phases
    stop_ev, mon_thread, mon_results = start_memory_collection("")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    finally:
        collected_mb: float | None = None
        try:
            mem = end_memory_collection(stop_ev, mon_thread, mon_results)  # type: ignore[arg-type]
            if isinstance(mem, dict):
                collected_mb = float(mem.get("total", 0.0))  # MB, per your helper
        except Exception:
            collected_mb = None

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    combined = stdout + (("\n[stderr]\n" + stderr) if stderr else "")

    # Primary timing and "Rust" memory (if your runner prints them)
    time_s, mem_mb_rust = parse_metrics(combined)
    mem_mb_psutil = collected_mb
    mem_mb_primary = mem_mb_psutil if mem_mb_psutil is not None else mem_mb_rust

    # ECC stats (robust to ANSI and prefixes)
    ecc = parse_ecc_stats(combined)

    if phase == "compile":
        log.debug("ECC parsed keys: %s", ",".join(sorted(ecc.keys())))

    # (Optional) append a compact ECC summary to combined output so you can see it in terminal logs
    if ecc:
        kv = " ".join(f"{k}={v}" for k, v in sorted(ecc.items()))
        combined += f"\n[ECC]\n{kv}\n"

    return (
        proc.returncode,
        combined,
        time_s,
        mem_mb_primary,
        cmd,
        mem_mb_rust,
        mem_mb_psutil,
    )


def _quantized_path_from_circuit(circuit_path: Path) -> Path:
    """
    Exact rule from prepare_io_files in helper_functions.py:
      <circuit_dir>/<circuit_stem>_quantized_model.onnx
    """
    return circuit_path.with_name(f"{circuit_path.stem}_quantized_model.onnx")


def _fmt_mean_sd(vals: list[float]) -> tuple[str, float | None, float | None]:
    if not vals:
        return "NA", None, None
    if len(vals) == 1:
        v = vals[0]
        return f"{v:.3f}", v, None
    mu, sd = mean(vals), stdev(vals)
    return f"{mu:.3f} ± {sd:.3f}", mu, sd


def _summary_card(
    model_name: str, tmap: dict[str, list[float]], mmap: dict[str, list[float]]
) -> None:
    phases = ("compile", "witness", "prove", "verify")
    rows = []
    t_means = []
    m_means = []
    for ph in phases:
        tlabel, tmean, _ = _fmt_mean_sd(tmap.get(ph, []))
        mlabel, mmean, _ = _fmt_mean_sd(mmap.get(ph, []))
        tbest = f"{min(tmap[ph]):.3f}" if tmap.get(ph) else "NA"
        mpeak = f"{max(mmap[ph]):.3f}" if mmap.get(ph) else "NA"
        rows.append((ph, tlabel, tbest, mlabel, mpeak))
        if tmean is not None:
            t_means.append(tmean)
        if mmean is not None:
            m_means.append(mmean)

    # bars scaled to max mean across phases
    tmax = max(t_means) if t_means else 1.0
    mmax = max(m_means) if m_means else 1.0

    tw = _term_width()
    # columns: phase | time (mean±sd, best) [bar] | mem (mean±sd, peak) [bar]
    bar_w = max(10, min(18, tw - 84))  # adapt to terminal width
    title = f" Summary for {model_name} "
    pad = max(0, tw - len(title) - 4)
    print("")  # noqa: T201
    print("┌" + title + "─" * pad + "┐")  # noqa: T201
    print(
        "│ phase    │ time (s)                 │ best  │ "  # noqa: T201
        f"{'t-bar':<{bar_w}} │ mem (MB)               │ peak  │ {'m-bar':<{bar_w}} │"
    )
    print(
        "├──────────┼──────────────────────────┼───────┼"
        + "─" * (bar_w + 2)  # noqa: T201
        + "┼──────────────────────────┼───────┼"
        + "─" * (bar_w + 2)
        + "┤"
    )  # noqa: T201

    for ph, tlabel, tbest, mlabel, mpeak in rows:
        # parse means back out for bar len (cheap and robust)
        try:
            tmean = float(tlabel.split("±")[0].strip())
        except Exception:
            tmean = 0.0
        try:
            mmean = float(mlabel.split("±")[0].strip())
        except Exception:
            mmean = 0.0

        tbar = _bar(int(tmean * 1000), int(tmax * 1000), width=bar_w)
        mbar = _bar(int(mmean * 1000), int(mmax * 1000), width=bar_w)

        print(
            f"│ {ph:<8} │ {tlabel:<26} │ {tbest:>5} │ {tbar} │ "  # noqa: T201
            f"{mlabel:<24} │ {mpeak:>5} │ {mbar} │"
        )
    print("└" + "─" * (tw - 2) + "┘")  # noqa: T201


def _fmt_stats(vals: list[float]) -> str:
    if not vals:
        return "NA"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"mean={mean(vals):.3f}  stdev={stdev(vals):.3f}  n={len(vals)}"


def summarize(rows: list[dict], model_name: str) -> None:
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


def summarize_old(rows: list[dict], model_name: str) -> None:
    phases = ("compile", "witness", "prove", "verify")
    tmap: dict[str, list[float]] = {p: [] for p in phases}
    mmap: dict[str, list[float]] = {p: [] for p in phases}
    for r in rows:
        if r.get("model") == model_name and r.get("return_code") == 0:
            if r.get("time_s") is not None:
                tmap[r["phase"]].append(r["time_s"])
            if r.get("mem_mb") is not None:
                mmap[r["phase"]].append(r["mem_mb"])

    log.info("")
    log.info("Summary for %s:", model_name)
    for ph in phases:
        tvals = tmap[ph]
        mvals = mmap[ph]
        tstr = _fmt_stats(tvals)
        mstr = _fmt_stats(mvals)
        # Wrap long line to satisfy E501
        log.info("  %-8s: time=%s | mem=%s", ph, tstr, mstr)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark JSTProve by calling the CLI directly.",
    )
    ap.add_argument(
        "--model",
        required=True,
        help="Path to ONNX model (e.g., python/models/models_onnx/lenet.onnx)",
    )
    ap.add_argument(
        "--input",
        required=False,
        help="Path to input JSON (if omitted, use your usual one).",
    )
    ap.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of E2E loops (default: 5).",
    )
    ap.add_argument(
        "--output",
        default="results.jsonl",
        help="JSONL file to append per-run rows (default: results.jsonl)",
    )
    ap.add_argument(
        "--summarize",
        action="store_true",
        help="Print simple per-phase summary at the end.",
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
                    rc, out, t, m, cmd, m_rust, m_psutil = run_cli(phase, io)

                    # ECC key=value stats from CLI logs
                    ecc_stats = parse_ecc_stats(out)

                    # Artifact sizes per phase
                    artifact_sizes: dict[str, Optional[int]] = {}
                    # ECC key=value stats from CLI logs
                    ecc_stats = parse_ecc_stats(out)

                    # Artifact sizes per phase (collect → JSONL, do NOT display now)
                    artifact_sizes: dict[str, Optional[int]] = {}
                    if phase == "compile":
                        circuit_size = file_size_bytes(io.circuit_path)
                        quantized_path = io.circuit_path.with_name(
                            f"{io.circuit_path.stem}_quantized_model.onnx"
                        )
                        quant_size = file_size_bytes(quantized_path)
                        artifact_sizes["circuit_size_bytes"] = circuit_size
                        artifact_sizes["quantized_size_bytes"] = quant_size
                    elif phase == "witness":
                        artifact_sizes["witness_size_bytes"] = file_size_bytes(
                            io.witness_path
                        )
                        artifact_sizes["output_size_bytes"] = file_size_bytes(
                            io.output_path
                        )
                    elif phase == "prove":
                        artifact_sizes["proof_size_bytes"] = file_size_bytes(
                            io.proof_path
                        )
                    elif phase == "verify":
                        artifact_sizes["proof_size_bytes"] = file_size_bytes(
                            io.proof_path
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
                        "ecc": ecc_stats,  # ECC circuit counters
                        "cmd": cmd,
                        "tmpdir": str(tmp),
                        "param_count": param_count,
                        **artifact_sizes,  # file sizes, when applicable
                    }
                    rows.append(row)
                    with out_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")

                    # Early guard: if compile says OK but files are missing, stop now
                    if phase == "compile" and rc == 0:
                        if not io.circuit_path.exists():
                            log.error(
                                "[compile] rc=0 but circuit file missing: %s\n----- compile output -----\n%s",
                                io.circuit_path,
                                out,
                            )
                            return 1

                        # Quantized model path is derived from circuit path. We expect it,
                        # but only warn (don’t fail) if it’s missing.
                        qpath = _quantized_path_from_circuit(io.circuit_path)
                        if not qpath.exists():
                            log.warning(
                                "[compile] expected quantized ONNX missing: %s", qpath
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
