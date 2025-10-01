# python/core/utils/benchmarking_helpers.py
from __future__ import annotations

# --- Standard library --------------------------------------------------------
import threading
import time

# --- Third-party -------------------------------------------------------------
import psutil

"""
Lightweight helpers to measure peak memory usage of child processes during
benchmarks. Uses a background thread that periodically sums the RSS of all
descendants of the current process (optionally filtered by a name keyword).
"""


def _safe_rss_kb(pid: int) -> int:
    """
    Return RSS for a PID in KB. On errors/missing process, return 0.
    """
    try:
        proc = psutil.Process(pid)
        rss_bytes = proc.memory_info().rss  # type: ignore[attr-defined]
        return int(rss_bytes // 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def _list_children(parent_pid: int) -> list[psutil.Process]:
    """
    Return all descendant processes of a parent PID.
    Empty list if the parent is gone or access is denied.
    """
    try:
        parent = psutil.Process(parent_pid)
        return parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def _safe_name_lower(proc: psutil.Process) -> str | None:
    """
    Lowercased process name, or None if unavailable.
    """
    try:
        return proc.name().lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def monitor_subprocess_memory(
    parent_pid: int,
    process_name_keyword: str,
    results: dict[str, int],
    stop_event: threading.Event,
    *,
    poll_interval_s: float = 0.1,
) -> None:
    """
    Track the peak sum of RSS across child processes of `parent_pid`.

    If `process_name_keyword` is non-empty, only include children whose
    name contains that lowercase keyword.

    Writes peaks (KB) into `results` under:
      - 'peak_subprocess_mem'   (RSS)
      - 'peak_subprocess_swap'  (0; swap not collected here)
      - 'peak_subprocess_total' (mem + swap)
    """
    keyword = process_name_keyword.strip().lower()
    peak_rss_kb = 0

    # Initialize keys so callers can inspect mid-run safely
    results["peak_subprocess_mem"] = 0
    results["peak_subprocess_swap"] = 0
    results["peak_subprocess_total"] = 0

    while not stop_event.is_set():
        children = _list_children(parent_pid)
        if not children and not psutil.pid_exists(parent_pid):
            break

        if keyword:
            filtered: list[psutil.Process] = []
            for c in children:
                nm = _safe_name_lower(c)
                if nm and keyword in nm:
                    filtered.append(c)
        else:
            filtered = children

        rss_sum_kb = 0
        for c in filtered:
            rss_sum_kb += _safe_rss_kb(c.pid)

        if rss_sum_kb > peak_rss_kb:
            peak_rss_kb = rss_sum_kb
            results["peak_subprocess_mem"] = peak_rss_kb
            results["peak_subprocess_swap"] = 0
            results["peak_subprocess_total"] = peak_rss_kb

        time.sleep(poll_interval_s)

    # Final write (covers the case where peak never changed inside the loop)
    results["peak_subprocess_mem"] = max(
        results.get("peak_subprocess_mem", 0),
        peak_rss_kb,
    )
    results["peak_subprocess_swap"] = 0
    results["peak_subprocess_total"] = results["peak_subprocess_mem"]


def start_memory_collection(
    process_name: str,
) -> tuple[threading.Event, threading.Thread, dict[str, int]]:
    """
    Spawn and start a monitoring thread for the current process' children.

    Args:
        process_name:
            Optional substring to filter child process names (case-insensitive).
            Pass "" to include all children.

    Returns:
        (stop_event, monitor_thread, monitor_results_dict)
    """
    parent_pid = psutil.Process().pid
    monitor_results: dict[str, int] = {}
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_subprocess_memory,
        args=(parent_pid, process_name, monitor_results, stop_event),
        kwargs={"poll_interval_s": 0.02},
        daemon=True,
    )
    monitor_thread.start()
    time.sleep(0.05)  # allow thread to start and populate initial keys
    return stop_event, monitor_thread, monitor_results


def end_memory_collection(
    stop_event: threading.Event,
    monitor_thread: threading.Thread,
    monitor_results: dict[str, int],
) -> dict[str, float]:
    """
    Stop the monitor thread and return a summary dict in MB:
      {'ram': <MB>, 'swap': <MB>, 'total': <MB>}
    """
    stop_event.set()
    monitor_thread.join(timeout=5.0)

    rss_kb = int(monitor_results.get("peak_subprocess_mem", 0))
    swap_kb = int(monitor_results.get("peak_subprocess_swap", 0))
    total_kb = int(monitor_results.get("peak_subprocess_total", rss_kb + swap_kb))

    kb_to_mb = 1.0 / 1024.0
    return {
        "ram": rss_kb * kb_to_mb,
        "swap": swap_kb * kb_to_mb,
        "total": total_kb * kb_to_mb,
    }
