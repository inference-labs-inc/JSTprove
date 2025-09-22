# python\core\utils\benchmarking_helpers.py
from __future__ import annotations

import threading
import time

import psutil


def _safe_rss_kb(pid: int) -> int:
    """
    Return the process RSS in KB using psutil.
    If the process no longer exists or memory info is unavailable, return 0.
    """
    try:
        proc = psutil.Process(pid)
        rss_bytes = proc.memory_info().rss  # type: ignore[attr-defined]
        return int(rss_bytes // 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def _list_children(parent_pid: int) -> list[psutil.Process]:
    """
    Safely list all descendant processes for a given parent PID.
    Returns an empty list if the parent is gone or access is denied.
    """
    try:
        parent = psutil.Process(parent_pid)
        return parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def _safe_name_lower(proc: psutil.Process) -> str | None:
    """
    Safely return lowercase process name, or None if not available.
    (Avoids try/except inside the monitor loop to satisfy PERF203.)
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
    Monitor memory (peak sum of RSS) across child processes of `parent_pid`.
    If `process_name_keyword` is non-empty, only children whose lowercase name
    contains that keyword are included.

    Stores peaks (in KB) under:
      - results['peak_subprocess_mem']   (RSS only)
      - results['peak_subprocess_swap']  (always 0 in this implementation)
      - results['peak_subprocess_total'] (mem + swap)
    """
    keyword = process_name_keyword.strip().lower()
    peak_rss_kb = 0

    # Initialize result keys to be robust if caller inspects mid-flight
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
            results["peak_subprocess_swap"] = 0  # Swap not collected here
            results["peak_subprocess_total"] = peak_rss_kb  # mem + swap(0)

        time.sleep(poll_interval_s)

    # Final write (in case no updates happened inside the loop)
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
    Start a background thread to monitor memory for child processes of the current PID.
    Returns (stop_event, monitor_thread, monitor_results_dict).
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
    time.sleep(0.05)  # allow thread to start
    return stop_event, monitor_thread, monitor_results


def end_memory_collection(
    stop_event: threading.Event,
    monitor_thread: threading.Thread,
    monitor_results: dict[str, int],
) -> dict[str, float]:
    """
    Stop the memory monitor thread and return a summary dict in MB:
      {'ram': <MB>, 'swap': <MB>, 'total': <MB>}
    Falls back to zeros if results are missing.
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
