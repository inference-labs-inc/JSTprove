import os
import re
import subprocess
import sys
import threading
import time
from typing import Set
import psutil


def get_mem_usage_cli(pid: int) -> int:
    """Get memory usage (resident set size in KB) from ps."""
    if sys.platform != 'darwin':
        # ps -o rss= might work on Linux, but vmmap won't
        return 0
    try:
        # Use full path to be sure we get the system's ps
        command = ['/bin/ps', '-p', str(pid), '-o', 'rss=']
        output = subprocess.check_output(command, stderr=subprocess.DEVNULL)
        # Output might be empty if process died; strip whitespace before int conversion
        rss_kb_str = output.decode().strip()
        return int(rss_kb_str) if rss_kb_str else 0
    except subprocess.CalledProcessError:
        # Process likely doesn't exist anymore
        return 0

    except Exception as e:
        print(f"Error getting RSS via CLI for PID {pid}: {e}")
        return 0


def get_swap_usage_cli(pid: int) -> int:
    """Estimate swap usage (in KB) using vmmap. Supports both legacy and modern vmmap output."""
    if sys.platform != 'darwin':
        return 0
    try:
        command = ['/usr/bin/sudo', '/usr/bin/vmmap', str(pid)]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=5)

        if process.returncode != 0:
            # print(f"[Debug Swap PID {pid}] vmmap exited with code {process.returncode}")
            if stderr:
                pass
                # print(f"[Debug Swap PID {pid}] vmmap stderr:\n{stderr.strip()}")
            return 0

        output = stdout

        # First attempt: legacy "Swap used"
        match_old = re.search(r'Swap used:\s+([\d,]+)\s*(K|KB)', output, re.IGNORECASE)
        if match_old:
            val = int(match_old.group(1).replace(',', ''))
            return val

        # Second attempt: modern "swapped_out=" or "swapped_out ="
        match_new = re.search(r'swapped[_ ]?out\s*=\s*([\d.]+)\s*([KMG])', output, re.IGNORECASE)
        if match_new:
            val = float(match_new.group(1))
            unit = match_new.group(2).upper()
            multiplier = {'K': 1, 'M': 1024, 'G': 1024 * 1024}.get(unit, 1)
            swap_kb = int(val * multiplier)
            return swap_kb

        return 0

    except Exception as e:
        print(f"Error getting Swap via vmmap for PID {pid}: {e}")
        return 0

def monitor_subprocess_memory(parent_pid, process_name_keyword, results, stop_event):
    """
    Monitors child processes of parent_pid using CLI tools (ps, vmmap).
    Tracks the peak SUM of RSS ('mem_cli') and Swap ('swap_cli') across
    all children matching the keyword.
    Updates the 'results' dictionary.
    """
    peak_mem_cli = 0
    peak_swap_cli = 0
    tracked_pids: Set[int] = set()

    # Ensure the results dictionary exists and initialize keys
    if not isinstance(results, dict):
        return

    # Use different keys to distinguish from psutil results
    results['peak_subprocess_mem'] = 0
    results['peak_subprocess_swap'] = 0
    results['peak_subprocess_total'] = 0 # Mem + Swap

    try:
        parent = psutil.Process(parent_pid)

        while not stop_event.is_set():
            current_total_mem_cli = 0
            current_total_swap_cli = 0
            children_found_this_cycle = []
            # print(parent.children(recursive=True))
            try:
                # Still use psutil to reliably get all descendants
                children_found_this_cycle = parent.children(recursive=True)
            except psutil.NoSuchProcess:
                # print(f"Parent process {parent_pid} seems to have ended. Stopping CLI monitor.")
                break
            except Exception as e:
                # print(f"Error getting children for PID {parent_pid}: {e}")
                time.sleep(0.1) # Avoid tight loop on error
                continue

            active_pids_this_cycle = set()

            # First pass: Identify and add new matching children
            for proc in children_found_this_cycle:
                try:
                    pid = proc.pid
                    active_pids_this_cycle.add(pid) # Keep track of currently running children
                    if pid not in tracked_pids:
                         # Check name only for newly found processes
                        proc_name = proc.name()
                        if process_name_keyword.lower() in proc_name.lower():
                            tracked_pids.add(pid)
                            # print(f"CLI Monitoring child: {proc_name} (PID: {pid})")

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process ended or permissions issue while checking name/pid
                    if pid in tracked_pids:
                        tracked_pids.remove(pid) # Stop tracking
                    continue
                except Exception as e:
                    # print(f"Error inspecting potential child {pid}: {e}")
                    if pid in tracked_pids:
                         tracked_pids.remove(pid) # Stop tracking
                    continue

            # Second pass: Check memory for all currently tracked PIDs
            pids_to_remove = set()
            for pid in tracked_pids:
                if pid not in active_pids_this_cycle:
                     # Process we were tracking is no longer a child (finished?)
                     pids_to_remove.add(pid)
                     continue # Don't try to get memory for it

                # Get memory and swap using CLI tools
                mem_kb = get_mem_usage_cli(pid)
                swap_kb = get_swap_usage_cli(pid)

                # Add to the total for this cycle
                # Note: If get_mem/swap returns 0 (e.g., process just died), it adds 0
                current_total_mem_cli += mem_kb
                current_total_swap_cli += swap_kb

            # Remove PIDs that are no longer active
            tracked_pids.difference_update(pids_to_remove)
            if pids_to_remove:
                pass
                 # print(f"Stopped CLI tracking for finished PIDs: {pids_to_remove}")


            # Update overall peaks for the SUM of memory/swap
            peak_mem_cli = max(peak_mem_cli, current_total_mem_cli)
            peak_swap_cli = max(peak_swap_cli, current_total_swap_cli)

            # Break if stop signal received after checking all children this cycle
            if stop_event.is_set():
                break

            # Sleep before next cycle (might need longer sleep due to CLI overhead)
            time.sleep(0.1) # Increased sleep interval

    except psutil.NoSuchProcess:
        print(f"Initial parent process {parent_pid} not found.")
    except Exception as e:
        print(f"Major error in CLI monitoring thread: {e}")
    finally:
        # Store final peak results (convert KB to Bytes for consistency if desired)
        # Store in KB as originally retrieved
        results['peak_subprocess_mem'] = peak_mem_cli
        results['peak_subprocess_swap'] = peak_swap_cli
        results['peak_subprocess_total'] = peak_mem_cli + peak_swap_cli

def start_memory_collection(process_name):
    parent_pid = os.getpid()
    monitor_results = {'peak_subprocess_rss': 0}
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target= monitor_subprocess_memory,
        args=(parent_pid, process_name, monitor_results, stop_event),
        daemon=True
    )
    monitor_thread.start()
    time.sleep(0.1)  # Give thread a moment to start up

    return stop_event, monitor_thread, monitor_results


def end_memory_collection(stop_event, monitor_thread, monitor_results):
    stop_event.set()
    monitor_thread.join(timeout=2.0)  # Wait briefly for the thread to finish
    if monitor_thread.is_alive():
        print("Warning: Child process memory monitor thread did not terminate cleanly.")
    subprocess_ram = monitor_results.get('peak_subprocess_mem', 0) / 1024
    subprocess_swap = monitor_results.get('peak_subprocess_swap', 0) / 1024
    subprocess_total = monitor_results.get('peak_subprocess_total', 0) / 1024

    memory = {'ram': subprocess_ram, 'swap': subprocess_swap, 'total': subprocess_total}

    return memory
