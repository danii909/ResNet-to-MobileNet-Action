#!/usr/bin/env python3
"""Live monitor for KD training jobs on the DMI cluster.

Displays SLURM job status, training progress (epoch, loss, accuracy),
GPU usage, and disk quota in a compact refreshing view.

Usage:
    python3 -m src.utils.monitor              # default poll 15s
    python3 -m src.utils.monitor --poll 10    # poll every 10s

Designed to run on the cluster login node.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

# -- Config -------------------------------------------------------------------
PROJ_DIR = Path(os.environ.get("HOME", "~")) / "dl26-projects"
LOGS_DIR = PROJ_DIR / "logs"

# -- ANSI colors --------------------------------------------------------------
_RST = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_WHITE = "\033[97m"
_GRAY = "\033[90m"

_STATE_ICONS = {
    "COMPLETED": f"{_GREEN}v{_RST}",
    "FAILED": f"{_RED}x{_RST}",
    "RUNNING": f"{_CYAN}>{_RST}",
    "PENDING": f"{_YELLOW}~{_RST}",
    "TIMEOUT": f"{_RED}T{_RST}",
    "CANCELLED": f"{_RED}C{_RST}",
}

_STATE_COLORS = {
    "COMPLETED": _GREEN,
    "FAILED": _RED,
    "RUNNING": _CYAN,
    "PENDING": _YELLOW,
    "TIMEOUT": _RED,
    "CANCELLED": _RED,
}


# -- Data structures ----------------------------------------------------------
@dataclass
class JobInfo:
    """Info about a SLURM training job."""
    slurm_id: str = ""
    name: str = ""
    state: str = "UNKNOWN"
    elapsed: str = ""
    exit_code: str = ""
    config: str = ""
    # Training metrics (parsed from log)
    current_epoch: int = 0
    total_epochs: int = 0
    train_loss: str = ""
    train_acc: str = ""
    test_acc: str = ""
    best_acc: str = ""
    lr: str = ""
    # Progress bar (tqdm)
    tqdm_step: int = 0
    tqdm_total: int = 0
    tqdm_pct: int = 0


# -- Shell helpers ------------------------------------------------------------
def _run(cmd: str) -> str:
    """Run a shell command and return stdout."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


# -- SLURM queries ------------------------------------------------------------
def _get_my_jobs() -> list[JobInfo]:
    """Get all recent SLURM jobs (last 2 days) matching kd-train."""
    out = _run(
        "sacct --me --starttime=$(date -d '2 days ago' +%Y-%m-%d) "
        "--format=JobID%15,JobName%20,State%15,Elapsed%12,ExitCode%10 "
        "--noheader --parsable2"
    )
    jobs: list[JobInfo] = []
    seen_ids: set[str] = set()
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        job_id, name, state, elapsed, exit_code = (
            parts[0], parts[1], parts[2], parts[3], parts[4],
        )
        # Skip sub-steps
        if "." in job_id:
            continue
        if job_id in seen_ids:
            continue
        seen_ids.add(job_id)

        # Normalize state
        if state.startswith("CANCELLED"):
            state = "CANCELLED"

        job = JobInfo(
            slurm_id=job_id,
            name=name,
            state=state,
            elapsed=elapsed,
            exit_code=exit_code,
        )
        jobs.append(job)

    # Also check squeue for pending/running not yet in sacct
    sq_out = _run('squeue --me --noheader --format="%i|%j|%T|%M"')
    for line in sq_out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        sid, name, state, elapsed = parts[0], parts[1], parts[2], parts[3]
        if sid in seen_ids:
            # Update state if running
            for j in jobs:
                if j.slurm_id == sid:
                    j.state = state
                    j.elapsed = elapsed
            continue
        seen_ids.add(sid)
        jobs.append(JobInfo(
            slurm_id=sid, name=name, state=state, elapsed=elapsed,
        ))

    # Sort by ID descending (most recent first)
    jobs.sort(key=lambda j: j.slurm_id, reverse=True)
    return jobs


# -- Log parsing --------------------------------------------------------------
_EPOCH_RE = re.compile(
    r"Epoch (\d+)/(\d+) \| "
    r"Train Loss: ([\d.]+) \| "
    r"Train Acc: ([\d.]+)% \| "
    r"Test Acc: ([\d.]+)% \| "
    r"Best: ([\d.]+)% \| "
    r"LR: ([\d.]+)"
)
_CONFIG_RE = re.compile(r"Config:\s+(\S+)")
_TQDM_RE = re.compile(r"(\d+)%\|.*\|\s*(\d+)/(\d+)")
_STARTING_RE = re.compile(r"Starting training.*epochs=(\d+)")


def _parse_log(job: JobInfo) -> None:
    """Parse the SLURM log file for a job and populate metrics."""
    candidate_paths = [
        LOGS_DIR / f"slurm-train-{job.slurm_id}.log",
        LOGS_DIR / f"slurm-train-seq-{job.slurm_id}.log",
    ]

    log_path = next((p for p in candidate_paths if p.exists()), None)
    if log_path is None:
        return

    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return

    # Parse from end for latest metrics
    for line in reversed(lines):
        m = _EPOCH_RE.search(line)
        if m:
            job.current_epoch = int(m.group(1))
            job.total_epochs = int(m.group(2))
            job.train_loss = m.group(3)
            job.train_acc = m.group(4)
            job.test_acc = m.group(5)
            job.best_acc = m.group(6)
            job.lr = m.group(7)
            break

    # Parse config from top
    for line in lines[:30]:
        m = _CONFIG_RE.search(line)
        if m:
            job.config = m.group(1)
            break

    # If no epoch found yet, check for starting message
    if job.total_epochs == 0:
        for line in lines:
            m = _STARTING_RE.search(line)
            if m:
                job.total_epochs = int(m.group(1))
                break

    # Parse tqdm progress from last lines
    for line in reversed(lines[-50:]):
        m = _TQDM_RE.search(line)
        if m:
            job.tqdm_pct = int(m.group(1))
            job.tqdm_step = int(m.group(2))
            job.tqdm_total = int(m.group(3))
            break


# -- GPU info -----------------------------------------------------------------
def _get_gpu_info() -> str:
    """Get GPU utilization from nvidia-smi (only works on compute node)."""
    out = _run(
        "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader,nounits 2>/dev/null"
    )
    if not out:
        return f"{_DIM}GPU info non disponibile (login node){_RST}"
    parts = out.split(",")
    if len(parts) >= 4:
        name = parts[0].strip()
        gpu_util = parts[1].strip()
        mem_used = parts[2].strip()
        mem_total = parts[3].strip()
        return (
            f"{_CYAN}{name}{_RST} | "
            f"GPU: {_WHITE}{gpu_util}%{_RST} | "
            f"VRAM: {_WHITE}{mem_used}/{mem_total} MB{_RST}"
        )
    return out


# -- Disk usage ---------------------------------------------------------------
def _get_disk_usage() -> str:
    """Get disk usage summary."""
    total = _run(f"du -sh {PROJ_DIR.parent} 2>/dev/null | cut -f1")
    hf_cache = _run("du -sh ~/.cache/huggingface 2>/dev/null | cut -f1")
    return f"Disco: {_WHITE}{total}{_RST} totale | HF cache: {_WHITE}{hf_cache}{_RST}"


# -- Display ------------------------------------------------------------------
def _progress_bar(current: int, total: int, width: int = 25) -> str:
    """Render a text progress bar."""
    if total <= 0:
        return ""
    pct = min(100, int(current / total * 100))
    filled = int(width * pct / 100)
    bar = f"{_CYAN}{'#' * filled}{_GRAY}{'.' * (width - filled)}{_RST}"
    return f"{bar} {_WHITE}{pct}%{_RST}"


def _display(jobs: list[JobInfo]) -> None:
    """Print the monitor dashboard."""
    os.system("clear")

    print(f"{_CYAN}{'=' * 70}{_RST}")
    print(f"  {_BOLD}{_CYAN}KD Training Monitor{_RST} -- {_DIM}{time.strftime('%Y-%m-%d %H:%M:%S')}{_RST}")
    print(f"{_CYAN}{'=' * 70}{_RST}")

    if not jobs:
        print(f"\n  {_DIM}Nessun job trovato. Lancia: train teacher.yaml{_RST}\n")
        return

    # -- Job table
    print(f"\n  {_BOLD}{'ID':<12s} {'Config':<30s} {'Stato':<12s} {'Tempo':<12s}{_RST}")
    print(f"  {'-' * 66}")

    for job in jobs[:10]:  # Show last 10 jobs max
        _parse_log(job)
        icon = _STATE_ICONS.get(job.state, "?")
        sc = _STATE_COLORS.get(job.state, "")

        config_short = Path(job.config).stem if job.config else job.name
        state_str = f"{sc}{job.state}{_RST}"

        detail = ""
        if job.state == "FAILED" and job.exit_code:
            detail = f" {_RED}(exit {job.exit_code}){_RST}"

        print(
            f"  {icon} {job.slurm_id:<10s} {config_short:<30s} "
            f"{state_str:<22s} {_DIM}{job.elapsed}{_RST}{detail}"
        )

    # -- Active job details
    active = [j for j in jobs if j.state == "RUNNING"]
    if active:
        j = active[0]
        _parse_log(j)
        print(f"\n{_CYAN}{'-' * 70}{_RST}")
        print(f"  {_BOLD}{_CYAN}Job attivo:{_RST} {j.slurm_id} ({Path(j.config).stem if j.config else j.name})")
        print(f"  Tempo: {_WHITE}{j.elapsed}{_RST}")

        if j.current_epoch > 0:
            epoch_bar = _progress_bar(j.current_epoch, j.total_epochs)
            print(f"\n  Epoca:     {_WHITE}{j.current_epoch}/{j.total_epochs}{_RST}  {epoch_bar}")
            print(f"  Train Loss: {_WHITE}{j.train_loss}{_RST}")
            print(f"  Train Acc:  {_WHITE}{j.train_acc}%{_RST}")
            print(f"  Test Acc:   {_WHITE}{j.test_acc}%{_RST}  (Best: {_GREEN}{j.best_acc}%{_RST})")
            print(f"  LR:         {_DIM}{j.lr}{_RST}")
        elif j.tqdm_step > 0:
            batch_bar = _progress_bar(j.tqdm_step, j.tqdm_total)
            print(f"\n  Batch:     {_WHITE}{j.tqdm_step}/{j.tqdm_total}{_RST}  {batch_bar}")
        else:
            print(f"\n  {_YELLOW}Avvio in corso...{_RST}")

    # -- Pending jobs
    pending = [j for j in jobs if j.state == "PENDING"]
    if pending:
        print(f"\n  {_YELLOW}In coda: {len(pending)} job{_RST}")

    # -- Disk usage
    print(f"\n{_DIM}{'-' * 70}{_RST}")
    print(f"  {_get_disk_usage()}")
    print()


# -- Main ---------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Live monitor per il training KD sul cluster DMI"
    )
    parser.add_argument(
        "--poll", type=int, default=15,
        help="Secondi tra un refresh e l'altro (default: 15)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Mostra lo stato una volta e esci",
    )
    args = parser.parse_args()

    print("KD Monitor -- Ctrl+C per uscire")
    if not args.once:
        print(f"Poll ogni {args.poll}s...")
    print()

    try:
        while True:
            jobs = _get_my_jobs()
            _display(jobs)

            if args.once:
                break

            # Exit if no running/pending jobs
            has_active = any(j.state in ("RUNNING", "PENDING") for j in jobs)
            if jobs and not has_active:
                print(f"  {_GREEN}Tutti i job completati.{_RST}")
                break

            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\nMonitor fermato.")


if __name__ == "__main__":
    main()
