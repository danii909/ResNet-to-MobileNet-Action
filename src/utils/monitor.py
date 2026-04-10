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
EXP_LOGS_DIR = PROJ_DIR / "experiments" / "logs"

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
    training_type: str = ""
    node: str = ""
    gpu_name: str = ""
    partition: str = ""
    qos: str = ""
    job_log_dir: str = ""
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


def _read_kv_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key:
                out[key] = value
    except OSError:
        return {}
    return out


def _read_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _coalesce(*values: str) -> str:
    for value in values:
        if value and str(value).strip():
            return str(value).strip()
    return ""


def _short(text: str, width: int) -> str:
    if not text:
        return "-"
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _infer_training_type(mode: str, config_path: str) -> str:
    mode_l = (mode or "").lower()
    cfg_l = (config_path or "").lower()
    if mode_l == "teacher_finetune" or "teacher" in cfg_l:
        return "teacher"
    if mode_l == "baseline" or "baseline" in cfg_l:
        return "baseline"
    if mode_l in ("distillation", "distillation_at") or "distill" in cfg_l:
        return "distillation"
    return "unknown"


def _infer_training_type_from_run_name(run_name: str) -> str:
    """Infer training type from the custom run label used in multi-run scripts."""
    name_l = (run_name or "").lower()
    if "teacher" in name_l:
        return "teacher"
    if "baseline" in name_l:
        return "baseline"
    if "kd" in name_l or "distill" in name_l:
        return "distillation"
    return "unknown"


def _default_config_for_type(training_type: str) -> str:
    """Return the canonical config path for a known training type."""
    if training_type == "teacher":
        return "experiments/configs/teacher.yaml"
    if training_type == "baseline":
        return "experiments/configs/baseline.yaml"
    if training_type == "distillation":
        return "experiments/configs/distillation.yaml"
    return ""


def _infer_pipeline_phase(job_dir: Path) -> tuple[str, str]:
    """Return the current phase and config path for a train-eval pipeline job."""
    phase_order = [
        ("teacher", "teacher_finetune"),
        ("baseline", "baseline"),
        ("distillation", "distillation"),
    ]

    for phase_name, mode_name in phase_order:
        phase_dir = job_dir / phase_name
        train_dir = phase_dir / "train"
        eval_dir = phase_dir / "eval"
        if not phase_dir.exists():
            continue

        train_success = (train_dir / "status_SUCCESS").exists()
        train_failed = (train_dir / "status_FAILED").exists()
        eval_success = (eval_dir / "status_SUCCESS").exists()
        eval_failed = (eval_dir / "status_FAILED").exists()

        if not train_success or (train_success and not eval_success and not eval_failed):
            train_meta = _read_kv_file(train_dir / "job_meta.txt")
            return mode_name, train_meta.get("config", "")

        if train_failed or eval_failed:
            train_meta = _read_kv_file(train_dir / "job_meta.txt")
            return mode_name, train_meta.get("config", "")

    last_train_meta = _read_kv_file(job_dir / "distillation" / "train" / "job_meta.txt")
    return "distillation", last_train_meta.get("config", "")


# -- SLURM queries ------------------------------------------------------------
def _get_my_jobs() -> list[JobInfo]:
    """Get all recent SLURM jobs (last 2 days) matching kd-train."""
    out = _run(
        "sacct --me --starttime=$(date -d '2 days ago' +%Y-%m-%d) "
        "--format=JobID%15,JobName%20,State%15,Elapsed%12,ExitCode%10,NodeList%24,Partition%16,QOS%16 "
        "--noheader --parsable2"
    )
    jobs: list[JobInfo] = []
    seen_ids: set[str] = set()
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 8:
            continue
        job_id, name, state, elapsed, exit_code, node, partition, qos = (
            parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7],
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
            node=node,
            partition=partition,
            qos=qos,
        )
        jobs.append(job)

    # Also check squeue for pending/running not yet in sacct
    sq_out = _run('squeue --me --noheader --format="%i|%j|%T|%M|%N|%P|%q"')
    for line in sq_out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 7:
            continue
        sid, name, state, elapsed, node, partition, qos = (
            parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6],
        )
        if sid in seen_ids:
            # Update state if running
            for j in jobs:
                if j.slurm_id == sid:
                    j.state = state
                    j.elapsed = elapsed
                    if node and node != "(null)":
                        j.node = node
                    if partition and partition != "(null)":
                        j.partition = partition
                    if qos and qos != "(null)":
                        j.qos = qos
            continue
        seen_ids.add(sid)
        jobs.append(JobInfo(
            slurm_id=sid,
            name=name,
            state=state,
            elapsed=elapsed,
            node=node,
            partition=partition,
            qos=qos,
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
_CONFIGS_RE = re.compile(r"Configs:\s+(.+)")
_SEQ_START_RE = re.compile(r"\[\d+/\d+\]\s+Starting:\s+(\S+)")
_MULTI_TRAIN_START_RE = re.compile(r"^>>>\s+TRAINING:\s+(\S+)")
_MULTI_CONFIG_RE = re.compile(r"^\s*config:\s+(\S+)")
_MODE_RE = re.compile(r"Training mode:\s+([A-Za-z0-9_]+)")
_NODE_RE = re.compile(r"^\s*Node:\s+(\S+)")
_GPU_RE = re.compile(r"\[main\]\s+GPU:\s+(.+)")
_TQDM_RE = re.compile(r"(\d+)%\|.*\|\s*(\d+)/(\d+)")
_STARTING_RE = re.compile(r"Starting training.*epochs=(\d+)")


def _enrich_from_job_artifacts(job: JobInfo) -> None:
    """Enrich a job with metadata saved under experiments/logs."""
    candidate_dirs = [
        EXP_LOGS_DIR / f"slurm-multiple-runs-{job.slurm_id}",
        EXP_LOGS_DIR / f"slurm-next-runs-4041-{job.slurm_id}",
        EXP_LOGS_DIR / f"slurm-train-eval-{job.slurm_id}",
        EXP_LOGS_DIR / f"slurm-train-{job.slurm_id}",
        EXP_LOGS_DIR / f"slurm-train-seq-{job.slurm_id}",
    ]
    job_dir = next((p for p in candidate_dirs if p.exists()), None)
    if job_dir is None:
        # Fallback: accept any job directory suffixing with the SLURM id.
        wildcard_matches = sorted(EXP_LOGS_DIR.glob(f"*-{job.slurm_id}"))
        job_dir = wildcard_matches[0] if wildcard_matches else None
    if job_dir is None:
        return

    job.job_log_dir = str(job_dir)

    root_meta = _read_kv_file(job_dir / "job_meta.txt")
    job.config = _coalesce(job.config, root_meta.get("config", ""))
    job.node = _coalesce(job.node, root_meta.get("hostname", ""))
    job.gpu_name = _coalesce(job.gpu_name, root_meta.get("gpu_name", ""))
    job.partition = _coalesce(job.partition, root_meta.get("partition", ""))
    job.qos = _coalesce(job.qos, root_meta.get("qos", ""))
    job.training_type = _coalesce(job.training_type, root_meta.get("training_type", ""))

    if "slurm-train-eval-" in job_dir.name:
        phase_type, phase_config = _infer_pipeline_phase(job_dir)
        job.training_type = _coalesce(job.training_type, phase_type)
        job.config = _coalesce(job.config, phase_config)

    summary = _read_json_file(job_dir / "job_training_summary.json")
    runs = summary.get("runs") if isinstance(summary.get("runs"), list) else []
    if not runs:
        return

    latest = runs[-1]
    if isinstance(latest, dict):
        job.config = _coalesce(job.config, latest.get("config_path", ""))
        job.node = _coalesce(job.node, latest.get("node", ""))
        job.gpu_name = _coalesce(job.gpu_name, latest.get("gpu", ""))
        job.training_type = _coalesce(job.training_type, latest.get("training_type", ""))


def _parse_log(job: JobInfo) -> None:
    """Parse the SLURM log file for a job and populate metrics."""
    _enrich_from_job_artifacts(job)

    candidate_paths = [
        LOGS_DIR / f"slurm-next-4041-{job.slurm_id}.log",
        LOGS_DIR / f"slurm-train-eval-{job.slurm_id}.log",
        LOGS_DIR / f"slurm-train-{job.slurm_id}.log",
        LOGS_DIR / f"slurm-train-seq-{job.slurm_id}.log",
    ]
    log_path = next((p for p in candidate_paths if p.exists()), None)
    if log_path is None:
        # Fallback: pick any SLURM log ending with this job id.
        wildcard_logs = sorted(LOGS_DIR.glob(f"*{job.slurm_id}.log"))
        log_path = wildcard_logs[-1] if wildcard_logs else None
    if log_path is None:
        if not job.training_type:
            job.training_type = _infer_training_type("", job.config)
        return

    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        if not job.training_type:
            job.training_type = _infer_training_type("", job.config)
        return

    seen_mode = ""
    latest_run_name = ""

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

    # Parse current config for sequential jobs from latest start marker.
    for line in reversed(lines[-250:]):
        m = _SEQ_START_RE.search(line)
        if m:
            job.config = m.group(1)
            break

    # Parse current run for single-job multi-run scripts.
    # We look for the latest ">>> TRAINING: <run_name>" marker and read its config line.
    tail_start = max(0, len(lines) - 400)
    tail_lines = lines[tail_start:]
    for i in range(len(tail_lines) - 1, -1, -1):
        m = _MULTI_TRAIN_START_RE.search(tail_lines[i])
        if not m:
            continue
        latest_run_name = m.group(1)
        for j in range(i, min(i + 8, len(tail_lines))):
            c = _MULTI_CONFIG_RE.search(tail_lines[j])
            if c:
                job.config = c.group(1)
                break
        break

    # Parse config/mode/node/gpu from top section.
    for line in lines[:150]:
        m = _CONFIG_RE.search(line)
        if m and not job.config:
            job.config = m.group(1)

        m = _CONFIGS_RE.search(line)
        if m and not job.config:
            all_cfgs = m.group(1).split()
            if all_cfgs:
                job.config = all_cfgs[0]

        m = _MODE_RE.search(line)
        if m:
            seen_mode = m.group(1)

        m = _NODE_RE.search(line)
        if m and (not job.node or job.node in ("Unknown", "None assigned")):
            job.node = m.group(1)

        m = _GPU_RE.search(line)
        if m and not job.gpu_name:
            job.gpu_name = m.group(1).strip()

    # Also parse latest mode from tail (important for multi-phase logs).
    for line in reversed(lines[-400:]):
        m = _MODE_RE.search(line)
        if m:
            seen_mode = m.group(1)
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

    if latest_run_name and job.state == "RUNNING":
        # For active multi-run jobs, the latest marker is the source of truth.
        run_type = _infer_training_type_from_run_name(latest_run_name)
        if run_type != "unknown":
            job.training_type = run_type
            if not job.config:
                job.config = _default_config_for_type(run_type)

    if not job.training_type or job.training_type == "unknown":
        job.training_type = _infer_training_type(seen_mode, job.config)
    if (not job.training_type or job.training_type == "unknown") and latest_run_name:
        job.training_type = _infer_training_type_from_run_name(latest_run_name)
    if (not job.config) and job.training_type:
        job.config = _default_config_for_type(job.training_type)
    if not job.training_type or job.training_type == "unknown":
        if "slurm-train-eval-" in job.job_log_dir:
            phase_type, phase_config = _infer_pipeline_phase(Path(job.job_log_dir))
            job.training_type = phase_type
            job.config = _coalesce(job.config, phase_config)


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
    print(
        f"\n  {_BOLD}{'ID':<10s} {'Type':<13s} {'Config':<20s} "
        f"{'Node':<14s} {'GPU':<20s} {'Stato':<10s} {'Tempo':<10s}{_RST}"
    )
    print(f"  {'-' * 108}")

    for job in jobs[:10]:  # Show last 10 jobs max
        _parse_log(job)
        icon = _STATE_ICONS.get(job.state, "?")
        sc = _STATE_COLORS.get(job.state, "")

        config_short = _short(Path(job.config).stem if job.config else job.name, 20)
        type_short = _short(job.training_type or _infer_training_type("", job.config), 13)
        node_short = _short(job.node, 14)
        gpu_short = _short(job.gpu_name, 20)
        state_str = f"{sc}{_short(job.state, 10)}{_RST}"

        detail = ""
        if job.state == "FAILED" and job.exit_code:
            detail = f" {_RED}(exit {job.exit_code}){_RST}"

        print(
            f"  {icon} {job.slurm_id:<8s} {type_short:<13s} {config_short:<20s} "
            f"{node_short:<14s} {gpu_short:<20s} {state_str:<19s} "
            f"{_DIM}{_short(job.elapsed, 10)}{_RST}{detail}"
        )

    # -- Active job details
    active = [j for j in jobs if j.state == "RUNNING"]
    if active:
        j = active[0]
        _parse_log(j)
        print(f"\n{_CYAN}{'-' * 70}{_RST}")
        print(f"  {_BOLD}{_CYAN}Job attivo:{_RST} {j.slurm_id} ({Path(j.config).stem if j.config else j.name})")
        print(f"  Tipo: {_WHITE}{j.training_type or _infer_training_type('', j.config)}{_RST}")
        print(f"  Nodo: {_WHITE}{j.node or '-'}{_RST}")
        print(f"  GPU: {_WHITE}{j.gpu_name or '-'}{_RST}")
        if j.partition or j.qos:
            print(f"  Partition/QoS: {_WHITE}{j.partition or '-'} / {j.qos or '-'}{_RST}")
        if j.job_log_dir:
            print(f"  Job log dir: {_DIM}{j.job_log_dir}{_RST}")
            summary_path = Path(j.job_log_dir) / "job_training_summary.txt"
            if summary_path.exists():
                print(f"  Job summary: {_DIM}{summary_path}{_RST}")
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
