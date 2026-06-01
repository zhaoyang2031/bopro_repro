#!/usr/bin/env python3
"""Small GPU-aware experiment queue for the BBO reproduction runs."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NFBO_CALL_RE = re.compile(r"(\d+) oracles has been called")
NFBO_SCORE_RE = re.compile(r"Best ([A-Za-z0-9_-]+) Score: ([0-9.]+)")
DIBO_MAX_RE = re.compile(r"Max so far: ([-\d.]+)")


def figure_group_key(job_name: str) -> str | None:
    """Map job name to aggregate figure basename (without .png)."""
    if job_name.startswith("nfbo_"):
        m = re.match(r"nfbo_([a-z0-9]+)_", job_name, re.I)
        if m:
            return f"nfbo_{m.group(1)}"
    if job_name.startswith("dibo_full_"):
        parts = job_name.split("_")
        if len(parts) >= 4:
            return f"dibo_{parts[2]}_{parts[3]}"
    if job_name.startswith("ddom_full_"):
        parts = job_name.split("_")
        if len(parts) >= 3:
            return f"ddom_{parts[2].replace('-', '_')}"
    if job_name.startswith("gtg_full_"):
        parts = job_name.split("_")
        if len(parts) >= 3:
            return f"gtg_{parts[2].replace('-', '_')}"
    if job_name.startswith("vsd_"):
        return job_name.rsplit("_seed", 1)[0]
    if job_name.startswith("genbo_"):
        return job_name.rsplit("_seed", 1)[0]
    return None


@dataclass(frozen=True)
class Job:
    name: str
    paper: str
    suite: str
    cwd: str
    command: str
    conda_env: str | None = None
    conda_sh: str = "/data/xk/zhaoyang/miniconda3/etc/profile.d/conda.sh"
    timeout_sec: int | None = None
    enabled: bool = True
    env_setup: str | None = None


def load_jobs(path: Path, suite: str) -> list[Job]:
    data = json.loads(path.read_text())
    jobs: list[Job] = []
    for item in data["experiments"]:
        suites = item.get("suite", [])
        if isinstance(suites, str):
            suites = [suites]
        if suite not in suites:
            continue
        job = Job(
            name=item["name"],
            paper=item["paper"],
            suite=suite,
            cwd=item["cwd"],
            command=item["command"],
            conda_env=item.get("conda_env"),
            conda_sh=item.get(
                "conda_sh", "/data/xk/zhaoyang/miniconda3/etc/profile.d/conda.sh"
            ),
            timeout_sec=item.get("timeout_sec"),
            enabled=item.get("enabled", True),
            env_setup=item.get("env_setup"),
        )
        if job.enabled:
            jobs.append(job)
    return jobs


def make_shell(job: Job) -> str:
    parts = [
        "set -eo pipefail",
        "export WANDB_API_KEY=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 && export WANDB_ENTITY=1585515136- && export WANDB_MODE=online",
        "export CUDA_DEVICE_ORDER=PCI_BUS_ID",
    ]
    if job.conda_env:
        parts.append(f"source {shlex.quote(job.conda_sh)}")
        parts.append(f"conda activate {shlex.quote(job.conda_env)}")
    if job.env_setup:
        parts.append(job.env_setup)
    parts.append(f"cd {shlex.quote(job.cwd)}")
    parts.append(job.command)
    return " && ".join(parts)


def append_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def check_completion(logfile: Path, job: Job) -> tuple[bool, int]:
    if not logfile.exists():
        return False, 0
    
    content = logfile.read_text(errors="replace")
    
    if "nfbo" in job.name.lower():
        max_calls_match = re.search(r"max_n_oracle_calls=(\d+)", job.command)
        target_calls = int(max_calls_match.group(1)) if max_calls_match else 500
        
        matches = NFBO_CALL_RE.findall(content)
        if matches:
            actual_calls = int(matches[-1])
            if actual_calls >= target_calls:
                return True, actual_calls
    
    elif "dibo" in job.name.lower():
        max_evals_match = re.search(r"max_evals=(\d+)", job.command)
        target_evals = int(max_evals_match.group(1)) if max_evals_match else 10000
        
        matches = DIBO_MAX_RE.findall(content)
        if matches:
            return True, len(matches)
    
    elif "ddom" in job.name.lower() or "gtg" in job.name.lower():
        if "training finished" in content.lower() or "evaluation completed" in content.lower() or "best score" in content.lower():
            matches = re.findall(r"best[:\s]+([-\d.]+)", content, re.IGNORECASE)
            if matches:
                return True, len(matches)
    
    elif "vsd" in job.name.lower() or "genbo" in job.name.lower():
        if "Solver done" in content or "best value found" in content.lower():
            return True, 1
    
    return False, 0


def worker(
    workq: "queue.Queue[Job]",
    gpuq: "queue.Queue[str]",
    args: argparse.Namespace,
    status_path: Path,
    lock: threading.Lock,
) -> None:
    while True:
        try:
            job = workq.get_nowait()
        except queue.Empty:
            return

        gpu = gpuq.get()
        start = time.time()
        logdir = Path(args.logdir) / args.suite
        logdir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in job.name)
        logfile = logdir / f"{safe_name}.gpu{gpu}.log"

        completed, current_calls = check_completion(logfile, job)
        if completed:
            record = {
                "event": "skip",
                "job": job.name,
                "paper": job.paper,
                "suite": job.suite,
                "gpu": gpu,
                "reason": "already_completed",
                "calls": current_calls,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with lock:
                print(f"[skip] gpu={gpu} {job.name} (already completed: {current_calls} calls)", flush=True)
                append_status(status_path, record)
            gpuq.put(gpu)
            workq.task_done()
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu

        cmd = ["bash", "-lc", make_shell(job)]
        record = {
            "event": "start",
            "job": job.name,
            "paper": job.paper,
            "suite": job.suite,
            "gpu": gpu,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "logfile": str(logfile),
        }
        with lock:
            print(f"[start] gpu={gpu} {job.name}", flush=True)
            append_status(status_path, record)

        rc = 124
        try:
            with logfile.open("a", encoding="utf-8", errors="replace") as f:
                f.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
                proc = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=job.cwd,
                    env=env,
                    text=True,
                )
                try:
                    rc = proc.wait(timeout=job.timeout_sec)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = 124
                    f.write(f"\n[timeout] exceeded {job.timeout_sec} seconds\n")
        except Exception as exc:  # noqa: BLE001
            rc = 125
            logfile.write_text(f"queue failure: {exc}\n", encoding="utf-8")

        end = time.time()
        record = {
            "event": "finish",
            "job": job.name,
            "paper": job.paper,
            "suite": job.suite,
            "gpu": gpu,
            "returncode": rc,
            "seconds": round(end - start, 2),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "logfile": str(logfile),
        }
        with lock:
            print(f"[finish] rc={rc} gpu={gpu} {job.name}", flush=True)
            append_status(status_path, record)
        gpuq.put(gpu)
        workq.task_done()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="repro/experiments.json")
    parser.add_argument("--suite", choices=["smoke", "full"], required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--logdir", default="/data/xk/zhaoyang/bbo_repro/logs")
    parser.add_argument("--only", default=None, help="Substring filter for job names.")
    parser.add_argument("--force", action="store_true", help="Force rerun all jobs.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip already completed jobs.")
    parser.add_argument(
        "--skip-figured",
        action="store_true",
        help="Skip job groups that already have a PNG in --figures-dir.",
    )
    parser.add_argument(
        "--figures-dir",
        default="/nas1/xk/zhaoyang/bbo_repro/figures",
        help="Directory of finished benchmark plots.",
    )
    args = parser.parse_args()

    jobs = load_jobs(Path(args.experiments), args.suite)
    if args.only:
        jobs = [job for job in jobs if args.only in job.name]
    if not jobs:
        print(f"No enabled jobs for suite={args.suite}", file=sys.stderr)
        return 1

    if args.skip_figured:
        figures_dir = Path(args.figures_dir)
        figured_groups: set[str] = set()
        if figures_dir.is_dir():
            figured_groups = {p.stem for p in figures_dir.glob("*.png")}
        skip_names: set[str] = set()
        for job in jobs:
            key = figure_group_key(job.name)
            if key and key in figured_groups:
                skip_names.add(job.name)
        if skip_names:
            print(f"Skipping {len(skip_names)} jobs with existing figures in {figures_dir}")
            jobs = [j for j in jobs if j.name not in skip_names]

    if args.skip_completed:
        logdir = Path(args.logdir) / args.suite
        skipped = 0
        remaining = []
        for job in jobs:
            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in job.name)
            job_completed = False
            for gpu in [x.strip() for x in args.gpus.split(",") if x.strip()]:
                logfile = logdir / f"{safe_name}.gpu{gpu}.log"
                completed, calls = check_completion(logfile, job)
                if completed:
                    print(f"[skip] {job.name} on gpu{gpu} already completed ({calls} calls)")
                    skipped += 1
                    job_completed = True
                    break
            if not job_completed:
                remaining.append(job)
        
        if skipped > 0:
            print(f"\nSkipped {skipped} completed jobs. {len(remaining)} jobs remaining.")
            jobs = remaining

    if not jobs:
        print("All jobs already completed!")
        return 0

    workq: "queue.Queue[Job]" = queue.Queue()
    for job in jobs:
        workq.put(job)

    gpuq: "queue.Queue[str]" = queue.Queue()
    for gpu in [x.strip() for x in args.gpus.split(",") if x.strip()]:
        gpuq.put(gpu)

    n_threads = min(args.max_parallel, len(jobs), gpuq.qsize())
    status_path = Path(args.logdir) / "status.jsonl"
    lock = threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(workq, gpuq, args, status_path, lock))
        for _ in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
