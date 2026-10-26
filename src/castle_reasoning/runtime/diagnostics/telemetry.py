"""Runtime diagnostics telemetry."""

import json
import os
import subprocess
import threading
import time
from pathlib import Path


def process(pid):
    """Read cumulative counters; exited/inaccessible processes remain explicit."""
    root = Path(f"/proc/{pid}")
    try:
        stat = root.joinpath("stat").read_text().rsplit(")", 1)[1].split()
        result = dict(
            pid=pid,
            cpu_seconds=(int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK"),
            rss_bytes=int(stat[21]) * os.sysconf("SC_PAGE_SIZE"),
            start_ticks=int(stat[19]),
        )
    except (OSError, ValueError, IndexError) as error:
        return dict(pid=pid, missing=str(error))
    try:
        result["io"] = {
            key: int(value)
            for key, value in (
                line.split(":") for line in root.joinpath("io").read_text().splitlines()
            )
        }
    except OSError as error:
        result["io_missing"] = str(error)
    children = set()
    for path in root.glob("task/*/children"):
        try:
            children.update(map(int, path.read_text().split()))
        except OSError:
            pass  # A thread may exit between enumeration and reading.
    result["children"] = sorted(children)
    return result


def cgroup():
    try:
        membership = Path("/proc/self/cgroup").read_text().splitlines()
        relative = next(
            line.split(":", 2)[2] for line in membership if line.startswith("0::")
        )
        root = Path("/sys/fs/cgroup") / relative.lstrip("/")
        return {
            name: root.joinpath(name).read_text().strip()
            for name in ("cpu.stat", "cpu.max", "memory.current")
        }
    except (OSError, StopIteration) as error:
        return {"missing": str(error)}


class Monitor:
    """Save approximately 1 Hz samples outside the collector owner thread."""

    def __init__(self, directory, interval=1, gpu=True):
        self.directory, self.interval, self.gpu = Path(directory), interval, gpu
        self.done = threading.Event()
        self.gpu_process = self.gpu_log = None

    def sample(self, final=False):
        pending, seen, records = [os.getpid()], set(), []
        while pending:
            pid = pending.pop(0)
            if pid in seen:
                continue
            seen.add(pid)
            record = process(pid)
            records.append(record)
            pending.extend(record.get("children", []))
        self.stream.write(
            json.dumps(
                dict(
                    time_ns=time.perf_counter_ns(),
                    unix_time=time.time(),
                    processes=records,
                    cgroup=cgroup(),
                    final=final,
                )
            )
            + "\n"
        )
        self.stream.flush()

    def run(self):
        while not self.done.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.stream = self.directory.joinpath("telemetry.jsonl").open("x")
        if self.gpu:
            self.gpu_log = self.directory.joinpath("gpu.csv").open("x")
            try:
                self.gpu_process = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw",
                        "--format=csv",
                        "--loop=1",
                    ],
                    stdout=self.gpu_log,
                    stderr=self.gpu_log,
                )
            except OSError as error:
                self.gpu_log.write(f"Unavailable: {error}\n")
        self.sample()
        self.thread = threading.Thread(
            target=self.run, name="timing-telemetry", daemon=True
        )
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.done.set()
        self.thread.join()
        self.sample(final=True)
        self.stream.close()
        if self.gpu_process is not None:
            if self.gpu_process.poll() is None:
                self.gpu_process.terminate()
            self.gpu_process.wait(timeout=5)
        if self.gpu_log is not None:
            self.gpu_log.close()
