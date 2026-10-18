"""Benchmark diagnostics: GPU, host CPU and server-process CPU samples, GPU idle check."""
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
GPU_FIELDS = ("timestamp,uuid,name,driver_version,pstate,utilization.gpu,memory.used,memory.total,"
              "power.draw,power.limit,temperature.gpu,clocks.current.graphics,clocks.current.memory")
TICK = os.sysconf("SC_CLK_TCK")


def gpu_snapshot():
    if not shutil.which("nvidia-smi"):
        return None
    p = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=" + GPU_FIELDS, "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True)
    return dict(zip(GPU_FIELDS.split(","), [x.strip() for x in p.stdout.strip().split(",")]))


def cpu_times():
    """(busy, total) jiffies over all cores from /proc/stat; None off Linux."""
    try:
        f = [int(x) for x in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
    except OSError:
        return None
    idle = f[3] + f[4]
    return sum(f) - idle, sum(f)


def proc_cpu_s(pid):
    """utime + stime of one process in seconds, from /proc/<pid>/stat; None if unavailable."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, TypeError):
        return None
    return (int(fields[11]) + int(fields[12])) / TICK


def wait_gpu_idle(samples=5, limit=120):
    """Wait until GPU utilisation is <= 5% for ``samples`` seconds in a row."""
    seen, idle = [], 0
    for _ in range(limit):
        snap = gpu_snapshot()
        if snap is None:
            return seen, True
        seen.append(snap)
        idle = idle + 1 if float(snap["utilization.gpu"]) <= 5 else 0
        if idle >= samples:
            return seen, True
        time.sleep(1)
    return seen, False


def prepare_gpu():
    """At start: record whether the GPU is idle and try to lock its clock (rented hosts refuse)."""
    if not shutil.which("nvidia-smi"):
        return None
    seen, ok = wait_gpu_idle()
    record = {"before_load": seen, "idle_pass": ok}
    lock = subprocess.run(["nvidia-smi", "-i", "0", "-lgc", "1500,1500"], capture_output=True, text=True)
    record["clock_lock"] = {"requested_mhz": 1500, "returncode": lock.returncode,
                            "stdout": lock.stdout, "stderr": lock.stderr}
    record["after_lock"] = gpu_snapshot()
    (RESULTS / "gpu-preflight.json").write_text(json.dumps(record, indent=2))
    return record


class Monitor:
    """Samples GPU, host CPU and (optionally) one process's CPU once a second in a thread."""

    def __init__(self, pid=None):
        self.pid = pid
        self.rows, self._stop = [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        last, last_proc, last_t = cpu_times(), proc_cpu_s(self.pid), time.time()
        while not self._stop.wait(1):
            now, now_proc, t = cpu_times(), proc_cpu_s(self.pid), time.time()
            row = {"t": t}
            if last and now and now[1] > last[1]:
                row["cpu_util"] = 100 * (now[0] - last[0]) / (now[1] - last[1])
            if last_proc is not None and now_proc is not None:
                row["proc_cpu_cores"] = (now_proc - last_proc) / (t - last_t)
            last, last_proc, last_t = now, now_proc, t
            try:
                snap = gpu_snapshot()
                if snap:
                    row.update(gpu_util=float(snap["utilization.gpu"]), gpu_mem_mb=float(snap["memory.used"]),
                               gpu_clock_mhz=float(snap["clocks.current.graphics"]),
                               gpu_temp_c=float(snap["temperature.gpu"]))
            except Exception as e:  # noqa: BLE001 - a failed sample must not stop the run
                row["error"] = str(e)
            self.rows.append(row)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)

    def between(self, t0, t1):
        return [r for r in self.rows if t0 <= r["t"] <= t1]
