"""Bench controller on the GPU host (vkong app_port 8000). Not in the request path: it starts the lite
server on port 6006, runs loadtest.py against it on this host, and serves the results.

    POST /server?name=lite&num_slots=32&steps=8&...      (re)start serve.py with these flags
    GET  /server                                          what runs
    POST /benchmark?tag=burst-same&mode=burst&concurrency=1,2,4&audio=same&...   run loadtest.py
    GET  /benchmark                                       running?
    GET  /results/<name>/<tag>/summary.json, /results/<name>/<tag>.log, ...      files
    GET  /health

Results of a server go to results/<name>/.
"""
import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

import audio
import loadtest
import model
from bench_support import RESULTS, gpu_snapshot, prepare_gpu

HERE = Path(__file__).resolve().parent
PORT, ASR_PORT = int(os.getenv("PORT", "8000")), 6006
PROVIDER = os.getenv("PROVIDER", "cuda")
STATE = dict(server=None, benchmark=None, preflight=None)


@asynccontextmanager
async def lifespan(app):
    STATE["preflight"] = prepare_gpu()
    model.ensure()
    for name in audio.SETS:
        audio.load(name)
    yield
    for key in ("benchmark", "server"):
        stop(key)


app = FastAPI(lifespan=lifespan)
app.mount("/results", StaticFiles(directory=RESULTS), name="results")


def stop(key):
    job = STATE[key]
    if job and job["proc"].poll() is None:
        job["proc"].terminate()
        try:
            job["proc"].wait(timeout=15)
        except subprocess.TimeoutExpired:
            job["proc"].kill()


def running(key):
    job = STATE[key]
    return job is not None and job["proc"].poll() is None


def tee(proc, path):
    """Server output to a file, each line stamped with wall time (loadtest.py reads batch sizes from it)."""
    with path.open("a") as f:
        for line in proc.stdout:
            f.write(f"{time.time():.3f} {line}")
            f.flush()


def port_open():
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", ASR_PORT)) == 0


async def check():
    """One ~10 s utterance, three times: the server answers, and how fast when idle."""
    utt = audio.load("same")[0]
    conn = await loadtest.open_conn(f"ws://127.0.0.1:{ASR_PORT}")
    seconds = []
    for _ in range(3):
        t0, t1, text = await loadtest.recognize(conn, loadtest.payload(utt["samples"]))
        seconds.append(t1 - t0)
    await loadtest.close(conn)
    return dict(utt=utt["id"], audio_s=utt["seconds"], latency_s=seconds, text=text, reference=utt["text"])


@app.post("/server")
def start_server(name: str = Query("lite", pattern=r"^[A-Za-z0-9_.-]+$"), num_slots: int = Query(32, ge=1, le=1024),
                 steps: int = Query(8, ge=1, le=256), max_utterance_length: float = Query(300, gt=0),
                 window_seconds: float = Query(30, gt=0), window_extra: float = Query(2, ge=0),
                 admit_max: int = Query(16, ge=1, le=1024), admit_max_seconds: float = Query(240, gt=0),
                 max_pending_seconds: float = Query(120, ge=0),
                 gpu_mem_gb: float = Query(8, ge=0), num_threads: int = Query(2, ge=1, le=64),
                 cuda_graph: int = Query(1, ge=0, le=1), shrink_arena: int = Query(0, ge=0, le=1)):
    if running("benchmark"):
        raise HTTPException(409, "Benchmark running")
    stop("server")
    config = dict(num_slots=num_slots, steps=steps, max_utterance_length=max_utterance_length,
                  window_seconds=window_seconds, window_extra=window_extra,
                  admit_max=admit_max, admit_max_seconds=admit_max_seconds, max_pending_seconds=max_pending_seconds,
                  gpu_mem_gb=gpu_mem_gb, num_threads=num_threads, cuda_graph=cuda_graph, shrink_arena=shrink_arena)
    command = [sys.executable, "-u", "serve.py", f"--port={ASR_PORT}", f"--provider={PROVIDER}",
               *[f"--{k.replace('_', '-')}={v}" for k, v in config.items()]]
    folder = RESULTS / name
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "server.out"
    out.write_text(f"{time.time():.3f} $ {' '.join(command)}\n")
    proc = subprocess.Popen(command, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    threading.Thread(target=tee, args=(proc, out), daemon=True).start()
    t = time.time()
    while not port_open():
        if proc.poll() is not None:
            raise HTTPException(500, f"server exited with {proc.returncode}; see /results/{name}/server.out")
        if time.time() - t > 600:
            proc.kill()
            raise HTTPException(504, "server did not open its port in 600 s")
        time.sleep(0.5)
    STATE["server"] = dict(proc=proc, name=name, command=command, config=dict(config, provider=PROVIDER),
                           ready_s=time.time() - t, check=asyncio.run(check()))
    return server_status()


@app.get("/server")
def server_status():
    job = STATE["server"]
    if not job:
        return {"running": False}
    return {"running": running("server"), "exit_code": job["proc"].poll(), "pid": job["proc"].pid,
            **{k: job[k] for k in ("name", "command", "config", "ready_s", "check")}}


@app.post("/benchmark")
def benchmark(tag: str = Query(..., pattern=r"^[A-Za-z0-9_.-]+$"), mode: str = Query("burst", pattern="^(burst|sustain)$"),
              concurrency: str = Query("1,2,4,8,16", pattern=r"^\d+(,\d+)*$"),
              audio_set: str = Query("same", alias="audio", pattern="^(same|mixed|long|long-utts)$"),
              repeats: int = Query(5, ge=1, le=100), duration: float = Query(60, gt=0, le=1800),
              warmup: float = Query(10, ge=0), sla: float = Query(1.0, gt=0), stop_x: float = Query(5, gt=1),
              procs: int = Query(8, ge=1, le=64)):
    if not running("server"):
        raise HTTPException(409, "No server running: POST /server first")
    if running("benchmark"):
        raise HTTPException(409, "Benchmark already running")
    server = STATE["server"]
    folder = RESULTS / server["name"]
    meta = {k: server[k] for k in ("name", "command", "config", "check")}
    command = [sys.executable, "-u", "loadtest.py", "--url", f"ws://127.0.0.1:{ASR_PORT}", "--mode", mode,
               "--concurrency", concurrency, "--audio", audio_set, "--repeats", str(repeats),
               "--duration", str(duration), "--warmup", str(warmup), "--sla", str(sla), "--stop", str(stop_x),
               "--procs", str(procs), "--server-pid", str(server["proc"].pid),
               "--server-log", str(folder / "server.out"), "--meta", json.dumps(meta), "--out", str(folder / tag)]
    with (folder / f"{tag}.log").open("w") as f:
        print(f"benchmark_host={socket.gethostname()} server={server['name']}\n$ {' '.join(command[2:])}",
              file=f, flush=True)
        proc = subprocess.Popen(command, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    STATE["benchmark"] = dict(proc=proc, tag=tag, server=server["name"])
    return {"status": "started", "log": f"/results/{server['name']}/{tag}.log",
            "summary": f"/results/{server['name']}/{tag}/summary.json"}


@app.get("/benchmark")
def benchmark_status():
    job = STATE["benchmark"]
    if not job:
        return {"started": False}
    return {"started": True, "running": running("benchmark"), "exit_code": job["proc"].poll(),
            "tag": job["tag"], "server": job["server"]}


def pkg(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@app.get("/health")
def health():
    return {"status": "ok", "gpu": gpu_snapshot(), "cpu_count": os.cpu_count(), "provider": PROVIDER,
            "model": model.NAME, "server": server_status(), "preflight": STATE["preflight"], "python": sys.version,
            "versions": {p: pkg(p) for p in ("torch", "onnxruntime-gpu", "onnxruntime", "websockets", "numpy")}}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1)
