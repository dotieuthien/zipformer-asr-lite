"""Bench controller on the GPU host (vkong app_port 8000). It is not in the request path: it starts
the server under test on port 6006 (server_cpp.sh or server_python.py, with the same flags), runs
loadtest.py against it on this host, and serves the results.

    POST /server?impl=cpp|python&max_batch_size=5&num_work_threads=3&num_threads=2   (re)start it
    GET  /server                                                                     what runs
    POST /benchmark?tag=burst-same&mode=burst&concurrency=1,2,4&audio=same&...       run loadtest.py
    GET  /benchmark                                                                  running?
    GET  /results/<server>/<tag>/summary.json, /results/<server>/<tag>.log, ...      files
    GET  /health

Defaults of /server are the C++ server's own defaults. Results of a server go to
results/<impl>-b<max_batch_size>-w<num_work_threads>-t<num_threads>/.
"""
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

import audio
import loadtest
import model
from bench_support import RESULTS, gpu_snapshot, prepare_gpu

HERE = Path(__file__).resolve().parent
PORT = int(os.getenv("PORT", "8000"))
ASR_PORT = 6006
PROVIDER = os.getenv("PROVIDER", "cuda")
STATE = dict(server=None, benchmark=None, preflight=None)


@asynccontextmanager
async def lifespan(app):
    STATE["preflight"] = prepare_gpu()
    model.ensure()
    model.ensure_cpp()
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
            job["proc"].wait()


def tee(proc, path):
    """Copy the server's stdout+stderr to a file, each line stamped with wall time (loadtest.py reads
    the batch sizes the server logs from it)."""
    with path.open("a") as f:
        for line in proc.stdout:
            f.write(f"{time.time():.3f} {line}")
            f.flush()


def port_open():
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", ASR_PORT)) == 0


async def warm_up():
    """One ~10 s utterance three times, one at a time, on one connection: the first CUDA runs are slow."""
    utt = audio.load("same")[0]
    conn = await loadtest.open_conn(f"ws://127.0.0.1:{ASR_PORT}")
    chunks, seconds = loadtest.payload(utt["samples"]), []
    for _ in range(3):
        t0, t1, text = await loadtest.recognize(conn, chunks)
        seconds.append(t1 - t0)
    await loadtest.close(conn)
    return dict(utt=utt["id"], audio_s=utt["seconds"], latency_s=seconds, text=text, reference=utt["text"])


@app.post("/server")
def start_server(impl: str = Query(..., pattern="^(cpp|python)$"), max_batch_size: int = Query(5, ge=1, le=256),
                 num_work_threads: int = Query(3, ge=1, le=64), num_threads: int = Query(2, ge=1, le=64)):
    job = STATE["benchmark"]
    if job and job["proc"].poll() is None:
        raise HTTPException(409, "Benchmark running")
    stop("server")
    name = f"{impl}-b{max_batch_size}-w{num_work_threads}-t{num_threads}"
    folder = RESULTS / name
    folder.mkdir(parents=True, exist_ok=True)
    paths = model.ensure()
    flags = [f"--port={ASR_PORT}", f"--provider={PROVIDER}", f"--tokens={paths['tokens']}",
             f"--encoder={paths['encoder']}", f"--decoder={paths['decoder']}", f"--joiner={paths['joiner']}",
             f"--num-threads={num_threads}", f"--num-work-threads={num_work_threads}",
             f"--max-batch-size={max_batch_size}", f"--log-file={folder / 'log.txt'}"]
    command = ["bash", "server_cpp.sh", *flags] if impl == "cpp" else [sys.executable, "-u", "server_python.py", *flags]
    out = folder / "server.out"
    (folder / "log.txt").unlink(missing_ok=True)  # a fresh log per start; C++ appends to it
    with out.open("w") as f:
        f.write(f"{time.time():.3f} $ {' '.join(command)}\n")
    proc = subprocess.Popen(command, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1)
    threading.Thread(target=tee, args=(proc, out), daemon=True).start()
    t = time.time()
    while not port_open():
        if proc.poll() is not None:
            raise HTTPException(500, f"server exited with {proc.returncode}; see /results/{name}/server.out")
        if time.time() - t > 300:
            proc.kill()
            raise HTTPException(504, "server did not open its port in 300 s")
        time.sleep(0.5)
    ready_s = time.time() - t
    warm = asyncio.run(warm_up())
    STATE["server"] = dict(proc=proc, name=name, impl=impl, command=command, started=t, ready_s=ready_s,
                           warmup=warm, config=dict(max_batch_size=max_batch_size, num_work_threads=num_work_threads,
                                                    num_threads=num_threads, provider=PROVIDER))
    return server_status()


@app.get("/server")
def server_status():
    job = STATE["server"]
    if not job:
        return {"running": False}
    return {"running": job["proc"].poll() is None, "exit_code": job["proc"].poll(), "pid": job["proc"].pid,
            **{k: job[k] for k in ("name", "impl", "command", "ready_s", "warmup", "config")}}


@app.post("/benchmark")
def benchmark(tag: str = Query(..., pattern=r"^[A-Za-z0-9_.-]+$"), mode: str = Query("burst", pattern="^(burst|sustain)$"),
              concurrency: str = Query("1,2,4,8,16", pattern=r"^\d+(,\d+)*$"),
              audio_set: str = Query("same", alias="audio", pattern="^(same|mixed)$"),
              repeats: int = Query(5, ge=1, le=100), duration: float = Query(60, gt=0, le=1800),
              warmup: float = Query(10, ge=0), sla: float = Query(1.0, gt=0), stop_x: float = Query(5, gt=1),
              procs: int = Query(8, ge=1, le=64)):
    server = STATE["server"]
    if not server or server["proc"].poll() is not None:
        raise HTTPException(409, "No server running: POST /server first")
    job = STATE["benchmark"]
    if job and job["proc"].poll() is None:
        raise HTTPException(409, "Benchmark already running")
    folder = RESULTS / server["name"]
    meta = {k: server[k] for k in ("name", "impl", "command", "config", "warmup")}
    command = [sys.executable, "-u", "loadtest.py", "--url", f"ws://127.0.0.1:{ASR_PORT}", "--mode", mode,
               "--concurrency", concurrency, "--audio", audio_set, "--repeats", str(repeats),
               "--duration", str(duration), "--warmup", str(warmup), "--sla", str(sla), "--stop", str(stop_x),
               "--procs", str(procs), "--server-pid", str(server["proc"].pid),
               "--server-log", str(folder / "server.out"), "--meta", json.dumps(meta), "--out", str(folder / tag)]
    log = folder / f"{tag}.log"
    with log.open("w") as f:
        print(f"benchmark_host={socket.gethostname()} server={server['name']} pid={server['proc'].pid}\n"
              f"$ {' '.join(command[2:])}", file=f, flush=True)
        proc = subprocess.Popen(command, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    STATE["benchmark"] = dict(proc=proc, tag=tag, server=server["name"])
    return {"status": "started", "log": f"/results/{server['name']}/{tag}.log",
            "summary": f"/results/{server['name']}/{tag}/summary.json"}


@app.get("/benchmark")
def benchmark_status():
    job = STATE["benchmark"]
    if not job:
        return {"started": False}
    return {"started": True, "running": job["proc"].poll() is None, "exit_code": job["proc"].poll(),
            "tag": job["tag"], "server": job["server"]}


def pkg(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@app.get("/health")
def health():
    snap = gpu_snapshot()
    return {"status": "ok", "gpu": snap, "cpu_count": os.cpu_count(), "provider": PROVIDER, "model": model.NAME,
            "cpp_server": str(model.CPP_SERVER), "server": server_status(), "preflight": STATE["preflight"],
            "versions": {p: pkg(p) for p in ("sherpa-onnx", "websockets", "numpy", "fastapi", "uvicorn")},
            "python": sys.version}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1)
