"""Load client for sherpa-onnx's offline websocket protocol: how many users at once, at what latency.

Speaks the protocol of sherpa-onnx's offline-websocket-client-decode-files-paralell.py: int32 sample
rate + int32 byte count + float32 samples, sent in 10240-byte messages; one JSON reply per
utterance; "Done" at the end. So the same client loads the upstream C++ server, the Python port
(server_python.py) and, later, the lite server.

Two modes, each swept over --concurrency (c = number of users). Every user keeps one connection
for the whole level, opened before the clock starts.

  burst    c users send one utterance each at the same instant; --repeats rounds after 2
           warm-up rounds. Latency = first byte sent -> reply received.
  sustain  c users talk without pause, cut into utterances; each utterance is sent the moment it
           ends, one in flight per user, users staggered. Delay = end of utterance -> reply
           received. It includes any backlog, so once the server cannot keep up the delay grows for
           the rest of the run. --duration seconds; utterances that end in the first --warmup
           seconds are not counted.

--audio picks the utterances (audio.py): "same" (LibriSpeech test-clean utterances of 9.75-10.25 s)
or "mixed" (300 random test-clean utterances, 1.4-33 s). User u's k-th request sends utterance
(u + k * c) mod n, so users that send together send different utterances.

A server may refuse an utterance (lite: {"error": "busy" | "too_long"}): that request is counted
as rejected, not as an error, and its user goes on with the next utterance. A level passes when no
request fails, at most --max-reject of them are rejected, and P90 latency (burst) or delay
(sustain) of the accepted ones is at most --sla seconds. The sweep stops after the first level
with errors, more than half rejected, or P90 above --stop x sla.

Every utterance of the set is first recognised alone, one at a time: that text is the reference
for "match" under load, and its word error rate against the LibriSpeech transcript is the WER
alone. Under load each reply is scored the same way. Users are spread over --procs processes so
that the client is not what saturates.

Output in --out: c<N>.jsonl (one row per request) and summary.json.
"""
import argparse
import asyncio
import json
import math
import multiprocessing as mp
import os
import random
import re
import resource
import socket
import struct
import time
import urllib.request
from pathlib import Path

from websockets.asyncio.client import connect

import audio
import model
from bench_support import Monitor, gpu_snapshot, wait_gpu_idle

CHUNK = 10240  # bytes per message, as the upstream client
BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 40)]  # seconds, for latency by utterance length


def payload(samples):
    buf = struct.pack("<ii", model.SR, samples.size * 4) + samples.tobytes()
    return [buf[i:i + CHUNK] for i in range(0, len(buf), CHUNK)]


async def open_conn(url):
    return await connect(url, compression=None, ping_interval=None, max_size=None, max_queue=None,
                         open_timeout=120, close_timeout=10)


class Rejected(Exception):
    """The server refused the utterance (overload, too long) and said so."""


async def recognize(conn, chunks, timeout=900):
    t0 = time.perf_counter()
    for chunk in chunks:
        await conn.send(chunk)
    reply = json.loads(await asyncio.wait_for(conn.recv(), timeout))
    if reply.get("error"):
        raise Rejected(f"{reply['error']}: {reply.get('message', '')}")
    return t0, time.perf_counter(), reply["text"].strip()


async def close(conn):
    try:
        await conn.send("Done")
        await conn.close()
    except Exception:  # noqa: BLE001 - the server may already have closed it
        pass


async def run_users(a, c, users, ref, barrier, wall_zero):
    """One process: these users, one level. Returns (rows, sustain start as wall time)."""
    utts = audio.load(a.audio)
    pick = lambda u, k: utts[(u + k * c) % len(utts)]  # noqa: E731
    conns = {u: await open_conn(a.url) for u in users}
    loop = asyncio.get_running_loop()
    rows = []

    async def send(u, utt, row):
        row.update(utt=utt["id"], length_s=utt["seconds"])
        try:
            if conns[u] is None:
                conns[u] = await open_conn(a.url)
            chunks = payload(utt["samples"])
            wall = time.time()
            t0, t1, text = await recognize(conns[u], chunks)
            errors, n_words = audio.word_errors(utt["text"], text)
            row.update(ok=True, t_send=wall - wall_zero, t_done=wall + t1 - t0 - wall_zero, latency_s=t1 - t0,
                       rtf=(t1 - t0) / utt["seconds"], match=text == ref[utt["id"]], word_errors=errors,
                       words=n_words, text=text)
        except Rejected as e:
            row.update(ok=False, rejected=True, error=str(e), t_done=time.time() - wall_zero)
        except Exception as e:  # noqa: BLE001 - one failed user must not stop the others
            row.update(ok=False, error=f"{type(e).__name__}: {e}")
            conns[u] = None
        return row

    start = None
    if a.mode == "burst":
        for rnd in range(-2, a.repeats):
            await loop.run_in_executor(None, barrier.wait)  # every process ready: go together
            rows += await asyncio.gather(*(send(u, pick(u, rnd), dict(mode="burst", concurrency=c, round=rnd + 1, user=u))
                                           for u in users))
    else:
        await loop.run_in_executor(None, barrier.wait)
        start = time.time() + 1.0
        end = start + a.duration

        async def user(u):
            rng = random.Random(f"{a.seed}-{c}-{u}")
            # Already talking before start: the first utterance ends at a uniform time in
            # [start, start + L), so requests arrive at a steady rate from the first second.
            spoke_from, k = start - rng.random() * pick(u, 0)["seconds"], 0
            while True:
                utt = pick(u, k)
                due = spoke_from + utt["seconds"]  # the utterance ends: send it now
                if due > end:
                    return
                await asyncio.sleep(max(0.0, due - time.time()))
                row = dict(mode="sustain", concurrency=c, user=u, k=k, t_due=due - wall_zero,
                           warm=due < start + a.warmup)
                row = await send(u, utt, row)
                if row["ok"]:
                    row["delay_s"] = row["t_done"] + wall_zero - due
                rows.append(row)
                spoke_from, k = due, k + 1  # talks on without a pause

        await asyncio.gather(*(user(u) for u in users))
    await asyncio.gather(*(close(conn) for conn in conns.values() if conn))
    return rows, start


def worker(a, c, users, ref, barrier, wall_zero, out):
    try:
        out.put(asyncio.run(run_users(a, c, users, ref, barrier, wall_zero)))
    except BaseException as e:  # noqa: BLE001 - report instead of hanging the parent
        barrier.abort()
        out.put(e)


def run_level(a, c, ref, wall_zero):
    ctx = mp.get_context("spawn")
    procs = min(a.procs, c)
    barrier, out = ctx.Barrier(procs), ctx.Queue()
    ps = [ctx.Process(target=worker, args=(a, c, list(range(p, c, procs)), ref, barrier, wall_zero, out))
          for p in range(procs)]
    for p in ps:
        p.start()
    results = [out.get() for _ in ps]
    for p in ps:
        p.join()
    errors = [r for r in results if isinstance(r, BaseException)]
    if errors:
        raise RuntimeError(f"client process failed: {errors[0]!r}")
    rows = [row for rs, _ in results for row in rs]
    starts = [s for _, s in results if s is not None]
    window = (min(starts) + a.warmup - wall_zero, min(starts) + a.duration - wall_zero) if starts else None
    return [r for r in rows if r.get("round", 1) > 0], window


def server_stats(url):
    """The lite server's counters (GET /health on its websocket port); None for a server without it."""
    try:
        with urllib.request.urlopen(url.replace("ws://", "http://", 1).rstrip("/") + "/health", timeout=5) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001 - sherpa has no /health
        return None


def stats_delta(before, after):
    """Per level: prefills, their mean size, the share of the time the prefill thread and the greedy
    thread were busy (they overlap), and the share of their busy time spent prefilling."""
    if not before or not after:
        return None
    keys = ("admits", "admitted", "admit_s", "ticks", "steps", "step_s", "clock_s", "vram_releases")
    d = {k: after.get(k, 0) - before.get(k, 0) for k in keys}
    busy = d["admit_s"] + d["step_s"]
    return dict(d, prefill_size_mean=d["admitted"] / d["admits"] if d["admits"] else None,
                admit_share=d["admit_s"] / busy if busy else None,
                prefill_busy=d["admit_s"] / d["clock_s"] if d["clock_s"] else None,
                greedy_busy=d["step_s"] / d["clock_s"] if d["clock_s"] else None,
                tick_ms=1e3 * d["step_s"] / d["ticks"] if d["ticks"] else None,
                gpu_used_mb=after.get("gpu_used_mb"), torch_reserved_mb=after.get("torch_reserved_mb"))


def q(xs, p):
    """Linear interpolation at (n-1)p, as in vieneu's analyze.py."""
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def batches(path, t0, t1):
    """Batch sizes the server logged ("size: N", timestamped by bench.py) between two wall times."""
    if not path or not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(errors="replace").splitlines():
        m = re.match(r"(\d+\.\d+) .*\bsize: (\d+)\s*$", line)
        if m and t0 <= float(m.group(1)) <= t1:
            out.append(int(m.group(2)))
    return out


def summarize(a, c, rows, window, monitor, wall_zero, client_cpu_s):
    metric = "latency_s" if a.mode == "burst" else "delay_s"
    counted = [r for r in rows if not r.get("warm")]
    ok = [r for r in counted if r["ok"]]
    xs = [r[metric] for r in ok]
    if a.mode == "burst":
        lo = min((r["t_send"] for r in ok), default=0)
        hi = max((r["t_done"] for r in ok), default=0)
        rounds = {}
        for r in ok:
            s, e = rounds.get(r["round"], (math.inf, -math.inf))
            rounds[r["round"]] = (min(s, r["t_send"]), max(e, r["t_done"]))
        busy = sum(e - s for s, e in rounds.values())
        audio_per_s = sum(r["length_s"] for r in ok) / busy if busy else 0
    else:
        lo, hi = window
        audio_per_s = sum(r["length_s"] for r in rows if r["ok"] and lo <= r["t_done"] <= hi) / (hi - lo)
    samples = monitor.between(wall_zero + lo, wall_zero + hi)
    col = lambda k: [s[k] for s in samples if k in s]  # noqa: E731
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    sizes = batches(a.server_log, wall_zero + lo, wall_zero + hi)
    p90 = q(xs, 0.9)
    level = dict(
        concurrency=c, n=len(counted), rejected=sum(1 for r in counted if r.get("rejected")),
        errors=sum(1 for r in counted if not r["ok"] and not r.get("rejected")),
        error_samples=sorted({r["error"] for r in counted if not r["ok"]})[:3],
        metric=metric, p50_s=q(xs, 0.5), p90_s=p90, p99_s=q(xs, 0.99), max_s=max(xs, default=None),
        rtf_p50=q([r["rtf"] for r in ok], 0.5), rtf_p90=q([r["rtf"] for r in ok], 0.9),
        audio_s_per_s=audio_per_s, offered_audio_s_per_s=c if a.mode == "sustain" else None,
        match=mean([r["match"] for r in ok]),
        wer=sum(r["word_errors"] for r in ok) / max(1, sum(r["words"] for r in ok)),
        by_length={f"{lo_s}-{hi_s}": dict(n=len(ys), p50_s=q(ys, 0.5), p90_s=q(ys, 0.9))
                   for lo_s, hi_s in BINS for ys in [[r[metric] for r in ok if lo_s <= r["length_s"] < hi_s]]},
        batches=len(sizes), batch_size_mean=mean(sizes),
        gpu_util_mean=mean(col("gpu_util")), gpu_util_max=max(col("gpu_util"), default=None),
        gpu_mem_max_mb=max(col("gpu_mem_mb"), default=None),
        cpu_util_mean=mean(col("cpu_util")), server_cpu_cores_mean=mean(col("proc_cpu_cores")),
        server_cpu_cores_max=max(col("proc_cpu_cores"), default=None),
        client_cpu_cores=client_cpu_s / (hi - lo) if hi > lo else None)
    level["reject_rate"] = level["rejected"] / max(1, level["n"])
    level["passed"] = (level["errors"] == 0 and level["reject_rate"] <= a.max_reject
                       and p90 is not None and p90 <= a.sla)
    return level


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://127.0.0.1:6006")
    ap.add_argument("--mode", choices=["burst", "sustain"], default="burst")
    ap.add_argument("--concurrency", default="1,2,4,8,16", help="comma-separated user counts, swept in order")
    ap.add_argument("--audio", choices=["same", "mixed", "long", "long-utts"], default="same", help="utterance set (audio.py)")
    ap.add_argument("--repeats", type=int, default=5, help="burst: measured rounds per level")
    ap.add_argument("--duration", type=float, default=60, help="sustain: seconds per level")
    ap.add_argument("--warmup", type=float, default=10, help="sustain: seconds not counted at the start")
    ap.add_argument("--sla", type=float, default=1.0, help="pass: P90 latency/delay <= this many seconds")
    ap.add_argument("--stop", type=float, default=5, help="stop the sweep once P90 > stop x sla")
    ap.add_argument("--max-reject", type=float, default=0.01, help="pass: at most this share rejected")
    ap.add_argument("--procs", type=int, default=min(8, os.cpu_count() or 1), help="client processes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--server-pid", type=int, help="sample this process's CPU use")
    ap.add_argument("--server-log", help="server output timestamped by bench.py, for batch sizes")
    ap.add_argument("--meta", default="{}", help="JSON stored as-is in summary.json (server config)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    levels = [int(x) for x in a.concurrency.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("c*.jsonl"):
        old.unlink()
    utts = audio.load(a.audio)

    async def reference():
        """Each utterance alone, one at a time; also the first run of every input shape."""
        conn = await open_conn(a.url)
        texts, errors, n_words, seconds = {}, 0, 0, []
        for utt in utts:
            t0, t1, texts[utt["id"]] = await recognize(conn, payload(utt["samples"]))
            e, n = audio.word_errors(utt["text"], texts[utt["id"]])
            errors, n_words = errors + e, n_words + n
            seconds.append(t1 - t0)
        await close(conn)
        return texts, errors / n_words, seconds

    ref, wer_alone, alone_s = asyncio.run(reference())
    print(f"reference: {len(utts)} utterances of {a.audio!r} alone, WER {wer_alone:.2%}, "
          f"latency P50 {q(alone_s, 0.5):.3f} s", flush=True)
    summary = dict(mode=a.mode, audio=a.audio, utterances=len(utts),
                   audio_s=dict(min=min(u["seconds"] for u in utts), max=max(u["seconds"] for u in utts),
                                mean=sum(u["seconds"] for u in utts) / len(utts)),
                   levels_requested=levels, repeats=a.repeats, duration_s=a.duration, warmup_s=a.warmup,
                   sla_s=a.sla, procs=a.procs, seed=a.seed, url=a.url, hostname=socket.gethostname(),
                   meta=json.loads(a.meta), wer_alone=wer_alone, alone_latency_s=alone_s, reference=ref,
                   gpu_before=gpu_snapshot(), levels=[])
    wall_zero = time.time()
    with Monitor(a.server_pid) as monitor:
        for c in levels:
            _, idle = wait_gpu_idle(samples=3, limit=60)
            cpu0 = resource.getrusage(resource.RUSAGE_CHILDREN)
            before = server_stats(a.url)
            rows, window = run_level(a, c, ref, wall_zero)
            after = server_stats(a.url)
            cpu1 = resource.getrusage(resource.RUSAGE_CHILDREN)
            client_cpu_s = cpu1.ru_utime + cpu1.ru_stime - cpu0.ru_utime - cpu0.ru_stime
            level = summarize(a, c, rows, window, monitor, wall_zero, client_cpu_s)
            level["gpu_idle_before"] = idle
            level["server_stats"] = stats_delta(before, after)
            with (out / f"c{c}.jsonl").open("w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            summary["levels"].append(level)
            (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            f3 = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
            f0 = lambda x: "-" if x is None else f"{x:.0f}"  # noqa: E731
            print(f"{a.mode} c={c}: n={level['n']} err={level['errors']} {level['metric']} "
                  f"P50 {f3(level['p50_s'])} P90 {f3(level['p90_s'])} P99 {f3(level['p99_s'])} | "
                  f"RTF P90 {f3(level['rtf_p90'])} | {level['audio_s_per_s']:.1f} audio-s/s | "
                  f"batch {f3(level['batch_size_mean'])} | GPU {f0(level['gpu_util_mean'])}% "
                  f"server CPU {f3(level['server_cpu_cores_mean'])} cores client {f3(level['client_cpu_cores'])} | "
                  f"match {f3(level['match'])} WER {level['wer']:.2%} | rejected {level['reject_rate']:.1%} | "
                  + (f"prefill {f3(ss['prefill_size_mean'])} busy prefill {f3(ss.get('prefill_busy'))} "
                     f"greedy {f3(ss.get('greedy_busy'))} tick {f3(ss['tick_ms'])}ms "
                     f"VRAM {f0(ss['gpu_used_mb'])}MB max {f0(level['gpu_mem_max_mb'])}MB torch {f0(ss['torch_reserved_mb'])}MB "
                     f"releases {ss.get('vram_releases', 0)} | " if (ss := level["server_stats"]) else "")
                  + f"{'PASS' if level['passed'] else 'FAIL'}",
                  flush=True)
            if level["errors"]:
                print(f"errors: {level['error_samples']}", flush=True)
            if level["errors"] or level["reject_rate"] > 0.5 or (level["p90_s"] or math.inf) > a.stop * a.sla:
                print(f"stop: c={c} had errors or P90 > {a.stop} x SLA", flush=True)
                break
        summary["monitor"] = monitor.rows
    passed = [lv["concurrency"] for lv in summary["levels"] if lv["passed"]]
    summary["max_concurrency_passed"] = max(passed, default=None)
    summary["gpu_after"] = gpu_snapshot()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"done: max users with P90 <= {a.sla}s and no errors: {summary['max_concurrency_passed']}", flush=True)


if __name__ == "__main__":
    main()
