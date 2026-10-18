"""Zipformer ASR lite server.

Protocol, the same as sherpa-onnx's offline websocket server, so its clients work unchanged: binary
messages carrying int32 sample rate, int32 byte count, then float32 samples, split however the
client likes; one JSON reply per utterance; the text "Done" closes the connection.

New: a refused utterance gets an immediate reply {"text": "", "error": "busy" | "too_long", ...}
and the connection stays open. GET /health returns the engine's state. sherpa's other flags
(--max-batch-size, --num-work-threads, model paths, ...) are accepted and ignored, with a warning.

    python serve.py                       # GPU, port 6006
    python serve.py --provider=cpu        # for a local test
"""
import argparse
import asyncio
import http
import json
import struct
import sys
import time

import numpy as np
import torch  # before onnxruntime: it loads the CUDA libraries both use
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

import model
from zipformer_lite.engine import Busy, Engine, TooLong
from zipformer_lite.transducer import Transducer


def get_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=6006)
    ap.add_argument("--provider", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--num-threads", type=int, default=2, help="onnxruntime threads")
    ap.add_argument("--num-slots", type=int, default=32, help="utterances decoded at the same time")
    ap.add_argument("--steps", type=int, default=8, help="greedy steps per tick, between admits")
    ap.add_argument("--max-utterance-length", type=float, default=300, help="refuse longer requests, seconds")
    ap.add_argument("--window-seconds", type=float, default=30, help="longer requests are cut into windows of this")
    ap.add_argument("--window-extra", type=float, default=2, help="seconds neighbouring windows share on each side")
    ap.add_argument("--admit-max", type=int, default=16, help="utterances per prefill, at most")
    ap.add_argument("--admit-max-seconds", type=float, default=240, help="padded audio per prefill, at most")
    ap.add_argument("--max-pending-seconds", type=float, default=120,
                    help="refuse when this much audio is already waiting; 0 = never")
    ap.add_argument("--gpu-mem-gb", type=float, default=8, help="onnxruntime memory limit; 0 = none")
    ap.add_argument("--cuda-graph", type=int, default=1, help="1: replay the greedy steps as a CUDA graph")
    ap.add_argument("--vram-release-gb", type=float, default=0,
                    help="when the GPU's used memory is over this, the next encoder run hands onnxruntime's free memory "
                         "back to CUDA; 0 never does")
    a, unknown = ap.parse_known_args()
    if unknown:
        print(f"ignoring sherpa flags: {' '.join(unknown)}", file=sys.stderr, flush=True)
    return a


def gpu_used_mb():
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()  # the whole GPU: torch, onnxruntime and the CUDA context
    return (total - free) / 2**20


def torch_reserved_mb():
    return torch.cuda.memory_reserved() / 2**20 if torch.cuda.is_available() else None


class Server:
    def __init__(self, asr, engine, port):
        self.asr, self.engine, self.port = asr, engine, port
        self.connections = 0
        self.started = time.time()

    async def reply(self, conn, samples, sample_rate):
        try:
            result = self.asr.result(*await asyncio.wrap_future(self.engine.submit(samples, sample_rate)))
        except Busy as e:
            result = dict(text="", error="busy", message=str(e), retry_after_s=round(e.wait_s, 2))
        except TooLong as e:
            result = dict(text="", error="too_long", message=str(e))
        except Exception as e:  # noqa: BLE001 - the batch failed: say so, keep the connection
            result = dict(text="", error="failed", message=repr(e))
        try:
            await conn.send(json.dumps(result, ensure_ascii=False))
        except ConnectionClosed:
            pass

    async def handle(self, conn):
        self.connections += 1
        expected = cur = sample_rate = 0
        buf = None
        try:
            async for msg in conn:
                if isinstance(msg, str):
                    await conn.close(1000, "Done" if msg == "Done" else f"Invalid payload: {msg}")
                    break
                if expected == 0:  # first message of an utterance: header, then samples
                    if len(msg) < 8:
                        await conn.close(1000, "Payload is too short")
                        break
                    sample_rate, expected = struct.unpack_from("<ii", msg)
                    buf, msg = bytearray(expected), memoryview(msg)[8:]
                buf[cur:cur + len(msg)] = msg
                cur += len(msg)
                if cur == expected:
                    asyncio.create_task(self.reply(conn, np.frombuffer(buf, dtype=np.float32), sample_rate))
                    expected = cur = 0
        except ConnectionClosed:
            pass
        finally:
            self.connections -= 1

    def http(self, conn, request):
        if request.path != "/health":
            return None  # carry on with the websocket handshake
        body = dict(status="ok", connections=self.connections, uptime_s=time.time() - self.started,
                    gpu_used_mb=gpu_used_mb(), torch_reserved_mb=torch_reserved_mb(), **self.engine.health())
        return conn.respond(http.HTTPStatus.OK, json.dumps(body) + "\n")

    async def run(self):
        async with serve(self.handle, "0.0.0.0", self.port, compression=None, ping_interval=None,
                         max_size=32_000_000, max_queue=None, process_request=self.http):
            print(f"Listening on: {self.port}", file=sys.stderr, flush=True)
            await asyncio.Future()


def main():
    a = get_args()
    t = time.perf_counter()
    asr = Transducer(model.ensure(), device=a.provider, gpu_mem_gb=a.gpu_mem_gb, num_threads=a.num_threads,
                     release_gb=a.vram_release_gb)
    engine = Engine(asr, num_slots=a.num_slots, steps=a.steps, max_seconds=a.max_utterance_length,
                    window_s=a.window_seconds, extra_s=a.window_extra,
                    admit_max=a.admit_max, admit_max_seconds=a.admit_max_seconds, max_pending_s=a.max_pending_seconds,
                    cuda_graph=bool(a.cuda_graph))
    shapes = engine.warm_up() if a.provider == "cuda" else []
    print(f"model ready in {time.perf_counter() - t:.1f}s; warmed prefills (utterances, padded s): {shapes}; "
          f"GPU memory used {gpu_used_mb()} MB, of which torch {torch_reserved_mb()} MB", file=sys.stderr, flush=True)
    asyncio.run(Server(asr, engine, a.port).run())


if __name__ == "__main__":
    main()
