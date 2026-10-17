"""Python port of sherpa-onnx's C++ offline websocket server, transducer models only.

Ported from sherpa-onnx v1.13.8, sherpa-onnx/csrc/offline-websocket-server.cc and
offline-websocket-server-impl.cc: same protocol, flags, defaults and scheduling. Compare it with the
C++ binary (server_cpp.sh) under the same load before trusting it as the baseline.

Protocol, any number of utterances per connection: binary messages carrying int32 sample rate,
int32 byte count, then float32 samples, split however the client likes; the reply is the result as
JSON text. The text message "Done" closes the connection.

Scheduling: each complete utterance is pushed to one queue and one Decode task is posted to a pool
of --num-work-threads threads. Decode takes up to --max-batch-size utterances from the queue,
computes their features while holding the queue lock, then runs decode_streams on the batch. The
network side is one asyncio loop, as C++ with --num-io-threads=1 (its default).

    python server_python.py --provider=cuda --port=6006
"""
import argparse
import asyncio
import collections
import os
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

import model


def get_args():
    paths = model.ensure()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=6006)
    ap.add_argument("--num-io-threads", type=int, default=1, help="only 1 is supported")
    ap.add_argument("--num-work-threads", type=int, default=3)
    ap.add_argument("--max-batch-size", type=int, default=5)
    ap.add_argument("--max-utterance-length", type=float, default=300, help="seconds")
    ap.add_argument("--log-file", default="./log.txt")
    ap.add_argument("--encoder", default=paths["encoder"])
    ap.add_argument("--decoder", default=paths["decoder"])
    ap.add_argument("--joiner", default=paths["joiner"])
    ap.add_argument("--tokens", default=paths["tokens"])
    ap.add_argument("--num-threads", type=int, default=2)
    ap.add_argument("--provider", default="cpu")
    ap.add_argument("--decoding-method", default="greedy_search")
    ap.add_argument("--feat-dim", type=int, default=80)
    a = ap.parse_args()
    if a.num_io_threads != 1:
        ap.error("--num-io-threads: only 1 is supported")
    if a.max_batch_size <= 0 or a.max_utterance_length <= 0:
        ap.error("expect --max-batch-size > 0 and --max-utterance-length > 0")
    return a


def loge(line):
    """SHERPA_ONNX_LOGE: stderr."""
    print(line, file=sys.stderr, flush=True)


class Log:
    """The websocketpp log, which C++ tees to stdout and --log-file (append)."""

    def __init__(self, path):
        self.file = open(path, "a", buffering=1)

    def __call__(self, line):
        print(line, flush=True)
        self.file.write(line + "\n")


class Decoder:
    """OfflineWebsocketDecoder."""

    def __init__(self, recognizer, max_batch_size, send):
        self.recognizer, self.max_batch_size, self.send = recognizer, max_batch_size, send
        self.lock = threading.Lock()
        self.streams = collections.deque()

    def push(self, conn, data):
        with self.lock:
            self.streams.append((conn, data))

    def decode(self):
        with self.lock:
            if not self.streams:
                return
            size = min(len(self.streams), self.max_batch_size)
            loge(f"size: {size}")
            items = [self.streams.popleft() for _ in range(size)]
            ss = []
            for _, (sample_rate, samples) in items:
                s = self.recognizer.create_stream()
                s.accept_waveform(sample_rate, samples)
                ss.append(s)
        self.recognizer.decode_streams(ss)  # thread-safe, as in C++
        for (conn, _), s in zip(items, ss):
            self.send(conn, str(s.result))  # str() is the C++ AsJsonString()


class Server:
    """OfflineWebsocketServer."""

    def __init__(self, a, recognizer):
        self.a = a
        self.log = Log(a.log_file)
        self.pool = ThreadPoolExecutor(max_workers=a.num_work_threads, thread_name_prefix="work")
        self.decoder = Decoder(recognizer, a.max_batch_size, self.send)
        self.connections = 0
        self.loop = None

    def send(self, conn, text):
        """From a work thread; C++ posts the send to the connection io_context."""
        asyncio.run_coroutine_threadsafe(self._send(conn, text), self.loop)

    async def _send(self, conn, text):
        try:
            await conn.send(text)
        except ConnectionClosed as e:
            self.log(str(e))

    def post_decode(self):
        """asio::post(io_work_, Decode)."""
        future = self.loop.run_in_executor(self.pool, self.decoder.decode)
        future.add_done_callback(lambda f: f.exception() and self.log(f"decode failed: {f.exception()!r}"))

    async def handle(self, conn):
        self.connections += 1
        loge(f"Number of active connections: {self.connections}")
        sample_rate = expected = cur = 0
        data = None
        try:
            async for payload in conn:
                if isinstance(payload, str):
                    reason = "Done" if payload == "Done" else f"Invalid payload: {payload}"
                    await conn.close(1000, reason)
                    break
                if expected == 0:
                    if len(payload) < 8:
                        await conn.close(1000, "Payload is too short")
                        break
                    sample_rate, expected = struct.unpack_from("<ii", payload)
                    if expected > int(self.a.max_utterance_length * sample_rate * 4):
                        await conn.close(1009, f"Max utterance length is configured to "
                                               f"{self.a.max_utterance_length:g} seconds, received length is "
                                               f"{expected / 4 / sample_rate:g} seconds. Payload is too large!")
                        break
                    data = bytearray(expected)
                    data[:len(payload) - 8] = memoryview(payload)[8:]
                    cur = len(payload) - 8
                else:
                    data[cur:cur + len(payload)] = payload
                    cur += len(payload)
                if expected == cur:
                    # Push this utterance and reset, so the client can send the next one on this connection.
                    self.decoder.push(conn, (sample_rate, np.frombuffer(data, dtype=np.float32)))
                    sample_rate = expected = cur = 0
                    self.post_decode()
        except ConnectionClosed:
            pass
        finally:
            self.connections -= 1
            loge(f"Number of active connections: {self.connections}")

    async def run(self):
        self.loop = asyncio.get_running_loop()
        # websocketpp, as built into sherpa-onnx: no compression, no keepalive pings, 32 MB max message.
        async with serve(self.handle, "0.0.0.0", self.a.port, compression=None, ping_interval=None,
                         max_size=32_000_000, max_queue=None):
            loge("Started!")
            loge(f"Listening on: {self.a.port}")
            loge(f"Number of work threads: {self.a.num_work_threads}")
            await asyncio.Future()


def cuda_env():
    """As server_cpp.sh: CUDA 12 / cuDNN 9 from the nvidia-* wheels on LD_LIBRARY_PATH. The loader reads
    it only at process start, so restart this process once."""
    dirs = model.cuda_lib_dirs()
    have = [d for d in os.getenv("LD_LIBRARY_PATH", "").split(":") if d]
    if sys.platform == "linux" and dirs and not set(dirs) <= set(have):
        os.environ["LD_LIBRARY_PATH"] = ":".join(dirs + have)
        os.execv(sys.executable, sys.orig_argv)


def main():
    cuda_env()
    a = get_args()
    import sherpa_onnx  # after cuda_env()

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=a.encoder, decoder=a.decoder, joiner=a.joiner, tokens=a.tokens, num_threads=a.num_threads,
        sample_rate=model.SR, feature_dim=a.feat_dim, decoding_method=a.decoding_method, provider=a.provider)
    asyncio.run(Server(a, recognizer).run())


if __name__ == "__main__":
    main()
