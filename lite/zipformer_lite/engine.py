"""Continuous batching over fixed GPU slots: prefill once per group, then decode every slot one step
at a time; a slot answers as soon as its window is done.

Long audio, as icefall's long_file_recog does it: a request is cut into windows of window_s (30 s)
every window_s - 2 x extra_s (26 s), so neighbours overlap by 2 x extra_s. Each window is decoded on
its own, in its own slot. A window keeps the tokens from extra_s after its start to extra_s before
its end (from the very start for the first window, to the very end for the last), so the kept
parts tile the audio, and the request's answer is the kept tokens of its windows, in order. A
request of up to window_s is one window and keeps everything.

One worker thread owns the GPU. Each tick:

1. Admit. If slots are free, take up to min(free slots, admit_max) pending windows in arrival
   order, as vieneu lite does. Run fbank and the encoder ONCE for the group, padded to its longest
   window, and copy each window's encoder output into its own slot. Padding exists only in this
   prefill: the greedy loop has none.
2. Step, `steps` times. One greedy step for all slots together: joiner on each slot's current
   frame, argmax, decoder on the new context. Sherpa's rule: at most one token per frame, <unk>
   counts as blank. Each slot has its own frame index; empty and finished slots run along,
   masked, so a step is the same few kernels whatever the slots hold. That is what lets the
   `steps` steps be recorded once as a CUDA graph and replayed with one call per tick.
3. Collect. Copy to the host which slots are done and free them at once; a request is answered
   when its last window is done.

Fixed GPU memory. The greedy loop's memory is allocated once (slots, CUDA graph). What a prefill
allocates depends on its shape, g windows x the longest one's length, which is bounded: no window
is longer than window_s, and a group stays within admit_max windows and admit_max_seconds of
padded audio. warm_up() runs the boundary, for every g the longest group allowed, so any later
prefill is no larger in either dimension than one already run: memory peaks before serving.
(The encoder ONNX takes its mask from the longest x_lens, so a group can only be padded to its
longest window, not to a fixed set of lengths.)

Admission control: requests longer than max_seconds are refused (TooLong), and when the audio
already waiting is over max_pending_s a new request is refused at once (Busy) instead of waiting
seconds for a late answer. max_pending_s=0 never refuses.
"""
import collections
import math
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field

import numpy as np
import torch

from .fbank import SR, num_frames
from .transducer import FRAME_S, START, out_frames


class Busy(Exception):
    pass


class TooLong(Exception):
    pass


@dataclass(eq=False)
class Request:
    windows: int
    parts: dict = field(default_factory=dict)       # window index -> (tokens, frames, logps)
    arrival: float = field(default_factory=time.perf_counter)
    future: Future = field(default_factory=Future)


@dataclass(eq=False)
class Job:
    """One window of a request."""
    samples: object
    seconds: float
    request: Request
    index: int
    offset: int       # where the window starts in the request, in encoder frames
    keep: tuple       # (from, to) seconds into the window whose tokens it keeps


class Engine:
    def __init__(self, asr, *, num_slots=32, steps=8, max_seconds=300.0, window_s=30.0, extra_s=2.0,
                 admit_max=16, admit_max_seconds=240.0, max_pending_s=120.0, cuda_graph=True):
        self.asr, self.steps, self.admit_max = asr, steps, admit_max
        self.max_seconds, self.window_s, self.extra_s = max_seconds, window_s, extra_s
        self.admit_max_seconds, self.max_pending_s = admit_max_seconds, max_pending_s
        self.config = dict(num_slots=num_slots, steps=steps, max_seconds=max_seconds, window_s=window_s,
                           extra_s=extra_s, admit_max=admit_max, admit_max_seconds=admit_max_seconds,
                           max_pending_s=max_pending_s, cuda_graph=cuda_graph and asr.device.type == "cuda")
        dev, b = asr.device, num_slots
        self.t_max = out_frames(num_frames(int(window_s * SR)))
        # Slot state, on the GPU, fixed size.
        self.enc = torch.zeros(b, self.t_max, 512, device=dev)   # encoder output of each slot
        self.length = torch.zeros(b, dtype=torch.int64, device=dev)  # its number of frames
        self.t = torch.zeros(b, dtype=torch.int64, device=dev)       # next frame to decode
        self.n = torch.zeros(b, dtype=torch.int64, device=dev)       # tokens out so far
        self.active = torch.zeros(b, dtype=torch.bool, device=dev)
        self.ctx = torch.tensor([START], device=dev).repeat(b, 1)    # last two tokens
        self.dec = asr.decode(self.ctx)                               # decoder output for ctx
        self.tokens = torch.zeros(b, self.t_max, dtype=torch.int64, device=dev)
        self.frames = torch.zeros(b, self.t_max, dtype=torch.int64, device=dev)
        self.logps = torch.zeros(b, self.t_max, device=dev)
        self.rows = torch.arange(b, device=dev)
        self.owner = [None] * b                                      # host side: the job in each slot
        self.pending = collections.deque()
        self.pending_s = 0.0
        self.stats = collections.Counter()
        self.lock = threading.Condition()
        self.graph = self._capture() if self.config["cuda_graph"] else None
        threading.Thread(target=self._run, name="gpu-worker", daemon=True).start()

    @torch.no_grad()
    def warm_up(self):
        """For g = 1..admit_max, prefill g windows of the longest length a group of g may have."""
        shapes = []
        for g in range(1, self.admit_max + 1):
            seconds = min(self.window_s, self.admit_max_seconds / g)
            wave = torch.randn(int(seconds * SR)).mul_(0.05).numpy()
            self.asr.prefill([wave] * g)
            shapes.append((g, round(seconds, 1)))
        if self.asr.device.type == "cuda":
            torch.cuda.synchronize()
        return shapes

    # -- requests and windows --------------------------------------------------------------------

    def windows(self, n):
        """(start, end) samples and the kept (from, to) seconds of each window of n samples."""
        win, hop = int(self.window_s * SR), int((self.window_s - 2 * self.extra_s) * SR)
        if n <= win:
            return [(0, n, (0.0, math.inf))]
        out, start = [], 0
        while True:
            end = min(start + win, n)
            last = end == n
            out.append((start, end, (0.0 if start == 0 else self.extra_s, math.inf if last else self.window_s - self.extra_s)))
            if last:
                return out
            start += hop

    def submit(self, samples, sample_rate=SR) -> Future:
        """Queue one request (float32 in [-1, 1]); the future gets (tokens, frames, logps)."""
        if sample_rate != SR:
            raise ValueError(f"sample rate must be {SR}, got {sample_rate}")
        seconds = len(samples) / SR
        if seconds > self.max_seconds:
            self.stats["too_long"] += 1
            raise TooLong(f"{seconds:.1f}s is over the {self.max_seconds:g}s limit")
        cuts = self.windows(len(samples))
        req = Request(windows=len(cuts))
        jobs = [Job(samples[s:e], (e - s) / SR, req, i, round(s / SR / FRAME_S), keep)
                for i, (s, e, keep) in enumerate(cuts)]
        with self.lock:
            if self.max_pending_s and self.pending_s + seconds > self.max_pending_s:
                self.stats["busy"] += 1
                raise Busy(f"{self.pending_s:.0f}s of audio already waiting")
            self.pending.extend(jobs)
            self.pending_s += sum(j.seconds for j in jobs)
            self.stats["accepted"] += 1
            self.stats["windows"] += len(jobs)
            self.lock.notify()
        return req.future

    def _finish(self, job, tokens, frames, logps):
        """Keep the window's tokens in its kept part; answer the request once every window is in."""
        req = job.request
        if req.future.done():  # another window failed
            return
        t = frames * FRAME_S
        keep = (t >= job.keep[0]) & (t < job.keep[1])
        req.parts[job.index] = (tokens[keep], frames[keep] + job.offset, logps[keep])
        if len(req.parts) == req.windows:
            parts = [req.parts[i] for i in range(req.windows)]
            req.future.set_result(tuple(np.concatenate([p[k] for p in parts]) for k in range(3)))
            self.stats["done"] += 1
            self.stats["latency_s"] += time.perf_counter() - req.arrival

    # -- the three parts of a tick ---------------------------------------------------------------

    def _take(self, free):
        """Up to min(free, admit_max) pending windows in arrival order, while the padded group stays
        within admit_max_seconds (it bounds the encoder's memory)."""
        group, top = [], 0.0
        while self.pending and len(group) < min(free, self.admit_max):
            j = self.pending[0]
            if group and max(top, j.seconds) * (len(group) + 1) > self.admit_max_seconds:
                break
            group.append(self.pending.popleft())
            top = max(top, j.seconds)
            self.pending_s -= j.seconds
        return group

    def _admit(self):
        free = [s for s, job in enumerate(self.owner) if job is None]
        with self.lock:
            group = self._take(len(free)) if free and self.pending else []
        if not group:
            return
        start = time.perf_counter()
        enc, lens = self.asr.prefill([j.samples for j in group])   # the one encoder run for the group
        slots = torch.tensor(free[:len(group)], device=self.asr.device)
        self.enc[slots, :enc.shape[1]] = enc
        self.length[slots] = lens
        self.t[slots] = 0
        self.n[slots] = 0
        self.ctx[slots] = torch.tensor(START, device=self.asr.device)
        self.dec[slots] = self.asr.decode(self.ctx[slots])
        self.active[slots] = True
        for s, j in zip(free, group):
            self.owner[s] = j
        print(f"size: {len(group)}", file=sys.stderr, flush=True)  # sherpa's log line, read by loadtest.py
        self.stats.update(admits=1, admitted=len(group))
        self.stats["admit_s"] += time.perf_counter() - start

    def _step(self):
        """One greedy step for every slot."""
        a, rows = self.asr, self.rows
        logit = a.join(self.enc[rows, self.t.clamp(max=self.t_max - 1)], self.dec)
        y = logit.argmax(dim=-1)
        live = self.active & (self.t < self.length)
        emit = live & (y != 0) & (y != a.unk)
        pos = self.n.clamp(max=self.t_max - 1)
        logp = logit.gather(1, y[:, None]).squeeze(1) - logit.logsumexp(dim=-1)
        self.tokens[rows, pos] = torch.where(emit, y, self.tokens[rows, pos])
        self.frames[rows, pos] = torch.where(emit, self.t, self.frames[rows, pos])
        self.logps[rows, pos] = torch.where(emit, logp, self.logps[rows, pos])
        self.n.add_(emit)
        self.ctx.copy_(torch.where(emit[:, None], torch.stack([self.ctx[:, 1], y], dim=1), self.ctx))
        self.dec.copy_(torch.where(emit[:, None], a.decode(self.ctx), self.dec))
        self.t.add_(live)

    def _steps(self):
        for _ in range(self.steps):
            self._step()

    @torch.no_grad()
    def _capture(self):
        """Record `steps` greedy steps as one CUDA graph. Every slot is still empty, so the warm-up
        runs that CUDA graphs need change nothing."""
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._steps()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._steps()
        return graph

    def _collect(self):
        done = (self.active & (self.t >= self.length)).nonzero().squeeze(1)
        done_h = done.tolist()  # the tick's one wait for the GPU
        if not done_h:
            return
        n, tokens, frames, logps = (x[done].cpu().numpy() for x in (self.n, self.tokens, self.frames, self.logps))
        self.active[done] = False
        for i, s in enumerate(done_h):
            job, self.owner[s] = self.owner[s], None
            self._finish(job, tokens[i, :n[i]], frames[i, :n[i]], logps[i, :n[i]])

    @torch.no_grad()
    def _run(self):
        while True:
            with self.lock:
                while not self.pending and all(j is None for j in self.owner):
                    self.lock.wait()
            try:
                self._admit()
                start = time.perf_counter()
                if self.graph:
                    self.graph.replay()
                else:
                    self._steps()
                self._collect()
                self.stats.update(ticks=1, steps=self.steps)
                self.stats["step_s"] += time.perf_counter() - start
            except Exception as e:  # noqa: BLE001 - fail what is in the slots, keep serving
                for s, job in enumerate(self.owner):
                    if job is not None:
                        if not job.request.future.done():
                            job.request.future.set_exception(e)
                        self.owner[s] = None
                self.active.zero_()

    def health(self):
        with self.lock:
            return dict(slots_busy=sum(j is not None for j in self.owner), pending_windows=len(self.pending),
                        pending_s=self.pending_s, **self.stats, config=self.config)
