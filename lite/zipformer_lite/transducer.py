"""The Zipformer transducer's pieces, on the GPU. Same model files as the baseline.

- prefill: fbank in torch for the whole group (fbank.py), then the encoder ONNX through
  onnxruntime's CUDA provider, with inputs and outputs bound to torch tensors on the GPU. All of it
  runs on the transducer's own CUDA stream, so it can overlap the greedy steps running on another;
  prefill returns a CUDA event the consumer waits on. Nothing goes through the host.
- Memory release. onnxruntime keeps the GPU memory of earlier runs in its arena, and runs of many
  different shapes can leave it holding regions no later run fits in, so it asks CUDA for more.
  When the GPU's used memory is over release_gb, the next run hands every free region of the arena
  back to CUDA when it ends: memory drops back to what the model holds, then grows only as far as
  the following runs need. 0 never releases.
- decode, join: the decoder and joiner ONNX re-written in torch, weights read from their files.
- result: tokens back to sherpa's JSON fields.

The greedy loop that uses them is in engine.py.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .fbank import Fbank

FRAME_S = 0.04  # one encoder frame
START = (-1, 0)  # sherpa's decoder context before the first token: -1 is "none", 0 is blank


def out_frames(feature_frames):
    """Encoder output length (checked against encoder_out_lens)."""
    return ((feature_frames - 7) // 2 + 1) // 2


def onnx_weights(path, device):
    import onnx
    from onnx import numpy_helper
    return {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()).to(device)
            for t in onnx.load(path).graph.initializer}


class Transducer:
    def __init__(self, paths, device="cuda", gpu_mem_gb=None, num_threads=2, release_gb=0):
        import onnxruntime as ort
        self.device = torch.device(device)
        self.fbank = Fbank(self.device)
        self.dec_w = onnx_weights(paths["decoder"], self.device)
        self.join_w = onnx_weights(paths["joiner"], self.device)
        self.symbols = {}
        for line in open(paths["tokens"], encoding="utf-8"):
            sym, i = line.rstrip("\n").rsplit(" ", 1)
            self.symbols[int(i)] = sym.replace("▁", " ")
        self.unk = next(i for i, s in self.symbols.items() if s == "<unk>")

        # The prefill stream: fbank and the encoder both run on it.
        self.stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = num_threads
        so.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        if self.device.type == "cuda":
            ort.preload_dlls()  # CUDA and cuDNN from the nvidia-* wheels torch installed
            cuda = {"device_id": self.device.index or 0, "cudnn_conv_algo_search": "HEURISTIC",
                    "arena_extend_strategy": "kSameAsRequested", "has_user_compute_stream": "1",
                    "user_compute_stream": str(self.stream.cuda_stream)}
            if gpu_mem_gb:
                cuda["gpu_mem_limit"] = str(int(gpu_mem_gb * 2**30))
            providers.insert(0, ("CUDAExecutionProvider", cuda))
        self.encoder = ort.InferenceSession(paths["encoder"], so, providers=providers)
        self.run_options, self.release_options = ort.RunOptions(), ort.RunOptions()
        if self.device.type == "cuda":
            self.release_options.add_run_config_entry("memory.enable_memory_arena_shrinkage",
                                                      f"gpu:{self.device.index or 0}")
        self.release_bytes = release_gb * 2**30 if self.device.type == "cuda" else 0
        self.releases = 0
        if self.device.type == "cuda" and "CUDAExecutionProvider" not in self.encoder.get_providers():
            raise RuntimeError(f"onnxruntime has no CUDA provider: {self.encoder.get_providers()}")

    @torch.no_grad()
    def prefill(self, waves):
        """Float32 numpy arrays at 16 kHz in [-1, 1] -> encoder output (G, T', 512) padded to the
        longest, the lengths (G,), on self.device, and a CUDA event recorded when they are ready
        (None on the CPU). A consumer on another stream waits on the event and calls record_stream
        on the tensors before using them."""
        if self.device.type != "cuda":
            x, x_lens = self.fbank([torch.from_numpy(w) for w in waves])
            out, lens = self.encoder.run(None, {"x": x.numpy(), "x_lens": x_lens.numpy()})
            return torch.from_numpy(out), torch.from_numpy(lens), None
        with torch.cuda.stream(self.stream):
            x, x_lens = self.fbank([torch.from_numpy(w) for w in waves])
            g, t, _ = x.shape
            out = torch.empty(g, out_frames(t), 512, device=self.device)
            lens = torch.empty(g, dtype=torch.int64, device=self.device)
            io, d = self.encoder.io_binding(), self.device.index or 0
            io.bind_input("x", "cuda", d, np.float32, list(x.shape), x.contiguous().data_ptr())
            io.bind_input("x_lens", "cuda", d, np.int64, [g], x_lens.contiguous().data_ptr())
            io.bind_output("encoder_out", "cuda", d, np.float32, list(out.shape), out.data_ptr())
            io.bind_output("encoder_out_lens", "cuda", d, np.int64, [g], lens.data_ptr())
            options = self.run_options
            if self.release_bytes:
                free, total = torch.cuda.mem_get_info(self.device)   # the whole GPU, as nvidia-smi shows it
                if total - free > self.release_bytes:
                    options = self.release_options
                    self.releases += 1
            self.encoder.run_with_iobinding(io, options)
            ready = torch.cuda.Event()
            ready.record(self.stream)
        return out, lens, ready

    def decode(self, ctx):
        """Last two tokens (B, 2), -1 = none -> (B, 512). The ONNX decoder, in torch."""
        e = self.dec_w["decoder.embedding.weight"][ctx.clamp_min(0)] * (ctx >= 0).unsqueeze(-1)
        h = F.conv1d(e.transpose(1, 2), self.dec_w["decoder.conv.weight"], groups=128).squeeze(-1).relu()
        return F.linear(h, self.dec_w["decoder_proj.weight"], self.dec_w["decoder_proj.bias"])

    def join(self, enc, dec):
        return F.linear(torch.tanh(enc + dec), self.join_w["output_linear.weight"], self.join_w["output_linear.bias"])

    def result(self, tokens, frames, logps):
        """One utterance's tokens, their frame indexes and log-probabilities -> sherpa's JSON fields."""
        syms = [self.symbols[int(k)] for k in tokens]
        return dict(lang="", emotion="", event="", text="".join(syms), timestamps=[round(float(f) * FRAME_S, 2) for f in frames],
                    durations=[], tokens=syms, ys_log_probs=[round(float(p), 6) for p in logps], words=[])
