"""Kaldi fbank in torch, for a whole batch at once on the GPU.

The same features sherpa-onnx computes on the CPU with kaldi-native-fbank, with its options:
16 kHz, 25 ms window every 10 ms (centred, edges reflected), DC removal, pre-emphasis 0.97, Povey
window, FFT of 512, power spectrum, 80 mel bins from 20 Hz to 7600 Hz, log. check.py compares
the two.
"""
import math

import torch

SR, SHIFT, LENGTH, N_FFT, N_MELS = 16_000, 160, 400, 512, 80
LOW_HZ, HIGH_HZ, PREEMPH = 20.0, SR / 2 - 400.0, 0.97
PAD = math.log(1e-10)  # sherpa pads a batch of features with this


def num_frames(n_samples):
    return (n_samples + SHIFT // 2) // SHIFT


def mel_banks():
    """(256, 80): Kaldi's triangles in mel space, over FFT bins 0..255."""
    mel = lambda f: 1127.0 * math.log(1.0 + f / 700.0)  # noqa: E731
    lo, hi = mel(LOW_HZ), mel(HIGH_HZ)
    step = (hi - lo) / (N_MELS + 1)
    banks = torch.zeros(N_FFT // 2, N_MELS, dtype=torch.float64)
    for m in range(N_MELS):
        left, centre, right = lo + m * step, lo + (m + 1) * step, lo + (m + 2) * step
        for i in range(N_FFT // 2):
            x = mel(i * SR / N_FFT)
            if left < x < right:
                banks[i, m] = (x - left) / (centre - left) if x <= centre else (right - x) / (right - centre)
    return banks.float()


class Fbank:
    def __init__(self, device):
        self.device = torch.device(device)
        n = torch.arange(LENGTH, dtype=torch.float64)
        self.window = (0.5 - 0.5 * torch.cos(2 * math.pi * n / (LENGTH - 1))).pow(0.85).float().to(self.device)
        self.banks = mel_banks().to(self.device)
        self.offsets = torch.arange(LENGTH, device=self.device)

    def frame_index(self, n_samples, base):
        """Sample index of every frame of one utterance, reflected at both ends, as Kaldi does."""
        start = torch.arange(num_frames(n_samples), device=self.device) * SHIFT + SHIFT // 2 - LENGTH // 2
        s = start[:, None] + self.offsets
        s = torch.where(s < 0, -s - 1, s)
        s = torch.where(s >= n_samples, 2 * n_samples - 1 - s, s)
        return s + base

    @torch.no_grad()
    def __call__(self, waves):
        """waves: list of 1-D float tensors in [-1, 1] -> features (B, T_max, 80) padded with
        log(1e-10), and lengths (B,) int64."""
        lengths = [num_frames(len(w)) for w in waves]
        bases = [0]
        for w in waves[:-1]:
            bases.append(bases[-1] + len(w))
        flat = torch.cat([w.to(self.device, torch.float32) for w in waves])
        x = flat[torch.cat([self.frame_index(len(w), b) for w, b in zip(waves, bases)])]  # (frames, 400)
        x = x - x.mean(dim=1, keepdim=True)
        x = torch.cat([x[:, :1] * (1 - PREEMPH), x[:, 1:] - PREEMPH * x[:, :-1]], dim=1) * self.window
        spec = torch.fft.rfft(x, n=N_FFT)
        power = (spec.real.square() + spec.imag.square())[:, : N_FFT // 2]
        feats = (power @ self.banks).clamp_min(torch.finfo(torch.float32).eps).log()
        out = torch.full((len(waves), max(lengths), N_MELS), PAD, device=self.device)
        for i, part in enumerate(feats.split(lengths)):
            out[i, : lengths[i]] = part
        return out, torch.tensor(lengths, device=self.device)
