"""Check that lite computes what sherpa-onnx computes.

1. fbank: lite's torch fbank against kaldi-native-fbank (sherpa's library, sherpa's options).
2. transcripts: lite's engine (slots, one prefill per group, step-by-step greedy) against
   sherpa_onnx's offline recognizer on LibriSpeech utterances, all submitted at once so groups,
   padding, masking and slot reuse are all exercised; and the WER of both.
3. long audio: lite cuts files of 1-5 min into 30 s windows and merges them (icefall's way);
   sherpa decodes each file in one piece. Compare their WER.

    pip install kaldi-native-fbank sherpa-onnx    # only for this check
    python check.py                  # on the GPU
    python check.py --device cpu     # anywhere
    python check.py --long 12        # also 12 long files
"""
import argparse

import numpy as np
import torch

import audio
import model
from zipformer_lite import fbank
from zipformer_lite.engine import Engine
from zipformer_lite.transducer import Transducer


def check_fbank(device, seconds=(0.3, 1.0, 7.43)):
    import kaldi_native_fbank as knf

    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.mel_opts.num_bins = fbank.N_MELS
    opts.mel_opts.high_freq = -400
    rng = np.random.default_rng(0)
    waves = [(rng.standard_normal(int(s * fbank.SR)) * 0.1).astype(np.float32) for s in seconds]
    ours, lens = fbank.Fbank(device)([torch.from_numpy(w) for w in waves])
    worst = 0.0
    for i, w in enumerate(waves):
        ref = knf.OnlineFbank(opts)
        ref.accept_waveform(fbank.SR, w)
        ref.input_finished()
        ref = np.stack([ref.get_frame(j) for j in range(ref.num_frames_ready)])
        assert len(ref) == lens[i], (len(ref), int(lens[i]))
        worst = max(worst, float(np.abs(ours[i, : lens[i]].cpu().numpy() - ref).max()))
    return worst


def check_transcripts(device, per_set=40, slots=8, long=0):
    import sherpa_onnx

    paths = model.ensure()
    asr = Transducer(paths, device=device)
    engine = Engine(asr, num_slots=slots, max_pending_s=0)
    ref = sherpa_onnx.OfflineRecognizer.from_transducer(**paths, num_threads=4, provider="cpu")
    for name, n in (("same", per_set), ("mixed", per_set), ("long", long)):
        if not n:
            continue
        utts = audio.load(name)[:n]
        futures = [engine.submit(u["samples"]) for u in utts]
        ours = [asr.result(*f.result())["text"].strip() for f in futures]
        theirs = []
        for u in utts:
            s = ref.create_stream()
            s.accept_waveform(fbank.SR, u["samples"])
            ref.decode_stream(s)
            theirs.append(s.result.text.strip())
        same = sum(a == b for a, b in zip(ours, theirs))
        wer = lambda hyps: sum(audio.word_errors(u["text"], h)[0] for u, h in zip(utts, hyps)) / sum(  # noqa: E731
            audio.word_errors(u["text"], h)[1] for u, h in zip(utts, hyps))
        print(f"{name}: {same}/{len(utts)} transcripts identical to sherpa; WER lite {wer(ours):.2%}, "
              f"sherpa {wer(theirs):.2%}")
        for u, a, b in zip(utts, ours, theirs):
            if a != b:
                e_a, e_b = audio.word_errors(u["text"], a)[0], audio.word_errors(u["text"], b)[0]
                print(f"  {u['id']} ({u['seconds']:.0f} s): word errors lite {e_a}, sherpa {e_b}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--per-set", type=int, default=40, help="utterances from the same and mixed sets")
    ap.add_argument("--long", type=int, default=0, help="files from the long set")
    a = ap.parse_args()
    print(f"fbank: max abs difference to kaldi-native-fbank {check_fbank(a.device):.2e}")
    check_transcripts(a.device, a.per_set, long=a.long)


if __name__ == "__main__":
    main()
