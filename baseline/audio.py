"""Test audio: LibriSpeech test-clean, real English read speech with reference transcripts, the test
set the model was evaluated on. Downloaded once from the HuggingFace copy of openslr/librispeech_asr
(one 350 MB parquet file); each set is decoded once and cached as .npz.

Sets:
  same    every utterance of 9.75-10.25 s (55 of them): one length, different speakers and words
  mixed   300 utterances drawn at random (seed 0) from all 2620: the natural spread of lengths

    python audio.py                      # download and build both sets
    python audio.py wavs mixed <folder>  # write a set as WAV files to listen to
"""
import io
import json
import random
import re
import sys
import urllib.request
import wave
from pathlib import Path

import numpy as np

from model import CACHE, SR

URL = "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main/clean/test/0000.parquet"
DIR = CACHE / "librispeech"
PARQUET = DIR / "test-clean.parquet"
SETS = {"same": dict(min_s=9.75, max_s=10.25), "mixed": dict(sample=300, seed=0)}


def build(name):
    """Decode the utterances of one set from the parquet file into DIR/<name>.npz."""
    import pyarrow.parquet as pq
    import soundfile as sf

    if not PARQUET.exists():
        DIR.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, PARQUET.with_suffix(".part"))
        PARQUET.with_suffix(".part").rename(PARQUET)
    rows = pq.read_table(PARQUET, columns=["id", "text", "audio"]).to_pylist()
    rows.sort(key=lambda r: r["id"])
    utts = []
    for r in rows:
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        assert sr == SR and x.ndim == 1, (r["id"], sr, x.shape)
        utts.append((r["id"], r["text"], x))
    spec = SETS[name]
    if "sample" in spec:
        utts = random.Random(spec["seed"]).sample(utts, spec["sample"])
    else:
        utts = [u for u in utts if spec["min_s"] <= len(u[2]) / SR <= spec["max_s"]]
    lengths = np.array([len(x) for _, _, x in utts])
    np.savez(DIR / f"{name}.npz", samples=np.concatenate([x for _, _, x in utts]), lengths=lengths,
             meta=json.dumps([dict(id=i, text=t) for i, t, _ in utts]))
    print(f"{name}: {len(utts)} utterances, {lengths.min() / SR:.2f}-{lengths.max() / SR:.2f} s, "
          f"mean {lengths.mean() / SR:.2f} s, {lengths.sum() / SR / 60:.1f} min", flush=True)


def load(name):
    """[{id, text, seconds, samples}] of one set, in a fixed order."""
    path = DIR / f"{name}.npz"
    if not path.exists():
        build(name)
    with np.load(path) as z:  # each z[...] reads the array again: read each once
        samples, lengths, meta = z["samples"], z["lengths"], json.loads(str(z["meta"]))
    out, start = [], 0
    for m, n in zip(meta, lengths):
        out.append(dict(m, seconds=float(n / SR), samples=samples[start:start + n]))
        start += n
    return out


def words(text):
    return re.sub(r"[^A-Z' ]", " ", text.upper()).split()


def word_errors(ref, hyp):
    """(edit distance in words, words in ref): WER = sum of the first / sum of the second."""
    a, b = words(ref), words(hyp)
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1], len(a)


def save_wavs(name, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for u in load(name):
        with wave.open(str(folder / f"{u['seconds']:05.2f}s-{u['id']}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.clip(u["samples"], -1, 1) * 32767).astype("<i2").tobytes())
    (folder / "transcripts.txt").write_text("".join(f"{u['id']} {u['seconds']:.2f}s {u['text']}\n" for u in load(name)))
    return folder


if __name__ == "__main__":
    if sys.argv[1:2] == ["wavs"]:
        print(save_wavs(sys.argv[2], sys.argv[3]))
    else:
        for name in SETS:
            build(name)
