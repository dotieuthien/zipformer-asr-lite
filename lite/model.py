"""The model: the original icefall Zipformer (LibriSpeech, medium) as exported to ONNX by the
sherpa-onnx authors, the same files the baseline serves. Downloaded once to CACHE, outside the
project folder, so `vkong attach --update` neither resyncs nor deletes it.
"""
import os
import tarfile
import urllib.request
from pathlib import Path

CACHE = Path(os.getenv("CACHE_DIR", Path.home() / ".cache" / "zipformer-asr-lite"))
NAME = "sherpa-onnx-zipformer-en-2023-06-26"
URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{NAME}.tar.bz2"
DIR = CACHE / NAME
FILES = dict(encoder="encoder-epoch-99-avg-1.onnx", decoder="decoder-epoch-99-avg-1.onnx",
             joiner="joiner-epoch-99-avg-1.onnx", tokens="tokens.txt")
SR = 16_000


def ensure():
    """Download and unpack the model if it is not there yet; return {name: path}."""
    if not (DIR / FILES["encoder"]).exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        part = CACHE / f"{NAME}.tar.bz2.part"
        print(f"downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, part)
        with tarfile.open(part) as tar:
            tar.extractall(CACHE, filter="data")
        part.unlink()
    return {k: str(DIR / v) for k, v in FILES.items()}


if __name__ == "__main__":
    print(ensure())
