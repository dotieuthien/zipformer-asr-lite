"""Downloads: the model, the upstream C++ server, and where the CUDA libraries are.

The model is the original Zipformer from icefall (egs/librispeech/ASR/zipformer, the medium model of
the paper, checkpoint Zengwei/icefall-asr-librispeech-zipformer-2023-05-15, trained on LibriSpeech),
as exported to ONNX by the sherpa-onnx authors; fp32.

Everything is cached under CACHE, outside the project folder, so `vkong attach --update` neither
resyncs nor deletes it.
"""
import os
import platform
import tarfile
import urllib.request
from pathlib import Path

CACHE = Path(os.getenv("CACHE_DIR", Path.home() / ".cache" / "zipformer-asr-lite"))
NAME = "sherpa-onnx-zipformer-en-2023-06-26"
URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{NAME}.tar.bz2"
DIR = Path(os.getenv("MODEL_DIR", CACHE / NAME))
FILES = dict(encoder="encoder-epoch-99-avg-1.onnx", decoder="decoder-epoch-99-avg-1.onnx",
             joiner="joiner-epoch-99-avg-1.onnx", tokens="tokens.txt")
# The upstream C++ binaries as released (built by sherpa-onnx CI): CUDA 12 + cuDNN 9 on Linux,
# the CPU build on macOS for local tests.
CPP_NAME = os.getenv("CPP_NAME", "sherpa-onnx-v1.13.8-osx-universal2-shared" if platform.system() == "Darwin"
                     else "sherpa-onnx-v1.13.8-cuda-12.x-cudnn-9.x-onnxruntime1.28.2-linux-x64-gpu")
CPP_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.8/{CPP_NAME}.tar.bz2"
CPP_DIR = CACHE / CPP_NAME
CPP_SERVER = CPP_DIR / "bin" / "sherpa-onnx-offline-websocket-server"
SR = 16_000


def fetch(url, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}", flush=True)
    part = dest_dir / (url.rsplit("/", 1)[1] + ".part")
    urllib.request.urlretrieve(url, part)
    with tarfile.open(part) as tar:
        tar.extractall(dest_dir, filter="data")
    part.unlink()


def ensure():
    """Download and unpack the model if it is not there yet; return {name: path}."""
    if not (DIR / FILES["encoder"]).exists():
        fetch(URL, DIR.parent)
    return {k: str(DIR / v) for k, v in FILES.items()}


def ensure_cpp():
    if not CPP_SERVER.exists():
        fetch(CPP_URL, CPP_DIR.parent)
    return CPP_SERVER


def cuda_lib_dirs():
    """CUDA 12 / cuDNN 9 from the nvidia-* wheels in requirements.txt. sherpa-onnx loads them by
    soname, so they must be on LD_LIBRARY_PATH when the process starts."""
    try:
        import nvidia
    except ImportError:
        return []
    return sorted({str(p) for root in nvidia.__path__ for p in Path(root).glob("*/lib")})


if __name__ == "__main__":
    print(ensure())
    print(ensure_cpp())
    print(sorted(p.name for p in (CPP_DIR / "lib").iterdir()))
