#!/usr/bin/env bash
# The upstream sherpa-onnx C++ offline websocket server, unmodified: the release binary
# (model.CPP_URL, downloaded by `python model.py`) with CUDA 12 / cuDNN 9 from the nvidia-* wheels.
# Takes the same flags as server_python.py, e.g.
#   bash server_cpp.sh --provider=cuda --port=6006 --num-work-threads=3 --max-batch-size=5 \
#     --encoder=... --decoder=... --joiner=... --tokens=...
set -euo pipefail
cd "$(dirname "$0")"
bin=$(python -c 'import model; print(model.CPP_SERVER)')
cuda=$(python -c 'import model; print(":".join(model.cuda_lib_dirs()))')
export LD_LIBRARY_PATH="$(dirname "$(dirname "$bin")")/lib:${cuda}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$bin" "$@"
