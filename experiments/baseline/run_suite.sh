#!/usr/bin/env bash
# Drive one server through the four baseline scenarios over the vkong tunnel. The load client runs on
# the GPU host (POST /benchmark); this script only starts things, waits and prints the summary lines.
#
#   experiments/baseline/run_suite.sh http://127.0.0.1:PORT cpp [max_batch_size num_work_threads num_threads]
#   SCENARIOS="sustain-same sustain-mixed" experiments/baseline/run_suite.sh ...   # only these
#
# If the tunnel drops, the run goes on on the GPU host: reattach and rerun with the scenarios left.
# A POST while a run is going gets 409, and the script then waits for that run and prints it.
set -uo pipefail
U=$1 IMPL=$2 B=${3:-5} W=${4:-3} T=${5:-2}
NAME=$IMPL-b$B-w$W-t$T
BURST=1,2,4,8,16,24,32,48,64,96,128
SUSTAIN=32,64,128,192,256,320,384,448,512,640,768

log() { echo "$(date +%H:%M:%S) $*"; }
get() { for _ in $(seq 1 30); do out=$(curl -s -m 15 "$@") && [ -n "$out" ] && { echo "$out"; return; }; sleep 5; done; }

if ! get "$U/server" | grep -q "\"name\":\"$NAME\".*" || ! get "$U/server" | grep -q '"running":true'; then
  log "START server $NAME"
  get -X POST "$U/server?impl=$IMPL&max_batch_size=$B&num_work_threads=$W&num_threads=$T" | cut -c1-200; echo
fi

run() {  # tag, query
  log "BENCH $NAME/$1: $2"
  get -X POST "$U/benchmark?tag=$1&$2" > /dev/null
  until get "$U/benchmark" | grep -q '"running":false'; do sleep 10; done
  get "$U/results/$NAME/$1.log" | grep -E "^(reference|burst|sustain|done|stop|errors)" | cut -c1-300
}

query() {  # bash 3.2 (macOS): no associative arrays
  case $1 in
    burst-same)    echo "mode=burst&concurrency=$BURST&audio=same&repeats=5" ;;
    burst-mixed)   echo "mode=burst&concurrency=$BURST&audio=mixed&repeats=5" ;;
    sustain-same)  echo "mode=sustain&concurrency=$SUSTAIN&audio=same&duration=60&warmup=10" ;;
    sustain-mixed) echo "mode=sustain&concurrency=$SUSTAIN&audio=mixed&duration=60&warmup=10" ;;
  esac
}
for tag in ${SCENARIOS:-burst-same burst-mixed sustain-same sustain-mixed}; do
  run "$tag" "$(query "$tag")"
done
log "DONE $NAME"
