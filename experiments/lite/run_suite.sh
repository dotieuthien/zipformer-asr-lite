#!/usr/bin/env bash
# Drive the lite server through the four phase 1 scenarios over the vkong tunnel. The load client runs
# on the GPU host (POST /benchmark); this script only starts things, waits and prints summary lines.
#
#   experiments/lite/run_suite.sh http://127.0.0.1:PORT [name] [extra /server query]
#   experiments/lite/run_suite.sh @/tmp/lite-url ...   # read the URL from a file kept by keep_tunnel.sh
#   experiments/lite/run_suite.sh $URL lite-b "max_pending_seconds=0"        # no refusals, as sherpa
#   SCENARIOS="sustain-mixed" experiments/lite/run_suite.sh $URL ...         # only these
#   SCENARIOS="burst-long" ...       # 1-5 min files (lite cuts them into 30 s windows)
#
# The levels go past phase 1's; a sweep stops by itself once P90 is over 5 s.
set -uo pipefail
U=$1 NAME=${2:-lite} EXTRA=${3:-}
BURST=1,2,4,8,16,24,32,48,64,96,128,192,256
SUSTAIN=32,64,128,192,256,320,384,448,512,640,768,1024,1280
SUSTAIN_MIXED=32,64,80,96,112,128,160,192,256,320,384,512

log() { echo "$(date +%H:%M:%S) $*"; }
base() { if [[ $U == @* ]]; then cat "${U#@}"; else echo "$U"; fi; }
api() {  # path, then curl options; retries while the tunnel is down
  local path=$1; shift
  for _ in $(seq 1 120); do out=$(curl -s -m 30 "$@" "$(base)$path") && [ -n "$out" ] && { echo "$out"; return; }; sleep 5; done
}

if ! api /server | grep -q "\"name\":\"$NAME\"" || ! api /server | grep -q '"running":true'; then
  log "START server $NAME $EXTRA"
  api "/server?name=$NAME&$EXTRA" -m 900 -X POST | cut -c1-300; echo
fi

run() {  # tag, query
  log "BENCH $NAME/$1: $2"
  api "/benchmark?tag=$1&$2" -X POST > /dev/null
  until api /benchmark | grep -q '"running":false'; do sleep 10; done
  api "/results/$NAME/$1.log" | grep -E "^(reference|burst|sustain|done|stop|errors)" | cut -c1-320
}

query() {  # bash 3.2 (macOS): no associative arrays
  case $1 in
    burst-same)    echo "mode=burst&concurrency=$BURST&audio=same&repeats=5" ;;
    burst-mixed)   echo "mode=burst&concurrency=$BURST&audio=mixed&repeats=5" ;;
    sustain-same)  echo "mode=sustain&concurrency=$SUSTAIN&audio=same&duration=60&warmup=10" ;;
    sustain-mixed) echo "mode=sustain&concurrency=$SUSTAIN_MIXED&audio=mixed&duration=60&warmup=10" ;;
    burst-long)    echo "mode=burst&concurrency=1,2,4,8,16,32&audio=long&repeats=2" ;;
    long-utts)     echo "mode=burst&concurrency=1&audio=long-utts&repeats=1" ;;   # its reference pass: each utterance alone
  esac
}
for tag in ${SCENARIOS:-burst-same burst-mixed sustain-same sustain-mixed}; do
  run "$tag" "$(query "$tag")"
done
log "DONE $NAME"
