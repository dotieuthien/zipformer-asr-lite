#!/usr/bin/env bash
# Keep a vkong tunnel to the GPU host open and write its local URL to a file. The tunnel drops
# now and then; run_suite.sh reads the file on every call, so it carries on after a reconnect.
#
#   experiments/lite/keep_tunnel.sh vk_xxxx /tmp/lite-url &
#   experiments/lite/run_suite.sh @/tmp/lite-url lite ...
set -u
ID=$1 FILE=$2 DIR=${3:-lite}
while true; do
  vkong attach "$ID" -C "$DIR" 2>&1 | while IFS= read -r line; do
    if [[ $line =~ (http://127\.0\.0\.1:[0-9]+) ]]; then
      echo "${BASH_REMATCH[1]}" > "$FILE"
      echo "$(date +%H:%M:%S) tunnel ${BASH_REMATCH[1]}"
    fi
  done
  echo "$(date +%H:%M:%S) tunnel lost, reconnecting"
  sleep 5
done
