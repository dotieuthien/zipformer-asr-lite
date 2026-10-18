"""Download a lite run from the GPU host (lite/bench.py over the vkong tunnel) into lite/results/<run>/.

    python3 experiments/lite/fetch.py http://127.0.0.1:PORT --run <folder> [--servers lite-b,...]

Fetches, for every server and scenario that has a summary.json: the summary, every c<N>.jsonl it
lists, the scenario log, and the server's own output; plus /health and gpu-preflight.json.
Standard library only.
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SERVERS = ["lite-b"]
SCENARIOS = ["burst-same", "burst-mixed", "sustain-same", "sustain-mixed", "burst-long", "long-utts"]


def get(url, tries=8):
    """The vkong tunnel drops connections now and then: retry."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
        except OSError as e:
            if i == tries - 1:
                raise
            print(f"retry {url}: {e}")
            time.sleep(5)


def save(base, path, dest, keep=False):
    if keep and (dest / path).exists():  # a finished level's rows never change
        return (dest / path).read_bytes()
    data = get(f"{base}/results/{path}")
    if data is not None:
        (dest / path).parent.mkdir(parents=True, exist_ok=True)
        (dest / path).write_bytes(data)
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("--run", required=True)
    ap.add_argument("--servers", default=",".join(SERVERS), help="comma-separated server names")
    a = ap.parse_args()
    dest = REPO / "lite" / "results" / a.run
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "health.json").write_bytes(get(f"{a.base}/health"))
    save(a.base, "gpu-preflight.json", dest)
    for server in a.servers.split(","):
        for name in ("server.out", "log.txt"):
            save(a.base, f"{server}/{name}", dest)
        for tag in SCENARIOS:
            summary = save(a.base, f"{server}/{tag}/summary.json", dest)
            if summary is None:
                continue
            save(a.base, f"{server}/{tag}.log", dest)
            levels = [lv["concurrency"] for lv in json.loads(summary)["levels"]]
            for c in levels:
                save(a.base, f"{server}/{tag}/c{c}.jsonl", dest, keep=True)
            print(f"{server}/{tag}: {len(levels)} levels")


if __name__ == "__main__":
    main()
