"""Recompute the tables in PHASE1.md from the raw per-request files.

    python3 experiments/baseline/analyze.py                  # the committed run
    python3 experiments/baseline/analyze.py --run <folder>   # your own rerun

Standard library only, no GPU. Input: baseline/results/<run>/<server>/<scenario>/c<N>.jsonl, one row
per request as loadtest.py wrote it, and summary.json next to it.

Definitions, per request:
  latency  burst: first byte sent -> transcript received.
           sustain: end of the utterance -> transcript received (queueing included).
  RTF      latency / audio length.
Percentiles use linear interpolation at (n-1)p over every counted request of a level, as in vieneu.
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = "gpu-20261017"
SERVERS = [("cpp-b5-w3-t2", "C++ gốc"), ("python-b5-w3-t2", "Python port")]
SCENARIOS = [("burst-same", "Gửi cùng lúc, câu ~10 s"), ("burst-mixed", "Gửi cùng lúc, câu 1.4–33 s"),
             ("sustain-same", "Nói liên tục, câu ~10 s"), ("sustain-mixed", "Nói liên tục, câu 1.4–33 s")]
BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 40)]  # seconds
SLAS = (0.5, 1.0, 2.0)


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latency(r):
    return r["latency_s"] if r["mode"] == "burst" else r["delay_s"]


def sweep(folder):
    """(summary, {users: stats}) for one server and scenario, or None."""
    if not (folder / "summary.json").exists():
        return None
    summary = json.loads((folder / "summary.json").read_text())
    levels = {}
    for lv in summary["levels"]:
        c = lv["concurrency"]
        rows = [r for r in read(folder / f"c{c}.jsonl") if not r.get("warm")]
        ok = [r for r in rows if r["ok"]]
        lat = [latency(r) for r in ok]
        rtf = [latency(r) / r["length_s"] for r in ok]
        levels[c] = dict(n=len(rows), errors=len(rows) - len(ok), lat50=q(lat, .5), lat90=q(lat, .9),
                         rtf50=q(rtf, .5), rtf90=q(rtf, .9), audio_s_per_s=lv["audio_s_per_s"],
                         wer=sum(r["word_errors"] for r in ok) / max(1, sum(r["words"] for r in ok)),
                         batch=lv["batch_size_mean"], gpu=lv["gpu_util_mean"], server_cpu=lv["server_cpu_cores_mean"],
                         by_length={b: q([latency(r) for r in ok if b[0] <= r["length_s"] < b[1]], .9) for b in BINS})
    return summary, levels


def capacity(levels, sla):
    return max((c for c, v in levels.items() if not v["errors"] and v["lat90"] is not None and v["lat90"] <= sla),
               default=None)


def f(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def pair(v, a, b, nd=2, unit=""):
    return "—" if v is None else f"{f(v[a], nd)} / {f(v[b], nd)}{unit}"


def speed(v):
    return "—" if v is None else f"{v['audio_s_per_s']:.0f} s/giây"


def alone(root):
    """Each utterance recognised alone, one at a time: the reference pass before every scenario."""
    print("## Một request mỗi lúc, theo độ dài câu\n")
    print("| Độ dài câu | Số câu | Latency P50 / P90, C++ gốc | Latency P50 / P90, Python | RTF P50, C++ gốc "
          "| RTF P50, Python |")
    print("|---|---:|---:|---:|---:|---:|")
    cells = {}
    for s, _ in SERVERS:
        folder = root / s / "burst-mixed"
        summary = json.loads((folder / "summary.json").read_text())
        lengths = {r["utt"]: r["length_s"] for path in folder.glob("c*.jsonl") for r in read(path)}
        pairs = [(lengths[u], t) for u, t in zip(summary["reference"], summary["alone_latency_s"]) if u in lengths]
        for b in BINS:
            xs = [(n, t) for n, t in pairs if b[0] <= n < b[1]]
            cells[(s, b)] = (len(xs), q([t for _, t in xs], .5), q([t for _, t in xs], .9),
                             q([t / n for n, t in xs], .5))
    for b in BINS:
        (n, c50, c90, crtf), (_, p50, p90, prtf) = cells[(SERVERS[0][0], b)], cells[(SERVERS[1][0], b)]
        print(f"| {b[0]}–{b[1]} s | {n} | {c50 * 1e3:.0f} / {c90 * 1e3:.0f} ms | {p50 * 1e3:.0f} / {p90 * 1e3:.0f} ms "
              f"| {crtf:.3f} | {prtf:.3f} |")


def completion_groups(rows, gap=0.002):
    """Requests whose replies arrived within ``gap`` seconds of each other: one decoded batch, since the
    server sends a batch's results together once the batch is done."""
    rows = sorted(rows, key=lambda r: r["t_done"])
    out, cur = [], [rows[0]]
    for r in rows[1:]:
        if r["t_done"] - cur[-1]["t_done"] <= gap:
            cur.append(r)
        else:
            out.append(cur)
            cur = [r]
    return out + [cur]


def batching(root):
    """How replies come back at high load: group sizes, one round's timeline, padding."""
    print("\n## Kết quả trả về theo batch\n")
    print("| Kịch bản, server | Nhóm trả về cùng lúc (≤ 2 ms): số nhóm theo cỡ | Audio GPU phải tính / audio thật |")
    print("|---|---|---:|")
    for t, c in (("burst-same", 128), ("burst-mixed", 128), ("sustain-mixed", 128)):
        for s, name in SERVERS:
            path = root / s / t / f"c{c}.jsonl"
            if not path.exists():
                continue
            rows = [r for r in read(path) if r["ok"]]
            groups = []
            for rnd in sorted({r.get("round", 0) for r in rows}):
                groups += completion_groups([r for r in rows if r.get("round", 0) == rnd])
            sizes = {}
            for g in groups:
                sizes[len(g)] = sizes.get(len(g), 0) + 1
            multi = [g for g in groups if len(g) > 1]
            padded = sum(max(r["length_s"] for r in g) * len(g) for g in multi) / sum(r["length_s"] for g in multi for r in g)
            print(f"| {t}, {c} người, {name} | " + ", ".join(f"cỡ {k}: {v}" for k, v in sorted(sizes.items()))
                  + f" | {padded:.2f}× |")
    rows = [r for r in read(root / SERVERS[0][0] / "burst-mixed" / "c128.jsonl") if r["ok"] and r["round"] == 1]
    t0 = min(r["t_send"] for r in rows)
    groups = completion_groups(rows)
    print(f"\nMột lượt 128 người gửi cùng lúc, câu 1.4–33 s, {SERVERS[0][1]}: {len(groups)} nhóm\n")
    print("| Nhóm | Trả về lúc | Số request | Độ dài các câu |")
    print("|---:|---:|---:|---|")
    for i, g in enumerate(groups):
        if i < 6 or i >= len(groups) - 3:
            lengths = ", ".join(f"{x:.1f}" for x in sorted(r["length_s"] for r in g))
            print(f"| {i + 1} | {g[0]['t_done'] - t0:.2f} s | {len(g)} | {lengths} s |")
        elif i == 6:
            print("| … | | | |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=RUN, help="folder under baseline/results/")
    a = ap.parse_args()
    root = REPO / "baseline" / "results" / a.run
    data = {(s, t): sweep(root / s / t) for s, _ in SERVERS for t, _ in SCENARIOS}

    alone(root)

    print(f"\n## Số người dùng tối đa ({a.run})\n")
    print("| Kịch bản | Server | " + " | ".join(f"Latency P90 ≤ {x:g} s" for x in SLAS) + " | Trần xử lý |")
    print("|---|---|" + "---:|" * (len(SLAS) + 1))
    for t, label in SCENARIOS:
        for s, name in SERVERS:
            if data[(s, t)]:
                lv = data[(s, t)][1]
                print(f"| {label} | {name} | " + " | ".join(str(capacity(lv, x) or "—") for x in SLAS)
                      + f" | {max(v['audio_s_per_s'] for v in lv.values()):.0f} s audio/giây |")

    for t, label in SCENARIOS:
        sides = [data[(s, t)] for s, _ in SERVERS]
        if not all(sides):
            continue
        (sc, cpp), (_, py) = sides
        print(f"\n## {label}\n")
        print(f"WER chạy riêng: {sc['wer_alone']:.2%}.\n")
        print("| Người dùng cùng lúc | Latency P50 / P90, C++ gốc | Latency P50 / P90, Python | RTF P50 / P90, C++ gốc "
              "| RTF P50 / P90, Python | Xử lý, C++ gốc | Xử lý, Python |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for c in sorted(set(cpp) | set(py)):
            vc, vp = cpp.get(c), py.get(c)
            print(f"| {c} | {pair(vc, 'lat50', 'lat90', unit=' s')} | {pair(vp, 'lat50', 'lat90', unit=' s')} "
                  f"| {pair(vc, 'rtf50', 'rtf90', 3)} | {pair(vp, 'rtf50', 'rtf90', 3)} "
                  f"| {speed(vc)} | {speed(vp)} |")
        for name, lv in (("C++ gốc", cpp), ("Python", py)):
            print(f"\n{name}, theo mức: batch TB / GPU bận % / CPU server (nhân) / WER: " + ", ".join(
                f"{c}: {f(v['batch'], 1)} / {f(v['gpu'], 0)} / {f(v['server_cpu'], 1)} / {v['wer']:.2%}"
                for c, v in lv.items()))
        if t.endswith("mixed"):
            print("\nLatency P90 theo độ dài câu, C++ gốc:\n")
            print("| Người dùng | " + " | ".join(f"{lo}–{hi} s" for lo, hi in BINS) + " |")
            print("|---:|" + "---:|" * len(BINS))
            for c, v in cpp.items():
                print(f"| {c} | " + " | ".join(f(v["by_length"][b]) for b in BINS) + " |")

    batching(root)


if __name__ == "__main__":
    main()
