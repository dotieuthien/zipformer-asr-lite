"""Recompute the tables in README.md from the raw per-request files.

    python3 experiments/baseline/analyze.py                  # the committed run
    python3 experiments/baseline/analyze.py --run <folder>   # your own rerun

Standard library only, no GPU. Input: baseline/results/<run>/<server>/<scenario>/c<N>.jsonl, one row
per request as loadtest.py wrote it, and summary.json next to it (GPU and CPU samples, batch sizes).
Percentiles use linear interpolation at (n-1)p over every counted request of a level.
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = "gpu-20261017"
SERVERS = [("cpp-b5-w3-t2", "C++ gốc"), ("python-b5-w3-t2", "Python port")]
SCENARIOS = [("burst-same", "Gửi cùng lúc, câu ~10 s"), ("burst-mixed", "Gửi cùng lúc, câu 1.4–33 s"),
             ("sustain-same", "Nói liên tục, câu ~10 s"), ("sustain-mixed", "Nói liên tục, câu 1.4–33 s")]
BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 40)]  # seconds, as loadtest.py
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


def sweep(folder):
    """{users: stats} for one server and scenario, recomputed from the per-request rows."""
    if not (folder / "summary.json").exists():
        return None
    meta = json.loads((folder / "summary.json").read_text())
    summary = {lv["concurrency"]: lv for lv in meta["levels"]}
    out = {"wer_alone": meta["wer_alone"]}
    for c, lv in summary.items():
        rows = [r for r in read(folder / f"c{c}.jsonl") if not r.get("warm")]
        ok = [r for r in rows if r["ok"]]
        metric = "latency_s" if rows[0]["mode"] == "burst" else "delay_s"
        xs = [r[metric] for r in ok]
        out[c] = dict(n=len(rows), errors=len(rows) - len(ok), p50=q(xs, .5), p90=q(xs, .9), p99=q(xs, .99),
                      match=sum(r["match"] for r in ok) / max(1, len(ok)),
                      wer=sum(r["word_errors"] for r in ok) / max(1, sum(r["words"] for r in ok)),
                      by_length={b: q([r[metric] for r in ok if b[0] <= r["length_s"] < b[1]], .9) for b in BINS},
                      audio_s_per_s=lv["audio_s_per_s"], batch=lv["batch_size_mean"], gpu=lv["gpu_util_mean"],
                      server_cpu=lv["server_cpu_cores_mean"])
    return out


def capacity(levels, sla):
    return max((c for c, v in users(levels) if not v["errors"] and v["p90"] is not None and v["p90"] <= sla),
               default=None)


def users(levels):
    return sorted((c, v) for c, v in levels.items() if isinstance(c, int))


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


def f(x, nd=2, unit=""):
    return "—" if x is None else f"{x:.{nd}f}{unit}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=RUN, help="folder under baseline/results/")
    a = ap.parse_args()
    root = REPO / "baseline" / "results" / a.run
    data = {(s, t): sweep(root / s / t) for s, _ in SERVERS for t, _ in SCENARIOS}

    print(f"## Số người dùng tối đa ({a.run})\n")
    print("Mức cao nhất đã đo mà mọi request đều thành công và P90 thời gian chờ kết quả không quá ngưỡng.\n")
    print("| Kịch bản | Server | " + " | ".join(f"P90 ≤ {x:g} s" for x in SLAS) + " | Mức cao nhất đã đo |")
    print("|---|---|" + "---:|" * (len(SLAS) + 1))
    for t, label in SCENARIOS:
        for s, name in SERVERS:
            lv = data[(s, t)]
            if lv:
                print(f"| {label} | {name} | " + " | ".join(str(capacity(lv, x) or "—") for x in SLAS)
                      + f" | {max(c for c, _ in users(lv))} |")

    for t, label in SCENARIOS:
        sides = [(name, data[(s, t)]) for s, name in SERVERS if data[(s, t)]]
        if not sides:
            continue
        print(f"\n## {label}\n")
        print("| Người dùng | " + " | ".join(f"P50 / P90, {name} | Audio/giây, {name}" for name, _ in sides) + " |")
        print("|---:|" + "---:|---:|" * len(sides))
        for c in sorted(set().union(*({c for c, _ in users(lv)} for _, lv in sides))):
            cells = []
            for _, lv in sides:
                v = lv.get(c)
                cells += [f"{f(v['p50'])} / {f(v['p90'])} s" + (f" ({v['errors']} lỗi)" if v["errors"] else ""),
                          f"{v['audio_s_per_s']:.0f} s"] if v else ["—", "—"]
            print(f"| {c} | " + " | ".join(cells) + " |")
        for name, lv in sides:
            print(f"\n{name}, WER chạy riêng {lv['wer_alone']:.2%}. Theo mức: batch TB / GPU % / CPU server (nhân) / "
                  "WER lúc tải / khớp transcript chạy riêng: " + ", ".join(
                      f"{c}: {f(v['batch'], 1)} / {f(v['gpu'], 0)} / {f(v['server_cpu'], 1)} / {v['wer']:.2%} / "
                      f"{v['match']:.0%}" for c, v in users(lv)))
        if t.endswith("mixed"):
            for name, lv in sides:
                print(f"\nP90 theo độ dài câu, {name}:\n")
                print("| Người dùng | " + " | ".join(f"{lo}–{hi} s" for lo, hi in BINS) + " |")
                print("|---:|" + "---:|" * len(BINS))
                for c, v in users(lv):
                    print(f"| {c} | " + " | ".join(f(v["by_length"][b]) for b in BINS) + " |")
    batching(root)


if __name__ == "__main__":
    main()
