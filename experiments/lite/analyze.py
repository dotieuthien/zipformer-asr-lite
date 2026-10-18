"""Recompute the tables in OPTIMIZATION.md: sherpa's C++ server against lite, from the raw rows.

    python3 experiments/lite/analyze.py

Standard library only, no GPU. Inputs, one row per request in <scenario>/c<N>.jsonl:

    baseline/results/gpu-20261017/cpp-b5-w3-t2/   sherpa C++ (phase 1)
    lite/results/gpu-20261018/lite-c/             lite, CUDA graph greedy
    lite/results/gpu-20261018/lite-d/             lite with 30 s windows, for long audio

latency: burst, first byte sent -> text; sustain, end of the utterance -> text (queueing included).
Percentiles: linear interpolation at (n-1)p over every request of a level, as in vieneu.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHERPA = REPO / "baseline/results/gpu-20261017/cpp-b5-w3-t2"
LITE = REPO / "lite/results/gpu-20261018/lite-c"
LONG = REPO / "lite/results/gpu-20261018/lite-d"
SCENARIOS = [("burst-same", "Dồn một lúc, câu ~10 s"), ("burst-mixed", "Dồn một lúc, câu 1.4–33 s"),
             ("sustain-same", "Rải đều, câu ~10 s"), ("sustain-mixed", "Rải đều, câu 1.4–33 s")]


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
    summary = json.loads((folder / "summary.json").read_text())
    out = {}
    for lv in summary["levels"]:
        c = lv["concurrency"]
        rows = [r for r in read(folder / f"c{c}.jsonl") if not r.get("warm")]
        ok = [r for r in rows if r["ok"]]
        lat = [r["latency_s"] if r["mode"] == "burst" else r["delay_s"] for r in ok]
        sends = sorted(r["t_send"] for r in ok)
        span = (sends[-1] - sends[0]) if len(sends) > 1 else None
        out[c] = dict(n=len(rows), errors=len(rows) - len(ok), p50=q(lat, .5), p90=q(lat, .9),
                      audio=lv["audio_s_per_s"], rate=(len(sends) - 1) / span if span else None,
                      wer=sum(r["word_errors"] for r in ok) / max(1, sum(r["words"] for r in ok)),
                      server=lv.get("server_stats"), gpu=lv.get("gpu_util_mean"))
    return summary, out


def capacity(levels, sla):
    return max((c for c, v in levels.items() if not v["errors"] and v["p90"] is not None and v["p90"] <= sla),
               default=None)


def f(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def main():
    print("## Tóm tắt\n")
    print("| Kịch bản | Người dùng tối đa (P90 ≤ 1 s), sherpa | lite | Trần xử lý (s audio/giây), sherpa | lite |")
    print("|---|---:|---:|---:|---:|")
    data = {}
    for tag, label in SCENARIOS:
        (_, s), (_, l) = sweep(SHERPA / tag), sweep(LITE / tag)
        data[tag] = (s, l)
        print(f"| {label} | {capacity(s, 1.0)} | {capacity(l, 1.0)} | {max(v['audio'] for v in s.values()):.0f} "
              f"| {max(v['audio'] for v in l.values()):.0f} |")

    for tag, label in SCENARIOS:
        s, l = data[tag]
        rate = tag.startswith("sustain")
        print(f"\n## {label}\n")
        head = "| Người dùng | " + ("Request đến mỗi giây | " if rate else "") + \
               "P50 / P90, sherpa | P50 / P90, lite | Xử lý, sherpa | Xử lý, lite |"
        print(head)
        print("|---:|" + ("---:|" if rate else "") + "---:|---:|---:|---:|")
        for c in sorted(set(s) | set(l)):
            vs, vl = s.get(c), l.get(c)
            cell = lambda v: "—" if v is None else f"{f(v['p50'])} / {f(v['p90'])} s"  # noqa: E731
            audio = lambda v: "—" if v is None else f"{v['audio']:.0f} s/giây"  # noqa: E731
            r = (vs or vl)["rate"]
            print(f"| {c} | " + (f"{f(r, 1)} | " if rate else "") + f"{cell(vs)} | {cell(vl)} | {audio(vs)} | {audio(vl)} |")
        print("\nLite, theo mức: câu mỗi lượt prefill / phần thời gian luồng GPU dành cho prefill / VRAM:")
        print(", ".join(f"{c}: {f(v['server']['prefill_size_mean'], 1)} / {f(v['server']['admit_share'], 2)} / "
                        f"{f(v['server']['gpu_used_mb'], 0)} MB" for c, v in l.items() if v["server"]))

    print("\n## Audio dài (lite, cửa sổ 30 s)\n")
    for tag in ("burst-long", "long-utts"):
        p = LONG / tag / "summary.json"
        if p.exists():
            sm = json.loads(p.read_text())
            print(f"{tag}: {sm['utterances']} files/utterances, {sm['audio_s']['min']:.0f}-{sm['audio_s']['max']:.0f} s, "
                  f"WER alone {sm['wer_alone']:.2%}, latency P50 alone {q(sm['alone_latency_s'], .5):.3f} s")
    if (LONG / "burst-long" / "summary.json").exists():
        _, lv = sweep(LONG / "burst-long")
        for c, v in lv.items():
            print(f"  burst-long {c} người: P50 / P90 {f(v['p50'])} / {f(v['p90'])} s, {v['audio']:.0f} s audio/giây, "
                  f"WER {v['wer']:.2%}, VRAM {f(v['server'] and v['server']['gpu_used_mb'], 0)} MB")


if __name__ == "__main__":
    main()
