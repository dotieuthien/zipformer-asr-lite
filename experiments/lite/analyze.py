"""Recompute the tables in OPTIMIZATION.md from the raw rows.

    python3 experiments/lite/analyze.py

Standard library only, no GPU. Inputs, one row per request in <scenario>/c<N>.jsonl:

    baseline/results/gpu-20261017/cpp-b5-w3-t2/   sherpa C++ (phase 1)
    lite/results/gpu-20261018/lite-c/             lite, one GPU thread (step 3)
    lite/results/gpu-20261018/lite-d/             + 30 s windows, boundary warm-up (steps 4, 5)
    lite/results/gpu-20261018b/lite-e/            + encoder on its own thread (step 6): the final lite
    lite/results/gpu-20261018b/lite-f/            + VRAM release at 7.4 GB (step 7)

latency: burst, first byte sent -> text; sustain, end of the utterance -> text (queueing included).
Percentiles: linear interpolation at (n-1)p over every request of a level, as in vieneu.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHERPA = REPO / "baseline/results/gpu-20261017/cpp-b5-w3-t2"
LITE_C = REPO / "lite/results/gpu-20261018/lite-c"
LITE_D = REPO / "lite/results/gpu-20261018/lite-d"
LITE_E = REPO / "lite/results/gpu-20261018b/lite-e"
LITE_F = REPO / "lite/results/gpu-20261018b/lite-f"
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


def summary(folder):
    return json.loads((folder / "summary.json").read_text())


def sweep(folder):
    out = {}
    for lv in summary(folder)["levels"]:
        c = lv["concurrency"]
        rows = [r for r in read(folder / f"c{c}.jsonl") if not r.get("warm")]
        ok = [r for r in rows if r["ok"]]
        lat = [r["latency_s"] if r["mode"] == "burst" else r["delay_s"] for r in ok]
        sends = sorted(r["t_send"] for r in ok)
        span = (sends[-1] - sends[0]) if len(sends) > 1 else None
        out[c] = dict(n=len(rows), errors=len(rows) - len(ok), p50=q(lat, .5), p90=q(lat, .9),
                      audio=lv["audio_s_per_s"], rate=(len(sends) - 1) / span if span else None,
                      wer=sum(r["word_errors"] for r in ok) / max(1, sum(r["words"] for r in ok)),
                      match=lv["match"], server=lv.get("server_stats") or {}, vram_max=lv.get("gpu_mem_max_mb"))
    return out


def capacity(levels, sla=1.0):
    return max((c for c, v in levels.items() if not v["errors"] and v["p90"] is not None and v["p90"] <= sla),
               default=None)


def ceiling(levels):
    return max(v["audio"] for v in levels.values())


def f(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def cell(v):
    return "—" if v is None else f"{f(v['p50'])} / {f(v['p90'])} s"


def table(title, cols, levels, rate=False):
    """cols: [(label, sweep)]. One row per level any of them ran."""
    print(f"\n### {title}\n")
    head = "| Người dùng | " + ("Request đến mỗi giây | " if rate else "")
    head += " | ".join(f"P50 / P90, {name}" for name, _ in cols) + " | "
    head += " | ".join(f"Xử lý, {name}" for name, _ in cols) + " |"
    print(head)
    print("|---:|" + ("---:|" if rate else "") + "---:|" * (2 * len(cols)))
    for c in levels:
        vs = [s.get(c) for _, s in cols]
        r = next(v["rate"] for v in vs if v)
        print(f"| {c} | " + (f"{f(r, 1)} | " if rate else "") + " | ".join(cell(v) for v in vs) + " | "
              + " | ".join("—" if v is None else f"{v['audio']:.0f} s/giây" for v in vs) + " |")


def main():
    data = {tag: (sweep(SHERPA / tag), sweep(LITE_C / tag), sweep(LITE_E / tag)) for tag, _ in SCENARIOS}

    print("## Kết quả: sherpa, lite một luồng (lite-c), lite cuối (lite-e)\n")
    print("| Kịch bản | Người dùng tối đa (P90 ≤ 1 s): sherpa / lite-c / lite-e | Trần xử lý (s audio/giây): sherpa / lite-c / lite-e |")
    print("|---|---:|---:|")
    for tag, label in SCENARIOS:
        s, c, e = data[tag]
        print(f"| {label} | {capacity(s)} / {capacity(c)} / {capacity(e)} | "
              f"{ceiling(s):.0f} / {ceiling(c):.0f} / {ceiling(e):.0f} |")
    s, c, e = data["burst-same"]
    print(f"\nMột câu ~10 s một mình (P50): sherpa {f(s[1]['p50'], 3)} s, lite-c {f(c[1]['p50'], 3)} s, "
          f"lite-e {f(e[1]['p50'], 3)} s")
    print("Độ đúng lúc có tải, lite-e: " + "; ".join(
        f"{tag} khớp {min(v['match'] for v in data[tag][2].values()):.3f}–{max(v['match'] for v in data[tag][2].values()):.3f}, "
        f"WER {min(v['wer'] for v in data[tag][2].values()):.2%}–{max(v['wer'] for v in data[tag][2].values()):.2%}"
        for tag, _ in SCENARIOS))
    errors = sum(v["errors"] for tag, _ in SCENARIOS for v in data[tag][2].values())
    print(f"Lỗi, lite-e, 4 kịch bản: {errors}")

    print("\n## Từng kịch bản: sherpa so với lite cuối")
    for tag, label in SCENARIOS:
        s, _, e = data[tag]
        table(label, [("sherpa", s), ("lite", e)], sorted(set(s) | set(e)), rate=tag.startswith("sustain"))

    print("\n## Bước 3: lite một luồng, đo theo mức (rải đều, câu ~10 s)\n")
    _, c, _ = data["sustain-same"]
    for lv, v in c.items():
        sv = v["server"]
        print(f"  {lv}: P90 {f(v['p90'])} s, xử lý {v['audio']:.0f}, câu mỗi lượt prefill {f(sv.get('prefill_size_mean'), 2)}, "
              f"prefill chiếm {f(sv.get('admit_share'), 2)} thời gian luồng GPU, tick {f(sv.get('tick_ms'), 2)} ms, "
              f"lượt prefill mỗi giây {f(sv['admits'] / sv['clock_s'] if sv.get('clock_s') else None, 1)}")

    print("\n## Bước 6: lite một luồng so với hai luồng (rải đều)")
    for tag in ("sustain-same", "sustain-mixed"):
        s, c, e = data[tag]
        table(dict(SCENARIOS)[tag], [("sherpa", s), ("lite một luồng", c), ("lite hai luồng", e)],
              sorted(set(s) | set(c) | set(e)), rate=True)
        print("\nLite hai luồng, theo mức: câu mỗi lượt prefill / luồng prefill bận / luồng greedy bận / tick")
        print(", ".join(f"{lv}: {f(v['server'].get('prefill_size_mean'), 1)} / {f(v['server'].get('prefill_busy'), 2)} / "
                        f"{f(v['server'].get('greedy_busy'), 2)} / {f(v['server'].get('tick_ms'), 2)} ms"
                        for lv, v in e.items()))
        print("Lite một luồng, câu mỗi lượt prefill / phần thời gian prefill: " + ", ".join(
            f"{lv}: {f(v['server'].get('prefill_size_mean'), 1)} / {f(v['server'].get('admit_share'), 2)}"
            for lv, v in c.items()))

    print("\n## Bước 4: audio dài (cửa sổ 30 s)\n")
    for folder in (LITE_D, LITE_E, LITE_F):
        for tag in ("burst-long", "long-utts"):
            p = folder / tag / "summary.json"
            if p.exists():
                sm = json.loads(p.read_text())
                print(f"{folder.name}/{tag}: {sm['utterances']} files/utterances, {sm['audio_s']['min']:.0f}-"
                      f"{sm['audio_s']['max']:.0f} s (mean {sm['audio_s']['mean']:.0f}), WER alone {sm['wer_alone']:.2%}, "
                      f"latency alone P50 {q(sm['alone_latency_s'], .5):.3f} s")
    for folder in (LITE_D, LITE_E):
        print(f"\n{folder.name}/burst-long:")
        for lv, v in sweep(folder / "burst-long").items():
            print(f"  {lv} người: P50 / P90 {cell(v)}, {v['audio']:.0f} s audio/giây, WER {v['wer']:.2%}, khớp {v['match']:.3f}")

    print("\n## Bước 5 và 7: VRAM\n")
    for folder in (LITE_D, LITE_E, LITE_F):
        out = folder / "server.out"
        line = next((x for x in out.read_text().splitlines() if "model ready" in x), "") if out.exists() else ""
        print(f"{folder.name}: sau làm nóng: {line.split('GPU memory used')[-1].strip() if line else '—'}")
        for tag in ("burst-same", "burst-mixed", "sustain-same", "sustain-mixed", "burst-long", "long-utts"):
            p = folder / tag / "summary.json"
            if not p.exists():
                continue
            lv = summary(folder / tag)["levels"]
            used = [x["server_stats"]["gpu_used_mb"] for x in lv if x.get("server_stats")]
            peak = [x["gpu_mem_max_mb"] for x in lv if x.get("gpu_mem_max_mb")]
            rel = sum(x["server_stats"].get("vram_releases", 0) for x in lv if x.get("server_stats"))
            print(f"  {tag}: VRAM cuối mỗi mức {f(min(used, default=None), 0)}–{f(max(used, default=None), 0)} MB, "
                  f"đỉnh lấy mẫu {f(max(peak, default=None), 0)} MB, số lần nhả trong các mức {rel}")
    f_sm = sweep(LITE_F / "sustain-mixed")
    e_sm = data["sustain-mixed"][2]
    print("\nlite-f so với lite-e, rải đều câu lẫn lộn (P90): " + ", ".join(
        f"{lv}: {f(e_sm[lv]['p90'])} / {f(v['p90'])} s (VRAM đỉnh {f(v['vram_max'], 0)} MB, nhả {v['server'].get('vram_releases', 0)})"
        for lv, v in f_sm.items() if lv in e_sm))
    print(f"Người dùng tối đa, rải đều câu lẫn lộn: lite-e {capacity(e_sm)}, lite-f {capacity(f_sm)}")


if __name__ == "__main__":
    main()
