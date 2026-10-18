"""Part 2 of the explainer: what the lite server does differently when many people send at the same moment.

Only the burst scenario (N người gửi cùng một lúc). Numbers come from OPTIMIZATION.md and the result files it links:
sherpa C++ in baseline/results/gpu-20261017/, the final lite in lite/results/gpu-20261018b/lite-e/.
"""
import random

import draw
from draw import (BLUE, F_B, F_H, F_S, F_XS, GREEN, INK, LINE, MUTED, ORANGE, PURPLE, RED, TEAL, USERS, W,
                  arrow, clamp, ctext, ease, font, mix, person, rbox, seg, text, tint)

F_BIG = font(56, True)
LANE_BG = (232, 234, 238)
# Burst, ~10 s utterances: P90 by number of users (OPTIMIZATION.md, bước 8).
P90 = {8: (0.75, 0.15), 16: (0.87, 0.29), 32: (1.33, 0.45), 64: (1.65, 0.81), 128: (2.75, 1.49)}


def cue():
    return draw.CUE


def appear(x, a, b=None):
    return ease(seg(x, a, a + 0.15 if b is None else b))


def token(d, x, y, color, a=1.0, w=40, h=28):
    d.rounded_rectangle([x, y, x + w, y + h], 6, fill=mix(tint(color, 0.3), a), outline=mix(color, a))


def slot_grid(d, x0, y0, cw, ch, states):
    """4 x 8 slots; states[i] = (color or None, progress 0..1, flash 0..1)."""
    for i, (col, prog, flash) in enumerate(states):
        r, c = divmod(i, 8)
        x, y = x0 + c * (cw + 8), y0 + r * (ch + 8)
        if col is None:
            rbox(d, x, y, cw, ch, (255, 255, 255), outline=LINE, r=8, width=1)
            continue
        rbox(d, x, y, cw, ch, tint(col, 0.12), outline=col, r=8, width=2)
        if prog > 0:
            d.rectangle([x + 6, y + ch - 14, x + 6 + (cw - 12) * clamp(prog), y + ch - 8], fill=col)
        if flash > 0:
            rbox(d, x, y, cw, ch, mix(tint(GREEN, 0.35), flash), outline=GREEN, r=8, width=3)
            ctext(d, x + cw / 2, y + ch / 2 - 6, "xong", F_XS, mix(GREEN, flash))


# ── scenes ───────────────────────────────────────────────────────────────────

def s_burst(img, d, p, t):
    waits = [0.18, 0.26, 0.31, 0.38, 0.44, 0.52, 0.61, 0.83]
    go = appear(p, 0.08, 0.2)
    for i in range(8):
        y = 160 + i * 52
        person(d, 100, y + 12, USERS[i], 0.45)
        if go > 0:
            arrow(d, 130, y + 12, 130 + 230 * go, 330, mix(USERS[i], go * 0.8), w=2, head=8)
    rbox(d, 370, 280, 220, 100, tint(GREEN, 0.12), outline=GREEN, r=16)
    ctext(d, 480, 330, "Server", F_H, GREEN)
    x0, scale = 680, 440                       # 1 s = 440 px
    text(d, x0, 128, "chờ bao lâu mới có chữ", F_S, MUTED)
    for i, w in enumerate(waits):
        y = 160 + i * 52
        a = appear(p, 0.25 + w * 0.35, 0.3 + w * 0.35)
        d.rectangle([x0, y, x0 + scale * w * a, y + 26], fill=mix(USERS[i], a))
        if a >= 1:
            text(d, x0 + scale * w + 8, y + 2, f"{w:.2f} s", F_XS, INK)
    one = x0 + scale
    for yy in range(158, 570, 14):
        d.line([one, yy, one, yy + 7], fill=RED, width=2)
    text(d, one + 8, 128, "1 giây", F_S, RED)
    k = appear(p, 0.7)
    if k > 0:
        rbox(d, 370, 430, 250, 120, mix((255, 255, 255), k), outline=mix(INK, k), r=14)
        ctext(d, 495, 466, "P90 = 0.83 s", F_H, mix(INK, k))
        ctext(d, 495, 512, "dưới 1 giây: đạt", F_S, mix(GREEN, k))
    ci, q = cue()
    if ci >= 1:
        a = appear(q, 0.0, 0.25)
        text(d, 70, 568, "N = 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128 … · mỗi mức 5 lượt", F_S, mix(INK, a))


def s_prefill(img, d, p, t):
    text(d, 70, 128, "16 câu mới vào", F_S, MUTED)
    move = appear(p, 0.2, 0.35)
    for k in range(16):
        r_, c_ = divmod(k, 2)
        x = 70 + c_ * 56 + move * 260
        token(d, x, 168 + r_ * 44, USERS[k % 8], a=1 - move)
    pulse = seg(p, 0.3, 0.55)
    hot = 0.12 + 0.25 * (0 < pulse < 1)
    rbox(d, 360, 220, 340, 200, tint(PURPLE, hot), outline=PURPLE, r=18, width=3)
    ctext(d, 530, 280, "fbank + encoder", F_H, PURPLE)
    ctext(d, 530, 330, "trên GPU", F_B, INK)
    ctext(d, 530, 370, "MỘT lần cho cả nhóm", F_B, RED)
    arrow(d, 210, 320, 352, 320, INK, w=4)
    arrow(d, 708, 320, 770, 320, INK, w=4)
    filled = int(appear(p, 0.55, 0.75) * 16 + 1e-6)
    states = [(USERS[i % 8], 0.0, 0.0) if i < filled else (None, 0, 0) for i in range(32)]
    text(d, 790, 128, "32 slot", F_S, MUTED)
    slot_grid(d, 790, 168, 40, 70, [s for s in states])
    k = appear(p, 0.75)
    text(d, 70, 560, "nhóm đệm theo câu dài nhất · tối đa 16 câu và 240 s audio sau khi đệm", F_S, mix(MUTED, k))


def s_graph(img, d, p, t):
    # sherpa: every frame goes CPU -> GPU -> CPU
    text(d, 90, 130, "Server gốc, mỗi khung:", F_B, INK)
    x, y = 90, 175
    for i in range(14):
        for col, w in ((ORANGE, 22), (LINE, 12), (BLUE, 14), (LINE, 12)):
            d.rectangle([x, y, x + w - 2, y + 30], fill=col)
            x += w
    text(d, 90, 214, "cam = CPU chọn token · xám = chép dữ liệu · xanh = GPU tính · câu 10 s = 250 lượt", F_XS, MUTED)
    # lite: 8 steps recorded once, replayed with one call
    b = appear(p, 0.48, 0.6)
    text(d, 90, 280, "Bản lite: 8 bước greedy ghi thành MỘT CUDA graph", F_B, mix(INK, b))
    rbox(d, 90, 324, 560, 130, mix(tint(RED, 0.06), b), outline=mix(RED, b), r=16)
    for s in range(8):
        for k in range(5):
            d.rectangle([110 + s * 66 + k * 11, 350, 110 + s * 66 + k * 11 + 8, 378],
                        fill=mix((PURPLE, BLUE, GREEN, ORANGE, TEAL)[k], b))
        text(d, 110 + s * 66 + 26, 388, f"{s + 1}", F_XS, mix(MUTED, b), anchor="ma")
    text(d, 110, 418, "mỗi bước: joiner, chọn token, decoder, cho cả 32 slot", F_XS, mix(MUTED, b))
    c = appear(p, 0.62, 0.74)
    if c > 0:
        arrow(d, 662, 389, 742, 389, mix(INK, c), w=4)
        rbox(d, 752, 349, 120, 80, mix(GREEN, c), r=14)
        d.polygon([(795, 370), (795, 408), (830, 389)], fill=(255, 255, 255))
        text(d, 812, 440, "1 lệnh", F_S, mix(GREEN, c), anchor="ma")
        arrow(d, 880, 389, 940, 389, mix(INK, c), w=4)
        rbox(d, 950, 344, 230, 90, mix(tint(BLUE, 0.12), c), outline=mix(BLUE, c), r=14)
        ctext(d, 1065, 374, "GPU", F_H, mix(BLUE, c))
        ctext(d, 1065, 410, "32 slot × 8 khung", F_XS, mix(INK, c))
    e = appear(p, 0.78)
    ctext(d, W / 2, 520, "khoảng 1 ms cho 8 bước của cả 32 slot, CPU không phải chen vào từng bước", F_B, mix(GREEN, e))



def _jobs(seed=4, n=32):
    """Per slot: list of (start, length) in seconds of animation; a slot refills one tick after it frees."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        s, jobs = -rng.random() * 3.0, []
        while s < 40:
            L = rng.uniform(2.0, 5.0)
            jobs.append((s, L))
            s += L + 0.35
        out.append(jobs)
    return out


JOBS = _jobs()


def s_slots(img, d, p, t):
    states = []
    for i, jobs in enumerate(JOBS):
        st = (None, 0.0, 0.0)
        for k, (s, L) in enumerate(jobs):
            if s <= t < s + L:
                st = (USERS[(i + k * 3) % 8], (t - s) / L, 0.0)
            elif s + L <= t < s + L + 0.35:
                st = (USERS[(i + k * 3) % 8], 1.0, 1.0 - (t - s - L) / 0.35)
        states.append(st)
    text(d, 70, 128, "32 slot, mỗi thanh = tiến độ greedy của một câu", F_S, MUTED)
    slot_grid(d, 70, 166, 78, 54, states)
    ci, q = cue()
    x = 800
    lines = ["Slot xong: trả chữ ngay,", "không chờ slot khác.", "", "Slot trống: nhận câu mới", "ở tick sau (~1 ms)."]
    for i, ln in enumerate(lines):
        text(d, x, 176 + i * 38, ln, F_B, INK if i < 2 else MUTED)
    b = 1.0 if ci >= 1 else 0.0
    if b:
        a = appear(q, 0.0, 0.2)
        y = 470
        text(d, 70, y - 34, "hai luồng, hai CUDA stream, chạy song song", F_S, mix(INK, a))
        rbox(d, 300, y, 880, 40, LANE_BG, r=8)
        rbox(d, 300, y + 56, 880, 40, LANE_BG, r=8)
        text(d, 70, y + 8, "luồng prefill", F_S, mix(PURPLE, a))
        text(d, 70, y + 64, "luồng greedy", F_S, mix(GREEN, a))
        for k in range(5):
            bx = 310 + k * 175
            d.rounded_rectangle([bx, y + 6, bx + 120, y + 34], 6, fill=mix(PURPLE, a))
            text(d, bx + 60, y + 9, f"nhóm {k + 1}", F_XS, mix((255, 255, 255), a), anchor="ma")
        for k in range(108):
            bx = 310 + k * 8
            d.rectangle([bx, y + 62, bx + 5, y + 90], fill=mix(GREEN, a))


def s_result(img, d, p, t):
    x0, y0, w, h = 120, 150, 620, 360
    ymax = 3.0
    users = list(P90)
    gw = w / len(users)
    d.line([x0, y0 + h, x0 + w, y0 + h], fill=INK, width=2)
    for v in (1, 2, 3):
        yy = y0 + h - v / ymax * h
        text(d, x0 - 10, yy - 10, f"{v} s", F_XS, MUTED, anchor="ra")
    one = y0 + h - 1 / ymax * h
    for xx in range(x0, x0 + w, 16):
        d.line([xx, one, xx + 8, one], fill=RED, width=2)
    text(d, x0 + w + 6, one - 10, "1 giây", F_XS, RED)
    g = appear(p, 0.02, 0.3)
    for i, u in enumerate(users):
        s_, l_ = P90[u]
        cx = x0 + i * gw + gw / 2
        for k, (v, col) in enumerate(((s_, ORANGE), (l_, GREEN))):
            bh = v / ymax * h * g
            bx = cx - 40 + k * 42
            d.rectangle([bx, y0 + h - bh, bx + 36, y0 + h], fill=col)
            if g >= 1:
                text(d, bx + 18, y0 + h - bh - 22, f"{v:.2f}", F_XS, col, anchor="ma")
        text(d, cx, y0 + h + 10, f"{u}", F_S, INK, anchor="ma")
    text(d, x0 + w / 2, y0 + h + 40, "số người gửi cùng lúc · P90 thời gian chờ · câu ~10 s", F_XS, MUTED, anchor="ma")
    d.rectangle([x0, 120, x0 + 16, 136], fill=ORANGE)
    text(d, x0 + 24, 116, "server gốc", F_S, ORANGE)
    d.rectangle([x0 + 180, 120, x0 + 196, 136], fill=GREEN)
    text(d, x0 + 204, 116, "bản lite", F_S, GREEN)
    ci, q = cue()
    x = 800
    a = appear(p, 0.15, 0.35)
    text(d, x, 150, "Người dùng tối đa, P90 ≤ 1 s", F_S, mix(MUTED, a))
    text(d, x, 182, "câu ~10 s", F_B, mix(INK, a))
    text(d, x, 220, "24 → 64", F_BIG, mix(GREEN, a))
    if ci >= 1:
        rows = [("câu 1.4–33 s", "16 → 64 người"), ("trần xử lý, câu ~10 s", "422 → 825 s audio/giây"),
                ("chữ so với server gốc", "giống hệt 355/355 câu")]
        for i, (k, v) in enumerate(rows):
            b = appear(q, 0.05 + i * 0.2, 0.25 + i * 0.2)
            text(d, x, 312 + i * 76, k, F_S, mix(MUTED, b))
            text(d, x, 340 + i * 76, v, F_H, mix(INK, b))


def curve(d, x0, y0, w, h, ymax, pts, color, upto=1.0, width=4):
    """Polyline of (x fraction, GB) points, drawn up to x fraction `upto`."""
    xy = []
    for i, (fx, v) in enumerate(pts):
        if fx > upto:
            if i:
                (px, pv) = pts[i - 1]
                k = (upto - px) / (fx - px) if fx > px else 1.0
                xy.append((x0 + upto * w, y0 + h - (pv + (v - pv) * k) / ymax * h))
            break
        xy.append((x0 + fx * w, y0 + h - v / ymax * h))
    if len(xy) > 1:
        d.line(xy, fill=color, width=width, joint="curve")


# Illustrative paths; the levels are measured (PHASE1.md, OPTIMIZATION.md bước 5 và 7). x = time, y = GB.
SHERPA_VRAM = [(0, 1.0), (0.12, 1.0), (0.12, 2.4), (0.35, 2.4), (0.35, 4.1), (0.6, 4.1), (0.6, 6.6), (1.0, 6.6)]
LITE_KEEP = [(0, 7.2), (0.4, 7.2), (0.55, 7.6), (0.7, 7.9), (0.85, 8.2), (1.0, 8.4)]
LITE_RELEASE = [(0, 7.2), (0.4, 7.2), (0.52, 7.58), (0.52, 1.4), (0.62, 2.3), (0.8, 3.9), (1.0, 4.9)]


def s_vram(img, d, p, t):
    x0, y0, w, h, ymax = 130, 140, 700, 400, 9.0
    for v in range(0, 10, 3):
        yy = y0 + h - v / ymax * h
        d.line([x0, yy, x0 + w, yy], fill=(232, 234, 238), width=1)
        text(d, x0 - 12, yy - 10, f"{v} GB", F_XS, MUTED, anchor="ra")
    d.line([x0, y0 + h, x0 + w, y0 + h], fill=INK, width=2)
    text(d, x0 + w, y0 + h + 8, "thời gian →", F_XS, MUTED, anchor="ra")
    ci, q = cue()
    a0 = 1.0 if ci > 0 else appear(q, 0.05, 0.6)
    curve(d, x0, y0, w, h, ymax, SHERPA_VRAM, ORANGE, upto=a0)
    text(d, x0 + w + 12, y0 + h - 6.6 / ymax * h - 12, "6,6", F_S, mix(ORANGE, a0))
    if ci >= 1:
        a1 = 1.0 if ci > 1 else appear(q, 0.05, 0.5)
        curve(d, x0, y0, w, h, ymax, LITE_RELEASE, GREEN, upto=0.4 * a1)
        text(d, x0 + 6, y0 + h - 7.2 / ymax * h + 12, "lite: đỉnh 7,2 GB ngay lúc khởi động", F_XS, mix(GREEN, a1))
    if ci >= 2:
        a2 = appear(q, 0.05, 0.7)
        thr = y0 + h - 7.58 / ymax * h
        for xx in range(x0, x0 + w, 16):
            d.line([xx, thr, xx + 8, thr], fill=RED, width=2)
        text(d, x0 + 0.42 * w, y0 + 4, "câu đủ kiểu độ dài →", F_XS, MUTED)
        curve(d, x0, y0, w, h, ymax, LITE_KEEP, (150, 156, 165), upto=0.4 + 0.6 * a2, width=3)
        curve(d, x0, y0, w, h, ymax, LITE_RELEASE, GREEN, upto=0.4 + 0.6 * a2)
        if a2 >= 1:
            text(d, x0 + w + 12, y0 + h - 8.4 / ymax * h - 12, "8,4", F_S, MUTED)
            text(d, x0 + w + 12, y0 + h - 4.9 / ymax * h - 12, "4,9", F_S, GREEN)
            text(d, x0 + 0.52 * w + 8, y0 + h - 1.4 / ymax * h - 6, "nhả: 1,4", F_S, GREEN)
    # what each line means
    rows = [(ORANGE, "server gốc", "tăng theo batch lớn nhất, không trần", 0),
            (GREEN, "bản lite, nhả ở ngưỡng", "chặn lượt prefill lớn nhất, chạy thử lúc khởi động", 1),
            ((150, 156, 165), "bản lite, không nhả", "phân mảnh khi độ dài thay đổi liên tục", 2)]
    for i, (col, name, what, k) in enumerate(rows):
        a = 1.0 if ci > k else (appear(q, 0.05, 0.3) if ci == k else 0.0)
        y = 140 + i * 118
        d.rectangle([900, y + 6, 920, y + 26], fill=mix(col, a))
        text(d, 932, y, name, F_B, mix(INK, a))
        for j, ln in enumerate(wrap_text(what, 31)):
            text(d, 932, y + 36 + j * 26, ln, F_XS, mix(MUTED, a))
    if ci >= 2:
        text(d, 900, 510, "ngưỡng nhả: 7,4 GiB", F_S, mix(RED, appear(q, 0.1, 0.4)))
    text(d, x0, y0 + h + 34, "đường minh họa · các mốc là số đo thật", F_XS, MUTED)


def wrap_text(s, n):
    out, cur = [], ""
    for wd in s.split():
        if len(cur) + len(wd) + 1 > n and cur:
            out.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    return out + [cur] if cur else out


def s_summary(img, d, p, t):
    steps = [("Prefill một lần cho cả nhóm tới 16 câu", PURPLE),
             ("Greedy trên GPU: 8 bước = 1 CUDA graph ≈ 1 ms", BLUE),
             ("Xong là trả, slot trống nhận câu mới ngay", GREEN),
             ("Encoder chạy song song với greedy", ORANGE),
             ("VRAM có trần, vượt ngưỡng thì tự nhả", RED)]
    for i, (s, c) in enumerate(steps):
        a = appear(p, i * 0.1, i * 0.1 + 0.15)
        y = 136 + i * 74
        d.ellipse([110, y, 162, y + 52], fill=mix(c, a))
        ctext(d, 136, y + 26, str(i + 1), F_H, mix((255, 255, 255), a))
        text(d, 190, y + 12, s, F_H, mix(INK, a))
    e = appear(p, 0.6)
    ctext(d, W / 2, 530, "Cùng model, cùng loại GPU: 24 → 64 người gửi cùng lúc", font(32, True), mix(GREEN, e))


MEASURE = ("Cách đo", "Dồn một lúc", s_burst, [
    "Kịch bản đo: N người bấm gửi cùng một lúc. Đo xem chờ bao lâu mới có chữ; P90 dưới 1 giây thì đạt.",
    "Mỗi mức gửi 5 lượt như vậy, rồi tăng N, tới khi P90 vượt 1 giây.",
])
SCENES = [
    ("Thay đổi 1", "Prefill một lần cho cả nhóm", s_prefill, [
        "Bản lite có 32 slot. Câu mới được gom thành nhóm tới 16 câu, chạy fbank và encoder trên GPU đúng một lần.",
    ]),
    ("Thay đổi 2", "Greedy trên GPU bằng CUDA graph", s_graph, [
        "Ở server gốc, mỗi khung CPU phải gọi GPU rồi chờ kết quả về. Câu 10 giây là 250 lượt như vậy.",
        "Bản lite chạy greedy hẳn trên GPU: 8 bước ghi thành một CUDA graph, một lệnh là cả 32 slot đi 8 khung.",
    ]),
    ("Thay đổi 3", "Xong là trả, slot trống nhận câu mới", s_slots, [
        "Slot nào xong là trả chữ ngay, slot trống nhận câu mới ở tick sau. Câu ngắn không phải chờ câu dài.",
        "Encoder chạy ở luồng riêng: trong lúc nhóm sau đang prefill, greedy của các slot vẫn chạy tiếp.",
    ]),
    ("Kết quả", "Dồn một lúc, cùng loại RTX 3060", s_result, [
        "Câu 10 giây, giữ P90 dưới 1 giây: server gốc phục vụ 24 người, bản lite 64 người.",
        "Câu dài ngắn lẫn lộn: 16 lên 64 người. Trần xử lý gần gấp đôi, và chữ giống hệt server gốc.",
    ]),
    ("VRAM", "Có trần, vượt ngưỡng thì nhả", s_vram, [
        "Còn VRAM: server gốc tăng bộ nhớ theo batch lớn nhất từng gặp, lên tới 6,6 GB, không có trần.",
        "Bản lite chặn lượt prefill lớn nhất ở 16 câu và 240 giây audio, rồi chạy thử nó ngay lúc khởi động.",
        "Khi độ dài câu thay đổi liên tục, bộ nhớ bị phân mảnh. Vượt ngưỡng thì onnxruntime trả vùng trống, "
        "VRAM tụt về 1,4 GB.",
    ]),
    ("Tóm tắt", "Năm thay đổi", s_summary, [
        "Tóm lại: cùng model, cùng loại GPU. Chỉ đổi cách server chạy, số người gửi cùng lúc tăng từ 24 lên 64.",
    ]),
]
