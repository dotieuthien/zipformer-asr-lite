"""Part 1 of the explainer: how Zipformer turns speech into text, and how sherpa-onnx's own server runs it.

utt.npz is utterance 1188-133604-0041 of LibriSpeech test-clean (10.07 s): its waveform, a 0.3 s zoom (0.78–1.08 s),
and a 40-band log-mel picture (only for drawing). TOKENS are the tokens and frames sherpa-onnx outputs for it.
Numbers come from PHASE1.md.
"""
import random

import numpy as np
from PIL import Image

from draw import (BG, BLUE, F_B, F_H, F_S, F_XS, GREEN, HERE, INK, LINE, MUTED, ORANGE, PURPLE, RED, USERS, W,
                  arrow, clamp, ctext, ease, font, mix, person, rbox, seg, text, tint, wrap)

UTT = np.load(HERE / "utt.npz")
ENV, ZOOM, FBANK = UTT["env"].astype(np.float32), UTT["zoom"].astype(np.float32), UTT["fbank"]
# (token, encoder frame) from sherpa-onnx for 1188-133604-0041, first 1.2 s; a frame is 40 ms.
TOKENS = [("▁IT", 0), ("▁IS", 10), ("▁A", 16), ("▁G", 20), ("LE", 22), ("AN", 25), ("ER", 29)]
TEXT = "IT IS A GLEANER BRINGING DOWN HER ONE SHEAF OF CORN TO AN OLD WATER MILL ITSELF MOSSY IN RENT…"


def waveform(d, x, y, w, h, samples, color, upto=1.0):
    """Min/max per pixel column of samples in [-1, 1], drawn up to a fraction `upto` of the width."""
    n = len(samples)
    for i in range(int(w * clamp(upto))):
        a = samples[int(i / w * n): max(int(i / w * n) + 1, int((i + 1) / w * n))]
        lo, hi = float(a.min()), float(a.max())
        d.line([x + i, y - hi * h / 2, x + i, y - lo * h / 2], fill=color, width=1)


def envelope(d, x, y, w, h, color, upto=1.0):
    n = len(ENV)
    for i in range(int(w * clamp(upto))):
        v = float(ENV[min(n - 1, int(i / w * n))])
        d.line([x + i, y - v * h / 2, x + i, y + v * h / 2], fill=color, width=1)


def fbank_image(x, y, w, h, frames=None, upto=1.0):
    """The utterance's log-mel picture (low frequencies at the bottom) as an RGB image."""
    f = FBANK if frames is None else FBANK[frames[0]:frames[1]]
    a = f.T[::-1].astype(np.float32) / 255.0                     # (40, T)
    lo, hi = np.array(BG, np.float32), np.array(BLUE, np.float32) * 0.85
    rgb = (lo[None, None] + (hi - lo)[None, None] * a[..., None] ** 1.3).astype(np.uint8)
    img = Image.fromarray(rgb).resize((w, h), Image.NEAREST)
    cut = int(w * clamp(upto))
    return img.crop((0, 0, max(1, cut), h)), (x, y)


def s_title(img, d, p, t):
    ctext(d, W / 2, 210, "Zipformer, sherpa-onnx và bản lite", font(46, True))
    ctext(d, W / 2, 290, "Chép tiếng nói thành chữ · phục vụ nhiều người gửi cùng lúc", F_B, MUTED)
    envelope(d, 190, 430, 900, 110, BLUE, upto=ease(seg(p, 0.0, 0.6)))
    a = ease(seg(p, 0.45, 0.8))
    ctext(d, W / 2, 530, TEXT[: int(len(TEXT) * a)], F_S, mix(INK, a * 2))


def s_offline(img, d, p, t):
    person(d, 150, 260, BLUE)
    text(d, 150, 312, "người dùng", F_S, MUTED, anchor="ma")
    x0, x1 = 230, 760
    talk = seg(p, 0.0, 0.5)
    text(d, x0, 178, "đang nói câu 1 · 10 giây", F_B, INK)
    rbox(d, x0, 224, x1 - x0, 64, (255, 255, 255), outline=LINE, r=12)
    envelope(d, x0 + 10, 256, x1 - x0 - 20, 54, BLUE, upto=talk)
    text(d, x0, 300, "server chưa nhận gì, chưa làm gì cho người này", F_S, mix(MUTED, 1 - seg(p, 0.5, 0.55)))
    s = ease(seg(p, 0.5, 0.62))
    if s > 0:
        arrow(d, x1 + 10, 256, x1 + 10 + 150 * s, 256, mix(ORANGE, s), w=5)
        text(d, x1 + 14, 272, "nói xong,", F_S, mix(ORANGE, s))
        text(d, x1 + 14, 300, "gửi cả câu", F_S, mix(ORANGE, s))
    busy = seg(p, 0.6, 0.72)
    rbox(d, 940, 196, 230, 120, tint(GREEN, 0.10 + 0.25 * (busy > 0 and busy < 1)), outline=GREEN, r=18)
    ctext(d, 1055, 238, "Server", F_H, GREEN)
    ctext(d, 1055, 278, "chép cả câu ~0.1 s" if busy > 0 else "đang rảnh", F_S, MUTED)
    r = ease(seg(p, 0.72, 0.85))
    if r > 0:
        arrow(d, 1055, 330, 1055, 330 + 60 * r, mix(GREEN, r), w=5)
        lines = wrap(TEXT, F_S, 1000, d)
        rbox(d, 110, 410, 1060, 30 + 32 * len(lines), mix((255, 255, 255), r), outline=mix(GREEN, r), r=14)
        for i, ln in enumerate(lines):
            text(d, 134, 424 + 32 * i, ln, F_S, mix(INK, r))
        text(d, 110, 432 + 32 * len(lines) + 10, "chữ của cả câu, trả về một lần", F_S, mix(GREEN, r))


def s_fbank(img, d, p, t):
    x0, w = 110, 1060
    text(d, x0 + w, 58, "0.3 giây âm thanh, phóng to", F_S, MUTED, anchor="ra")
    rbox(d, x0, 166, w, 130, (255, 255, 255), outline=LINE, r=12)
    waveform(d, x0 + 10, 231, w - 20, 110, ZOOM, BLUE)
    step = (w - 20) / 30                                          # 10 ms
    k = int(seg(p, 0.05, 0.85) * 30 + 1e-6)
    kk = max(1, min(k, 27))
    wx = x0 + 10 + kk * step
    # a frame is 25 ms long; the next one starts 10 ms later, so they overlap
    d.rectangle([wx - step, 174, wx + 1.5 * step, 288], outline=mix(ORANGE, 0.4), width=2)
    d.rectangle([wx, 170, wx + 2.5 * step, 292], outline=ORANGE, width=3)
    arrow(d, wx - step, 154, wx - 2, 154, ORANGE, w=3, head=9)
    text(d, wx - step - 8, 142, "10 ms", F_XS, ORANGE, anchor="ra")
    d.line([wx, 304, wx + 2.5 * step, 304], fill=ORANGE, width=2)
    for xx in (wx, wx + 2.5 * step):
        d.line([xx, 298, xx, 310], fill=ORANGE, width=2)
    text(d, wx + 1.25 * step, 312, "khung dài 25 ms", F_XS, ORANGE, anchor="ma")
    # one column of the picture per 10 ms step, filled as the window moves
    frame0 = 78                                                   # 0.78 s = frame 78
    pic, at = fbank_image(x0 + 10, 350, int(w - 20), 170, (frame0, frame0 + 30), upto=k / 30)
    rbox(d, x0, 340, w, 190, (255, 255, 255), outline=LINE, r=12)
    img.paste(pic, at)
    text(d, x0 - 8, 352, "cao", F_XS, MUTED, anchor="ra")
    text(d, x0 - 8, 500, "trầm", F_XS, MUTED, anchor="ra")
    a = ease(seg(p, 0.55, 0.75))
    text(d, x0, 540, "khung dài 25 ms, cứ 10 ms lấy một khung (chồng lên nhau) = 100 khung mỗi giây · mỗi khung 80 số",
         F_S, mix(INK, a))


def s_encoder(img, d, p, t):
    # input: the whole utterance's fbank
    text(d, 60, 136, "fbank cả câu", F_S, MUTED)
    pic, at = fbank_image(60, 170, 220, 300)
    img.paste(pic, at)
    text(d, 60, 478, "100 khung/giây", F_XS, MUTED)
    arrow(d, 292, 320, 340, 320, INK, w=4)
    # the Zipformer stacks, each at its own frame rate
    rbox(d, 350, 136, 560, 400, (255, 255, 255), outline=PURPLE, r=18)
    text(d, 370, 150, "Encoder Zipformer", F_H, PURPLE)
    rates = [(50, 16), (25, 8), (12.5, 4), (6.25, 2), (12.5, 4), (25, 8)]
    lit = seg(p, 0.05, 0.5) * len(rates)
    for i, (rate, n) in enumerate(rates):
        y = 205 + i * 52
        on = clamp(lit - i)
        bw = 400 / n
        for j in range(n):
            rbox(d, 380 + j * bw, y, bw - 4, 36, tint(PURPLE, 0.08 + 0.3 * on), outline=mix(PURPLE, 0.3 + 0.7 * on),
                 r=6, width=1)
        text(d, 792, y + 8, f"{rate:g} khung/s", F_XS, mix(INK, 0.4 + 0.6 * on))
    # output: 25 frames per second, 512 numbers each
    o = ease(seg(p, 0.5, 0.65))
    arrow(d, 920, 320, 920 + 50 * o, 320, mix(INK, o), w=4)
    if o > 0:
        rng = random.Random(1)
        for j in range(10):
            for k in range(12):
                v = rng.random()
                d.rectangle([985 + j * 19, 170 + k * 25, 985 + j * 19 + 15, 170 + k * 25 + 21],
                            fill=mix(tint(GREEN, 0.15 + 0.6 * v), o))
        text(d, 985, 478, "25 khung/giây", F_XS, mix(INK, o))
        text(d, 985, 500, "mỗi khung 512 số", F_XS, mix(INK, o))
    c = ease(seg(p, 0.6, 0.75))
    ctext(d, 630, 572, "chạy MỘT lần cho cả câu · 64 triệu tham số · phần nặng nhất", F_B, mix(RED, c))


def s_greedy(img, d, p, t):
    n, bw, x0, y0 = 30, 35, 115, 150
    f = min(n - 1, int(seg(p, 0.04, 0.9) * n + 1e-6))
    out = dict((fr, tok) for tok, fr in TOKENS)
    text(d, x0, 118, "các khung của encoder (mỗi khung 40 ms)", F_S, MUTED)
    for i in range(n):
        x = x0 + i * bw
        cur = i == f
        rbox(d, x, y0, bw - 5, 46, tint(GREEN, 0.5 if cur else 0.15) if i <= f else BG,
             outline=INK if cur else (GREEN if i <= f else LINE), r=6, width=3 if cur else 1)
        if i <= f:
            if i in out:
                d.text((x + (bw - 5) / 2, y0 + 62), out[i].lstrip("▁"), font=F_XS, fill=BLUE, anchor="ma")
            else:
                d.text((x + (bw - 5) / 2, y0 + 62), "·", font=F_XS, fill=MUTED, anchor="ma")
    text(d, x0, y0 + 90, "chữ xanh = token mới · chấm xám = blank (chưa có chữ mới)", F_XS, MUTED)
    # the step for the current frame
    done = [tok for tok, fr in TOKENS if fr < f]
    ctx = (["–", "–"] + done)[-2:]
    cx = x0 + f * bw + (bw - 5) / 2
    d.line([cx, y0 + 46, cx, 330], fill=mix(INK, 0.5), width=2)
    rbox(d, cx - 70, 330, 140, 64, tint(GREEN, 0.15), outline=GREEN, r=12)
    ctext(d, cx, 362, f"khung {f}", F_S, GREEN)
    rbox(d, 140, 430, 300, 90, tint(ORANGE, 0.12), outline=ORANGE, r=14)
    ctext(d, 290, 458, "Decoder", F_H, ORANGE)
    ctext(d, 290, 494, "2 token vừa viết: " + " ".join(c.lstrip("▁") for c in ctx), F_S, MUTED)
    jx = 560
    rbox(d, jx, 430, 180, 90, tint(PURPLE, 0.12), outline=PURPLE, r=14)
    ctext(d, jx + 90, 460, "Joiner", F_H, PURPLE)
    ctext(d, jx + 90, 494, "điểm cho 500 token", F_XS, MUTED)
    arrow(d, 440, 475, jx - 6, 475, ORANGE, w=3, head=11)
    arrow(d, cx, 394, clamp(cx, jx + 30, jx + 150), 424, GREEN, w=3, head=11)
    tok = out.get(f)
    rbox(d, 960, 430, 210, 90, (255, 255, 255), outline=BLUE if tok else LINE, r=14)
    ctext(d, 1065, 460, "điểm cao nhất", F_XS, MUTED)
    ctext(d, 1065, 492, tok.lstrip("▁") if tok else "blank", F_H, BLUE if tok else MUTED)
    arrow(d, jx + 186, 475, 954, 475, PURPLE, w=3, head=11)
    words = "".join(tok.replace("▁", " ") for tok, fr in TOKENS if fr <= f).strip()
    text(d, 140, 548, "chữ đến giờ:", F_B, MUTED)
    text(d, 310, 548, words, font(28, True), INK)


def s_server(img, d, p, t):
    # clients -> network thread -> one queue -> 3 workers (up to 5 utterances each) -> GPU
    for i in range(5):
        person(d, 80, 195 + i * 84, USERS[i], 0.75)
    rbox(d, 150, 260, 150, 110, (255, 255, 255), outline=INK, r=14)
    ctext(d, 225, 300, "luồng mạng", F_S)
    ctext(d, 225, 332, "websocket", F_XS, MUTED)
    arrow(d, 112, 315, 146, 315, INK, w=3, head=10)
    arrow(d, 304, 315, 346, 315, INK, w=3, head=10)
    rbox(d, 350, 200, 170, 230, (255, 255, 255), outline=INK, r=14)
    ctext(d, 435, 226, "hàng đợi chung", F_S)
    q = seg(p, 0.0, 0.3)
    taken = seg(p, 0.3, 0.45)
    waiting = int(13 * q) if taken == 0 else int(13 * (1 - taken))
    for k in range(waiting):
        r_, c_ = divmod(k, 3)
        d.rounded_rectangle([370 + c_ * 46, 252 + r_ * 34, 370 + c_ * 46 + 38, 252 + r_ * 34 + 26], 6,
                            fill=tint(USERS[k % 8], 0.3), outline=USERS[k % 8])
    sizes = [5, 5, 3]
    for w_ in range(3):
        y = 160 + w_ * 128
        rbox(d, 600, y, 330, 108, (255, 255, 255), outline=INK, r=14)
        text(d, 616, y + 10, f"luồng làm việc {w_ + 1}", F_S)
        for k in range(int(sizes[w_] * taken + 1e-6)):
            col = USERS[(w_ * 5 + k) % 8]
            d.rounded_rectangle([616 + k * 46, y + 46, 616 + k * 46 + 38, y + 72], 6, fill=tint(col, 0.3), outline=col)
        text(d, 616, y + 82, "tối đa 5 câu mỗi batch", F_XS, MUTED)
        arrow(d, 524, 315, 596, y + 54, MUTED, w=2, head=9)
        arrow(d, 934, y + 54, 990, 315, MUTED, w=2, head=9)
    rbox(d, 995, 250, 190, 130, tint(BLUE, 0.12), outline=BLUE, r=16)
    ctext(d, 1090, 292, "GPU", F_H, BLUE)
    ctext(d, 1090, 332, "onnxruntime", F_S, MUTED)
    s = ease(seg(p, 0.55, 0.75))
    if s > 0:
        rbox(d, 110, 540, 1060, 44, mix((255, 255, 255), s), outline=mix(LINE, s), r=12)
        text(d, 130, 549, "mỗi batch:  fbank (CPU)  →  đệm bằng câu dài nhất  →  encoder 1 lần (GPU)  →  greedy",
             F_S, mix(INK, s))


def s_batch(img, d, p, t):
    lens = [3.0, 4.5, 10.0, 2.0, 6.0]
    x0, w, y0 = 250, 800, 140
    scale = w / 10.0
    cur = seg(p, 0.05, 0.55) * 10.0
    text(d, x0, y0 - 30, "một batch 5 câu, greedy chạy tới hết câu dài nhất", F_S, MUTED)
    for i, L in enumerate(lens):
        y = y0 + i * 50
        col = USERS[i]
        text(d, x0 - 20, y + 6, f"{L:g} s", F_S, col, anchor="ra")
        d.rectangle([x0, y, x0 + w, y + 34], fill=(236, 237, 240))
        for k in range(int(w / 12)):
            if x0 + L * scale + k * 12 < x0 + w:
                d.line([x0 + L * scale + k * 12, y + 34, x0 + L * scale + k * 12 + 10, y], fill=LINE, width=1)
        rbox(d, x0, y, L * scale, 34, tint(col, 0.25), outline=col, r=6)
        if cur >= L and cur < 10:
            text(d, x0 + w + 16, y + 6, "xong, vẫn phải chờ", F_XS, RED)
        elif cur >= 10:
            text(d, x0 + w + 16, y + 6, "trả cùng lúc", F_XS, GREEN)
    cx = x0 + cur * scale
    d.line([cx, y0 - 6, cx, y0 + 5 * 50 - 10], fill=INK, width=3)
    text(d, x0, y0 + 5 * 50, "phần gạch xám = đệm, GPU vẫn phải tính", F_XS, MUTED)
    # CPU <-> GPU every frame
    b = ease(seg(p, 0.55, 0.7))
    if b > 0:
        y = 445
        rbox(d, 110, y, 200, 80, mix((255, 255, 255), b), outline=mix(INK, b), r=14)
        ctext(d, 210, y + 40, "CPU: chọn token", F_S, mix(INK, b))
        rbox(d, 970, y, 200, 80, mix(tint(BLUE, 0.12), b), outline=mix(BLUE, b), r=14)
        ctext(d, 1070, y + 40, "GPU: joiner", F_S, mix(BLUE, b))
        ctext(d, 640, y + 4, "mỗi khung chép qua rồi chép về", F_S, mix(ORANGE, b))
        d.line([330, y + 44, 950, y + 44], fill=mix(LINE, b), width=2)
        u = (t * 1.4) % 2.0
        x = 330 + (u if u < 1 else 2 - u) * 620
        d.ellipse([x - 10, y + 34, x + 10, y + 54], fill=mix(ORANGE, b))
        ctext(d, 640, y + 80, "câu 10 s = 250 lượt qua lại", F_XS, mix(MUTED, b))


def s_result(img, d, p, t):
    a = ease(seg(p, 0.0, 0.25))
    rbox(d, 390, 170, 500, 230, mix(tint(BLUE, 0.1), a), outline=mix(BLUE, a), r=20)
    ctext(d, 640, 250, "24 người", font(72, True), mix(BLUE, a))
    ctext(d, 640, 335, "gửi cùng một lúc", F_B, mix(INK, a))
    c = ease(seg(p, 0.3, 0.5))
    ctext(d, W / 2, 450, "câu ~10 s · P90 thời gian chờ ≤ 1 giây", F_H, mix(INK, c))
    ctext(d, W / 2, 495, "một RTX 3060 12 GB · server C++ v1.13.8, cấu hình mặc định", F_B, mix(MUTED, c))


TITLE = ("", "", s_title, [
    "Video này giải thích cách Zipformer chép tiếng nói thành chữ, cách server sherpa-onnx chạy nó, "
    "và bản lite làm gì để phục vụ nhiều người hơn.",
])
SCENES = [
    ("Bài toán", "ASR offline: nói xong cả câu mới chép", s_offline, [
        "Đây là ASR offline: người dùng nói xong cả câu mới gửi, server trả về chữ của cả câu một lần.",
    ]),
    ("Bước 1", "Sóng âm thành fbank", s_fbank, [
        "Bước một: cứ 10 mili giây lấy một khung âm thanh, đổi thành 80 con số theo dải tần, gọi là fbank.",
    ]),
    ("Bước 2", "Encoder Zipformer đọc cả câu", s_encoder, [
        "Bước hai: encoder Zipformer đọc cả câu cùng lúc, các tầng bên trong nhìn câu ở nhiều độ chi tiết.",
        "Nó ra 25 khung mỗi giây, mỗi khung 512 số. Đây là phần nặng nhất: 64 triệu tham số.",
    ]),
    ("Bước 3", "Greedy search: đi qua từng khung", s_greedy, [
        "Bước ba: greedy search đi qua từng khung. Joiner ghép khung với 2 token vừa viết, chọn token điểm cao nhất.",
        "Phần lớn khung ra blank, tức chưa có chữ mới. Mỗi khung tối đa một token, ghép lại thành câu.",
    ]),
    ("Server", "sherpa-onnx gom câu thành batch", s_server, [
        "Server sherpa-onnx xếp câu vào một hàng đợi chung. Ba luồng làm việc, mỗi luồng lấy tối đa 5 câu.",
        "Mỗi batch: fbank trên CPU, rồi encoder chạy một lần trên GPU, đệm mọi câu bằng câu dài nhất.",
    ]),
    ("Giới hạn", "Cả batch xong mới trả", s_batch, [
        "Greedy chạy tới hết câu dài nhất, cả batch xong mới trả, nên câu ngắn phải chờ câu dài.",
        "Mỗi khung, dữ liệu lại chép qua lại giữa CPU và GPU, nên GPU thường đứng chờ.",
    ]),
]
BASELINE = ("Server gốc", "Sức chứa khi dồn một lúc", s_result, [
    "Server gốc trên một RTX 3060: giữ P90 dưới 1 giây, phục vụ được 24 người gửi cùng một lúc.",
])
