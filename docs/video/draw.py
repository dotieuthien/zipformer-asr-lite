"""Drawing, timeline, voice-over and rendering for the explainer video (make_video.py).

Same approach as vieneu's docs/video: every frame is drawn with Pillow and piped to ffmpeg; the subtitles are drawn
into the picture and also written as .srt. With a voice-over, each subtitle is read by the VieNeu lite server
(POST /v1/audio/speech, 48 kHz PCM), cached in docs/video/.voice/ by what is read, and lasts as long as its reading
plus a short pause. Without one, a subtitle lasts long enough to read (~16 characters a second).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import urllib.request
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 24
HERE = Path(__file__).resolve().parent
OUT = HERE.parent
SCENE_PAUSE = 0.7       # extra seconds when the scene changes
PAUSE = 0.4             # seconds after each subtitle that is read aloud
SR = 48_000             # VieNeu's PCM
VOICE_DIR = HERE / ".voice"
FONT_DIR = Path(os.getenv("FONT_DIR", "/System/Library/Fonts/Supplemental"))
# Text the TTS reads differently from the subtitle. VieNeu reads English terms (CPU, GPU, encoder…) well by itself;
# only "mili giây" needs help, as in vieneu's own video.
SPOKEN = {"mili giây": "mi li giây"}


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ("Arial Bold.ttf" if bold else "Arial.ttf")), size)


F_TITLE, F_H, F_B, F_S, F_XS, F_SUB = font(40, True), font(30, True), font(26), font(22), font(18), font(28)

BG = (247, 246, 241)
INK = (31, 35, 40)
MUTED = (110, 118, 129)
LINE = (208, 212, 218)
BLUE = (47, 111, 235)
ORANGE = (232, 131, 58)
GREEN = (45, 164, 78)
RED = (207, 34, 46)
PURPLE = (130, 80, 223)
TEAL = (26, 150, 160)
USERS = [BLUE, ORANGE, GREEN, PURPLE, TEAL, RED, (191, 135, 0), (219, 97, 162)]


# ── drawing helpers ──────────────────────────────────────────────────────────

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def seg(p, a, b):
    """Progress 0..1 of p inside [a, b]."""
    return clamp((p - a) / (b - a)) if b > a else float(p >= a)


def mix(c, a):
    """Fade colour c into the background: a=1 full colour, 0 invisible."""
    return tuple(int(BG[i] + (c[i] - BG[i]) * clamp(a)) for i in range(3))


def tint(c, k=0.18):
    return tuple(int(255 + (c[i] - 255) * k) for i in range(3))


def rbox(d, x, y, w, h, fill, outline=None, r=14, width=2):
    d.rounded_rectangle([x, y, x + w, y + h], r, fill=fill, outline=outline, width=width)


def text(d, x, y, s, f, fill=INK, anchor="la"):
    d.text((x, y), s, font=f, fill=fill, anchor=anchor)


def ctext(d, cx, cy, s, f, fill=INK):
    lines = s.split("\n")
    lh = f.size * 1.25
    y0 = cy - lh * (len(lines) - 1) / 2
    for i, ln in enumerate(lines):
        d.text((cx, y0 + i * lh), ln, font=f, fill=fill, anchor="mm")


def arrow(d, x1, y1, x2, y2, fill=MUTED, w=4, head=14):
    d.line([x1, y1, x2, y2], fill=fill, width=w)
    a = math.atan2(y2 - y1, x2 - x1)
    d.polygon([(x2, y2), (x2 - head * math.cos(a - 0.45), y2 - head * math.sin(a - 0.45)),
               (x2 - head * math.cos(a + 0.45), y2 - head * math.sin(a + 0.45))], fill=fill)


def chip(d, x, y, s, color, f=F_S):
    w = d.textlength(s, font=f) + 24
    rbox(d, x, y, w, f.size + 14, tint(color, 0.16), outline=color, r=12)
    text(d, x + 12, y + 7, s, f, color)
    return w


def wrap(s, f, maxw, d):
    words, lines, cur = s.split(), [], ""
    for wd in words:
        test = f"{cur} {wd}".strip()
        if d.textlength(test, font=f) <= maxw or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def person(d, cx, cy, color, k=1.0):
    d.ellipse([cx - 16 * k, cy - 44 * k, cx + 16 * k, cy - 12 * k], fill=color)
    d.rounded_rectangle([cx - 26 * k, cy - 6 * k, cx + 26 * k, cy + 40 * k], 14 * k, fill=color)


def spoken(s):
    for a, b in sorted(SPOKEN.items(), key=lambda kv: -len(kv[0])):
        s = re.sub(rf"(?<!\w){re.escape(a)}(?!\w)", b, s)
    return s


def voice_path(s, voice):
    return VOICE_DIR / f"{hashlib.sha1(json.dumps([spoken(s), voice]).encode()).hexdigest()[:16]}.wav"


def fetch_voice(base, s, voice):
    """Read one subtitle with the VieNeu lite server; keep it as a 48 kHz mono WAV."""
    path = voice_path(s, voice)
    if path.exists():
        return
    body = json.dumps({"input": spoken(s), "voice": voice, "response_format": "pcm"}).encode()
    req = urllib.request.Request(f"{base}/v1/audio/speech", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        pcm = r.read()
    VOICE_DIR.mkdir(exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)


def load_voice(s, voice):
    path = voice_path(s, voice)
    if not path.exists():
        return None
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def voice_track(tl, total, voice):
    """The whole voice-over as one WAV file, or None if any subtitle has not been read."""
    track = np.zeros(int(total * SR) + SR, dtype=np.int16)
    for *_, cues in tl:
        for a, _, c in cues:
            v = load_voice(c, voice)
            if v is None:
                return None
            track[int(a * SR): int(a * SR) + len(v)] = v
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(f.name, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(track[: int(total * SR)].tobytes())
    return f.name


def cue_seconds(s):
    return max(3.0, len(s) / 16.0 + 0.6)


def build_timeline(scenes, voice=None):
    """Scenes with absolute cue times: (t0, t1, label, title, fn, [(a, b, subtitle)])."""
    tl, t0 = [], 0.0
    for label, title, fn, cues in scenes:
        times, t = [], t0
        for c in cues:
            v = load_voice(c, voice) if voice is not False else None
            dur = len(v) / SR + PAUSE if v is not None else cue_seconds(c)
            times.append((t, t + dur, c))
            t += dur
        t += SCENE_PAUSE
        tl.append((t0, t, label, title, fn, times))
        t0 = t
    return tl, t0


def srt_time(x):
    ms = int(round(x * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


CUE = (0, 0.0)   # (index of the subtitle cue inside the current scene, progress 0..1 through it)


def render_frame(tl, now):
    global CUE
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for t0, t1, label, title, fn, cues in tl:
        if t0 <= now < t1 or (now >= t1 and t1 == tl[-1][1]):
            p = clamp((now - t0) / (t1 - t0))
            if title:
                x = 60
                if label:
                    x += chip(d, 60, 44, label, BLUE, F_S) + 16
                text(d, x, 46, title, F_TITLE)
            ci = next((i for i, (a, b, _) in enumerate(cues) if a <= now < b), len(cues) - 1)
            a, b, cue = cues[ci]
            CUE = (ci, clamp((now - a) / (b - a)))
            fn(img, d, p, now - t0)
            lines = wrap(cue, F_SUB, W - 200, d)
            bh = 22 + 38 * len(lines)
            rbox(d, 70, H - bh - 20, W - 140, bh, (24, 27, 31), r=14)
            for i, ln in enumerate(lines):
                d.text((W / 2, H - bh - 20 + 14 + i * 38), ln, font=F_SUB, fill=(255, 255, 255), anchor="ma")
            break
    return img


def render(name, scenes, poster, voice=None):
    """docs/<name>.mp4, .vi.srt and .png; the poster is (scene label, progress through it). With every subtitle
    read (cached), the voice-over goes in and sets the timing; otherwise the video is silent."""
    tl, total = build_timeline(scenes, voice)
    audio = voice_track(tl, total, voice)
    if audio is None:
        tl, total = build_timeline(scenes, False)
    with open(OUT / f"{name}.vi.srt", "w", encoding="utf-8") as f:
        n = 0
        for *_, cues in tl:
            for a, b, c in cues:
                n += 1
                f.write(f"{n}\n{srt_time(a)} --> {srt_time(b)}\n{c}\n\n")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-"]
    if audio:
        cmd += ["-i", audio, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "96k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "slow", "-tune", "animation",
            "-movflags", "+faststart", str(OUT / f"{name}.mp4")]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    frames = int(total * FPS)
    for i in range(frames):
        ff.stdin.write(render_frame(tl, i / FPS).tobytes())
    ff.stdin.close()
    if ff.wait():
        raise SystemExit("ffmpeg failed")
    t0, t1 = next((a, b) for a, b, lb, *_ in tl if lb == poster[0])
    render_frame(tl, t0 + poster[1] * (t1 - t0)).save(OUT / f"{name}.png", optimize=True)
    if audio:
        os.unlink(audio)
    print(f"{total:.1f}s, {frames} frames, voice-over: {'yes' if audio else 'no'} -> {OUT / f'{name}.mp4'}")
