"""Render docs/how-it-works.mp4 (+ .vi.srt, poster): one explainer video for both phases.

    python docs/video/make_video.py --voice http://127.0.0.1:PORT   # read the subtitles with VieNeu lite
    python docs/video/make_video.py                                 # cached voice-over, or a silent video

Part 1 (phase1.py): how Zipformer turns speech into text and how sherpa-onnx's own server runs it.
Part 2 (phase2.py): what the lite server does differently when many people send at the same moment.
Needs Pillow, numpy and ffmpeg (libx264, aac). Drawing, timing and the voice-over are in draw.py.
"""
import argparse

import phase1
import phase2
from draw import fetch_voice, render

SCENES = [phase1.TITLE, *phase1.SCENES, phase2.MEASURE, phase1.BASELINE, *phase2.SCENES]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", metavar="URL", help="VieNeu lite server that reads the subtitles aloud")
    ap.add_argument("--speaker", default=None, help="one of its voices (GET /v1/voices); default: the server's")
    a = ap.parse_args()
    if a.voice:
        cues = [c for *_, cs in SCENES for c in cs]
        for i, c in enumerate(cues):
            fetch_voice(a.voice.rstrip("/"), c, a.speaker)
            print(f"voice {i + 1}/{len(cues)}", flush=True)
    render("how-it-works", SCENES, ("Bước 3", 0.62), a.speaker)   # poster: greedy search, mid-way


if __name__ == "__main__":
    main()
