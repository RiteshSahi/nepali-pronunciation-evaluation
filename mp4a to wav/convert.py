#!/usr/bin/env python3
"""
mp4a_to_wav.py — Convert .mp4a / .m4a audio files to .wav

Usage:
    python3 mp4a_to_wav.py input.mp4a
    python3 mp4a_to_wav.py input.mp4a -o output.wav
    python3 mp4a_to_wav.py /path/to/folder            # batch-convert all .mp4a/.m4a files in a folder
    python3 mp4a_to_wav.py /path/to/folder --sample-rate 44100 --channels 2

Requires ffmpeg to be installed and on PATH.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

AUDIO_EXTS = {".mp4a", ".m4a"}


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "Error: ffmpeg is not installed or not on PATH.\n"
            "Install it first, e.g.:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )


def convert_file(src: Path, dst: Path, sample_rate: int, channels: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",                       # overwrite output without asking
        "-i", str(src),
        "-ar", str(sample_rate),    # sample rate
        "-ac", str(channels),       # channels
        "-vn",                      # no video stream (in case of embedded artwork)
        str(dst),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ✗ Failed: {src.name}")
        if result.stderr:
            print(result.stderr.strip().splitlines()[-1])
        return False

    print(f"  ✓ {src.name} -> {dst.name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert .mp4a/.m4a audio to .wav"
    )

    parser.add_argument(
        "input",
        help="Input .mp4a/.m4a file, or a folder to batch-convert"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output .wav path (single-file mode only)"
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Output sample rate (default: 44100)"
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=2,
        help="Output channels (default: 2)"
    )

    args = parser.parse_args()

    check_ffmpeg()

    input_path = Path(args.input)

    if not input_path.exists():
        sys.exit(f"Error: '{input_path}' does not exist.")

    if input_path.is_dir():
        files = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in AUDIO_EXTS
        )

        if not files:
            sys.exit(f"No .mp4a/.m4a files found in '{input_path}'.")

        print(f"Converting {len(files)} file(s) in '{input_path}':")

        ok = 0
        for f in files:
            dst = f.with_suffix(".wav")
            if convert_file(f, dst, args.sample_rate, args.channels):
                ok += 1

        print(f"\nDone: {ok}/{len(files)} converted.")

    else:
        if input_path.suffix.lower() not in AUDIO_EXTS:
            print(
                f"Warning: '{input_path.suffix}' is not .mp4a/.m4a — attempting conversion anyway."
            )

        dst = Path(args.output) if args.output else input_path.with_suffix(".wav")

        print(f"Converting '{input_path}' -> '{dst}'")

        if not convert_file(input_path, dst, args.sample_rate, args.channels):
            sys.exit(1)


if __name__ == "__main__":
    main()