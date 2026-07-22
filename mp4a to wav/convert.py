#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

AUDIO_EXTS = {".m4a", ".mp4a"}


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is not installed. Install it first.")


def convert_file(src, dst, sample_rate, channels):

    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src),
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-vn",
        str(dst)
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        print(f"\n✗ Failed : {src}")
        print(result.stderr)

        return False

    print(f"✓ {src} -> {dst}")

    return True


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        help="File or folder"
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=1
    )

    args = parser.parse_args()

    check_ffmpeg()

    input_path = Path(args.input)

    if not input_path.exists():

        sys.exit(f"{input_path} not found.")

    # --------------------------------------------------
    # Folder
    # --------------------------------------------------

    if input_path.is_dir():

        files = sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        )

        if len(files) == 0:

            sys.exit("No m4a files found.")

        print(f"\nFound {len(files)} file(s)\n")

        success = 0

        for src in files:

            dst = src.with_suffix(".wav")

            if convert_file(
                src,
                dst,
                args.sample_rate,
                args.channels
            ):

                success += 1

        print("\n" + "=" * 50)
        print(f"Converted {success}/{len(files)} files")
        print("=" * 50)

    # --------------------------------------------------
    # Single File
    # --------------------------------------------------

    else:

        dst = input_path.with_suffix(".wav")

        convert_file(
            input_path,
            dst,
            args.sample_rate,
            args.channels
        )


if __name__ == "__main__":

    main()