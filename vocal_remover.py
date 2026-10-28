#!/usr/bin/env python3
"""
classic-vocal-remover
=====================

A small, dependency-light implementation of the classic "center channel
cancellation" (a.k.a. phase-cancellation / karaoke) trick.

Most commercial stereo mixes pan the lead vocal to the center, which means the
vocal is (almost) identical in the left and right channels. Subtracting one
channel from the other cancels whatever is identical in both, which removes a
large part of the lead vocal -- but also the kick, bass and anything else that
is panned to the center.

This tool improves on the naive ``L - R`` approach by:

* keeping the low end: everything below ``--bass-cutoff`` Hz in the *mid*
  signal is added back, so kick and bass survive;
* optionally limiting cancellation to a vocal band (``--band``), so center
  panned cymbals and highs are less affected;
* offering an ``isolate`` mode that keeps the center instead of removing it
  (a rough "acapella" estimate);
* peak-safe output so the result never clips.

It is fast, deterministic and runs anywhere NumPy runs, but it is a 1970s
technique: it cannot separate a vocal that is panned off-center, recorded in
stereo, or drenched in stereo reverb. For real stem separation use an AI
separator.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

__version__ = "1.0.0"


@dataclass
class Options:
    mode: str = "remove"          # "remove" or "isolate"
    bass_cutoff: float = 120.0    # Hz, 0 disables bass preservation
    band: tuple[float, float] | None = None  # (low, high) Hz vocal band
    strength: float = 1.0         # 0..1, how much center to cancel
    normalize: bool = True


def _lowpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    sos = butter(4, cutoff, btype="lowpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def _bandpass(x: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    high = min(high, sr / 2 * 0.99)
    sos = butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def process(audio: np.ndarray, sr: int, opts: Options = Options()) -> np.ndarray:
    """Process a (frames, 2) float array and return a (frames, 2) array."""
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("Center cancellation needs a stereo (2-channel) input.")
    if not 0.0 <= opts.strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")

    left, right = audio[:, 0], audio[:, 1]
    mid = (left + right) / 2.0      # what is common to both channels
    side = (left - right) / 2.0     # what differs between channels

    # The part of the mid signal we consider "vocal".
    center = mid
    if opts.band is not None:
        center = _bandpass(mid, sr, *opts.band)

    if opts.mode == "remove":
        new_mid = mid - opts.strength * center
        if opts.bass_cutoff and opts.bass_cutoff > 0:
            # Put the low end of the cancelled part back (kick + bass).
            new_mid = new_mid + opts.strength * _lowpass(center, sr, opts.bass_cutoff)
        out = np.stack([new_mid + side, new_mid - side], axis=1)
    elif opts.mode == "isolate":
        # Keep the center estimate and drop the sides.
        iso = center
        if opts.bass_cutoff and opts.bass_cutoff > 0:
            iso = iso - _lowpass(iso, sr, opts.bass_cutoff)
        out = np.stack([iso, iso], axis=1)
    else:
        raise ValueError(f"unknown mode: {opts.mode!r}")

    if opts.normalize:
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.99:
            out = out * (0.99 / peak)
    return out.astype(np.float32)


def process_file(src: Path, dst: Path, opts: Options) -> None:
    audio, sr = sf.read(str(src), always_2d=True, dtype="float32")
    if audio.shape[1] == 1:
        raise SystemExit(f"{src}: file is mono -- there is no stereo image to cancel.")
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    out = process(audio, sr, opts)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), out, sr, subtype="PCM_24" if dst.suffix.lower() == ".wav" else None)


def _parse_band(text: str) -> tuple[float, float]:
    try:
        low, high = (float(v) for v in text.split("-"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("band must look like 200-6000") from exc
    if not 0 < low < high:
        raise argparse.ArgumentTypeError("band must satisfy 0 < low < high")
    return low, high


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vocal_remover",
        description="Remove or isolate center-panned vocals with classic phase cancellation.",
    )
    p.add_argument("inputs", nargs="+", type=Path, help="stereo audio files (WAV, FLAC, OGG, AIFF)")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("output"), help="output folder (default: ./output)")
    p.add_argument("-m", "--mode", choices=["remove", "isolate"], default="remove",
                   help="remove = karaoke/instrumental, isolate = rough center (vocal) estimate")
    p.add_argument("--bass-cutoff", type=float, default=120.0,
                   help="keep everything below this frequency (Hz); 0 disables (default: 120)")
    p.add_argument("--band", type=_parse_band, default=None, metavar="LOW-HIGH",
                   help="only cancel the center inside this band, e.g. 200-6000")
    p.add_argument("--strength", type=float, default=1.0, help="0..1, amount of cancellation (default: 1)")
    p.add_argument("--no-normalize", action="store_true", help="do not apply peak protection")
    p.add_argument("--format", choices=["wav", "flac"], default="wav", help="output format (default: wav)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    opts = Options(mode=args.mode, bass_cutoff=args.bass_cutoff, band=args.band,
                   strength=args.strength, normalize=not args.no_normalize)
    suffix = "instrumental" if args.mode == "remove" else "center"
    for src in args.inputs:
        if not src.is_file():
            print(f"skip: {src} (not found)", file=sys.stderr)
            continue
        dst = args.out_dir / f"{src.stem}_{suffix}.{args.format}"
        process_file(src, dst, opts)
        print(f"{src} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
