#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf

try:
    import librosa
except Exception:
    librosa = None


# ----------------------------
# Audio I/O
# ----------------------------
def load_wav(path: Path, target_sr: Optional[int]) -> Tuple[np.ndarray, int]:
    """Load wav as float32 mono, optionally resample to target_sr."""
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32)

    if target_sr is not None and sr != target_sr:
        if librosa is None:
            raise RuntimeError(
                "Resampling requested but librosa is not installed. "
                "Install librosa or set --target_sr -1 to disable resampling."
            )
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return x, sr


def save_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    sf.write(str(path), x, sr)


def silence(sr: int, ms: int) -> np.ndarray:
    n = int(sr * (ms / 1000.0))
    return np.zeros((max(n, 0),), dtype=np.float32)


# ----------------------------
# Pool scanning
# ----------------------------
def list_wavs_recursive(root: Path, exclude_mix_prefix: str = "MIX__") -> List[Path]:
    """Recursively list .wav files under root, excluding files that look like MIX."""
    wavs = []
    for p in root.rglob("*.wav"):
        if p.name.startswith(exclude_mix_prefix):
            continue
        wavs.append(p)
    wavs = sorted(wavs)
    if not wavs:
        raise FileNotFoundError(
            f"No usable .wav found under: {root}\n"
            f"(Note: files starting with '{exclude_mix_prefix}' are excluded)"
        )
    return wavs


# ----------------------------
# Pattern + synthesis
# ----------------------------
def parse_patterns(s: str) -> List[str]:
    pats = [x.strip().upper() for x in s.split(",") if x.strip()]
    if not pats:
        raise ValueError("Empty --patterns")
    for p in pats:
        if len(p) != 3 or any(ch not in "AB" for ch in p):
            raise ValueError(f"Invalid pattern: {p} (must be 3 chars from A/B, e.g. AAB, ABA, BAB)")
    return pats


def synthesize(pattern: str, a: np.ndarray, b: np.ndarray, sr: int, gap_ms: int) -> np.ndarray:
    """Concatenate audio segments by 3-char pattern, insert silence gap between segments."""
    pat = pattern.upper().strip()
    if len(pat) != 3:
        raise ValueError(f"Pattern must have length 3, got: {pattern}")

    pieces: List[np.ndarray] = []
    gap = silence(sr, gap_ms)

    for i, ch in enumerate(pat):
        if ch == "A":
            pieces.append(a)
        elif ch == "B":
            pieces.append(b)
        else:
            raise ValueError(f"Unsupported char {ch} in pattern={pattern}")

        if i != 2 and gap.size > 0:
            pieces.append(gap)

    return np.concatenate(pieces, axis=0)


def safe_stem(p: Path) -> str:
    return p.stem.replace(" ", "_")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pool_dir",
        type=str,
        default="/mnt/data/robertchen/data/one_shot/one_shot_percussive_sounds",
        help="Root directory containing ONE-SHOT wavs (scanned recursively).",
    )
    parser.add_argument("--out_dir", type=str, default="/mnt/data/robertchen/data/audioremap", help="Output directory for MIX wavs.")

    # CHANGED: now we sample AB pairs, and for each AB we generate ALL patterns
    parser.add_argument(
        "--num_ab_pairs",
        type=int,
        default=300,
        help="How many (A,B) pairs to sample. Each pair generates |patterns| MIX wavs.",
    )

    parser.add_argument(
        "--patterns",
        type=str,
        default="AAB,ABA,BAB,BAA,ABB,BBA",
        help="Comma-separated 3-char patterns, e.g. AAB,ABA,BAB",
    )

    parser.add_argument("--gap_ms", type=int, default=0, help="Silence gap between segments in ms.")
    parser.add_argument("--target_sr", type=int, default=16000, help="Resample to SR. Use -1 to disable.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_same_file", action="store_true", help="Allow A and B to be the same wav.")
    parser.add_argument("--exclude_mix_prefix", type=str, default="MIX__", help="Exclude pool files with this prefix.")
    parser.add_argument("--write_manifest", action="store_true", help="Write manifest.jsonl mapping A/B->MIX.")

    args = parser.parse_args()

    rng = random.Random(args.seed)
    pool_dir = Path(args.pool_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_sr: Optional[int] = None if args.target_sr == -1 else args.target_sr
    patterns = parse_patterns(args.patterns)

    pool = list_wavs_recursive(pool_dir, exclude_mix_prefix=args.exclude_mix_prefix)
    if len(pool) < 2 and not args.allow_same_file:
        raise ValueError("Pool must contain at least 2 wav files unless --allow_same_file is set.")

    total = args.num_ab_pairs * len(patterns)
    print(f"Pool size: {len(pool)} wavs")
    print(f"AB pairs: {args.num_ab_pairs}")
    print(f"Patterns: {patterns} (n={len(patterns)})")
    print(f"Total MIX to generate: {total}")

    manifest_path = out_dir / "manifest.jsonl"
    mf = manifest_path.open("w", encoding="utf-8") if args.write_manifest else None

    global_i = 0
    for ab_i in range(args.num_ab_pairs):
        # sample one AB pair
        a_path = rng.choice(pool)
        b_path = rng.choice(pool)
        if not args.allow_same_file:
            while b_path == a_path:
                b_path = rng.choice(pool)

        # load once per AB pair (efficient)
        a, sr_a = load_wav(a_path, target_sr)
        b, sr_b = load_wav(b_path, target_sr)
        if sr_a != sr_b:
            raise RuntimeError(f"SR mismatch after load: A={sr_a}, B={sr_b}")

        a_st = safe_stem(a_path)
        b_st = safe_stem(b_path)

        # generate MIX for ALL patterns
        for pattern_i, pattern in enumerate(patterns):
            mix = synthesize(pattern, a, b, sr_a, args.gap_ms)

            mix_name = (
                f"MIX__{pattern}__ab{ab_i:04d}.wav"
            )
            mix_path = out_dir / mix_name
            save_wav(mix_path, mix, sr_a)

            if mf is not None:
                mf.write(
                    json.dumps(
                        {
                            "global_i": global_i,
                            "ab_i": ab_i,
                            "pattern_i": pattern_i,
                            "pattern": pattern,
                            "gap_ms": args.gap_ms,
                            "sr": sr_a,
                            "A_path": str(a_path),
                            "B_path": str(b_path),
                            "MIX_path": str(mix_path),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            global_i += 1

        if (ab_i + 1) % 25 == 0:
            print(f"[AB {ab_i+1}/{args.num_ab_pairs}] generated {len(patterns)} patterns (total {global_i}/{total})")

    if mf is not None:
        mf.close()
        print(f"Manifest written to: {manifest_path}")

    print("Done.")


if __name__ == "__main__":
    main()