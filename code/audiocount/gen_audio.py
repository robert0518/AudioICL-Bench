import json
import random
import re
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
    HAVE_SF = True
except ImportError:
    import wave
    HAVE_SF = False


def sanitize_stem(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^\w\-]+", "", s, flags=re.UNICODE)
    return s or "sound"


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    if HAVE_SF:
        audio, sr = sf.read(str(path), always_2d=True)
        audio = audio.mean(axis=1).astype(np.float32, copy=False)
        return audio, sr

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported sample width: {sampwidth} bytes")

    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)

    return x, sr


def write_wav(path: Path, audio: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

    if HAVE_SF:
        sf.write(str(path), audio, sr)
        return

    import wave
    pcm = (audio * 32767.0).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)


def place_non_overlapping(total_len: int, clip_len: int, n: int, rng: random.Random) -> list[int]:
    assert clip_len <= total_len
    intervals: list[tuple[int, int]] = []

    max_tries = 10000
    tries = 0
    while len(intervals) < n and tries < max_tries:
        tries += 1
        start = rng.randint(0, total_len - clip_len)
        end = start + clip_len
        if all(end <= s or start >= e for s, e in intervals):
            intervals.append((start, end))

    if len(intervals) < n:
        # deterministic fallback packing
        intervals = []
        gap = (total_len - n * clip_len) // (n + 1) if total_len > n * clip_len else 0
        cursor = gap
        for _ in range(n):
            intervals.append((cursor, cursor + clip_len))
            cursor += clip_len + gap

    intervals.sort()
    return [s for s, _ in intervals]


def main(
    dataset_root: str,
    out_root: str,
    licenses_path: str = "licenses.json",
    sample_n: int = 300,
    repeats_min: int = 1,
    repeats_max: int = 6,
    target_seconds: float = 10.0,
    seed: int = 42,
):
    rng = random.Random(seed)

    dataset_root = Path(dataset_root).resolve()
    out_root = Path(out_root).resolve()

    audio_dir = dataset_root / "one_shot_percussive_sounds"
    licenses_file = dataset_root / licenses_path

    if not audio_dir.exists():
        raise FileNotFoundError(f"Missing folder: {audio_dir}")
    if not licenses_file.exists():
        raise FileNotFoundError(f"Missing licenses JSON: {licenses_file}")

    # Load licenses mapping: id(str) -> {name, username, license}
    with open(licenses_file, "r", encoding="utf-8") as f:
        licenses = json.load(f)

    # ✅ recursive: supports one_shot_percussive_sounds/1..5/<id>.wav
    wav_paths = sorted(audio_dir.rglob("*.wav"))
    if not wav_paths:
        raise RuntimeError(f"No wav files found in {audio_dir} (recursive)")

    # Map from id -> wav path
    id_to_path = {p.stem: p for p in wav_paths}

    # valid ids = in licenses + have wav file
    valid_ids = [sid for sid in licenses.keys() if sid in id_to_path]
    if len(valid_ids) < sample_n:
        raise RuntimeError(f"Not enough valid ids. Have {len(valid_ids)}, need {sample_n}")

    chosen_ids = rng.sample(valid_ids, sample_n)

    out_root.mkdir(parents=True, exist_ok=True)
    labels_csv = out_root / "labels.csv"

    rows = []
    for sid in chosen_ids:
        src_path = id_to_path[sid]
        meta = licenses[sid]
        orig_name = meta.get("name", f"{sid}.wav")
        stem = sanitize_stem(Path(orig_name).stem)

        audio, sr = load_wav_mono(src_path)

        total_len = int(round(sr * target_seconds))
        clip = audio[:total_len]  # if longer than 10s, cut; normally it's 1s
        clip_len = len(clip)

        # safe peak
        peak = float(np.max(np.abs(clip))) if clip_len else 0.0
        if peak > 0.999:
            clip = clip / peak * 0.999

        for r in range(repeats_min, repeats_max + 1):
            canvas = np.zeros(total_len, dtype=np.float32)
            starts = place_non_overlapping(total_len, clip_len, r, rng)

            for st in starts:
                canvas[st:st + clip_len] += clip

            out_path = out_root / f"{stem}_{r}.wav"
            write_wav(out_path, canvas, sr)

            rows.append({
                "output_file": out_path.name,
                "repeat_times": r,
                "source_id": sid,
                "source_wav": str(src_path.relative_to(dataset_root)),
                "original_name": orig_name,
                "username": meta.get("username", ""),
                "license": meta.get("license", ""),
                # 也可以把 onset 秒數寫進去，方便訓練/驗證
                "onsets_sec": ",".join([f"{st/sr:.6f}" for st in starts]),
            })

    import csv
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "output_file", "repeat_times", "source_id", "source_wav",
                "original_name", "username", "license", "onsets_sec"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Generated {len(rows)} files into: {out_root}")
    print(f"Labels CSV: {labels_csv}")


if __name__ == "__main__":
    DATASET_ROOT = "/mnt/data/robertchen/data/one_shot" 
    OUT_ROOT = "/mnt/data/robertchen/data/audiocount"
    LICENSES = "/mnt/data/robertchen/data/one_shot/licenses.txt"

    main(
        dataset_root=DATASET_ROOT,
        out_root=OUT_ROOT,
        licenses_path=LICENSES,
        sample_n=300,
        repeats_min=1,
        repeats_max=8,
        target_seconds=10.0,
        seed=42,
    )
