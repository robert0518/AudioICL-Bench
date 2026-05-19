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


# -------------------------
# Basic audio I/O utilities
# -------------------------

def sanitize_stem(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^\w\-]+", "", s, flags=re.UNICODE)
    return s or "sound"


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load wav (or any format supported by soundfile) as mono float32."""
    if HAVE_SF:
        audio, sr = sf.read(str(path), always_2d=True)
        audio = audio.mean(axis=1).astype(np.float32, copy=False)
        return audio, sr

    # wave fallback supports wav only
    import wave
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
    """Write mono wav float32 to file."""
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


def safe_peak_normalize(x: np.ndarray, peak: float = 0.999) -> np.ndarray:
    """If peak too large, scale down to `peak`."""
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    m = float(np.max(np.abs(x)))
    if m > peak and m > 0:
        return (x / m * peak).astype(np.float32)
    return x.astype(np.float32, copy=False)


# -------------------------
# Placement (your original)
# -------------------------

def place_non_overlapping(total_len: int, clip_len: int, n: int, rng: random.Random) -> list[int]:
    """Return n start indices for clip placement with no overlap."""
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


# -------------------------
# Background (ESC-50 Animal) helpers
# -------------------------

# ESC-50 類別 0~10 為動物叫聲：
# 0=dog, 1=rooster, 2=pig, 3=cow, 4=frog,
# 5=cat, 6=hen, 7=insects, 8=sheep, 9=crow, 10=rain (非動物)
# 若只要純動物可改成 range(10)，不含 rain
ESC50_ANIMAL_CATEGORIES = set(range(11))  # 0~10


def load_esc50_animal_paths(bg_root: Path) -> list[Path]:
    """
    從 ESC-50 的 audio/ 資料夾中，只撈動物類別（類別 0~10）的音檔。
    ESC-50 檔名格式：{fold}-{clip_id}-{take}-{category}.wav
    """
    all_paths = sorted(bg_root.rglob("*.wav"))
    animal_paths = []
    for p in all_paths:
        parts = p.stem.split("-")
        if len(parts) < 4:
            continue  # 格式不符，跳過
        try:
            category_id = int(parts[-1])
        except ValueError:
            continue
        if category_id in ESC50_ANIMAL_CATEGORIES:
            animal_paths.append(p)
    return animal_paths


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Simple linear resample using numpy interpolation (no scipy dependency)."""
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    if len(x) < 2:
        return np.zeros(0, dtype=np.float32)

    dur = len(x) / sr_in
    n_out = int(round(dur * sr_out))
    if n_out < 2:
        return np.zeros(0, dtype=np.float32)

    t_in = np.linspace(0.0, dur, num=len(x), endpoint=False, dtype=np.float64)
    t_out = np.linspace(0.0, dur, num=n_out, endpoint=False, dtype=np.float64)
    y = np.interp(t_out, t_in, x.astype(np.float64)).astype(np.float32)
    return y


def crop_or_loop_to_length(x: np.ndarray, target_len: int, rng: random.Random) -> np.ndarray:
    """Make x length == target_len by random crop if longer, loop/tile if shorter."""
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)

    if len(x) == 0:
        return np.zeros(target_len, dtype=np.float32)

    if len(x) == target_len:
        return x.astype(np.float32, copy=False)

    if len(x) > target_len:
        start = rng.randint(0, len(x) - target_len)
        return x[start:start + target_len].astype(np.float32, copy=False)

    reps = int(np.ceil(target_len / len(x)))
    y = np.tile(x, reps)[:target_len]
    return y.astype(np.float32, copy=False)


def mix_background_snr_with_peak_cap(
    foreground: np.ndarray,
    background: np.ndarray,
    snr_db: float,
    peak_ratio_cap: float = 0.5,
    eps: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """
    Mix: foreground + alpha * background
    1) Choose alpha by RMS SNR target
    2) Cap alpha so background peaks won't cover one-shot peaks too much
    Returns (mixed, alpha).
    """
    fg_rms = float(np.sqrt(np.mean(foreground.astype(np.float64) ** 2) + eps))
    bg_rms = float(np.sqrt(np.mean(background.astype(np.float64) ** 2) + eps))
    if bg_rms <= 0:
        return foreground.copy(), 0.0

    target_ratio = 10 ** (snr_db / 20.0)
    alpha = (fg_rms / (bg_rms * target_ratio)) if fg_rms > 0 else (1.0 / target_ratio)

    fg_peak = float(np.max(np.abs(foreground))) if foreground.size else 0.0
    bg_peak = float(np.max(np.abs(background))) if background.size else 0.0
    if fg_peak > 0 and bg_peak > 0:
        max_alpha_by_peak = (peak_ratio_cap * fg_peak) / bg_peak
        alpha = min(alpha, max_alpha_by_peak)

    mixed = foreground + (background * alpha).astype(np.float32)
    return mixed.astype(np.float32, copy=False), float(alpha)


# -------------------------
# Main generation
# -------------------------

def main(
    dataset_root: str,
    out_root: str,
    licenses_path: str = "licenses.json",
    sample_n: int = 300,
    repeats_min: int = 1,
    repeats_max: int = 6,
    target_seconds: float = 10.0,
    seed: int = 42,

    # Background (ESC-50 Animal) settings
    animal_bg_root: str = "/mnt/data/robertchen/data/ESC-50/audio",
    snr_db_min: float = 18.0,
    snr_db_max: float = 32.0,
    bg_peak_ratio_cap: float = 0.5,
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

    with open(licenses_file, "r", encoding="utf-8") as f:
        licenses = json.load(f)

    wav_paths = sorted(audio_dir.rglob("*.wav"))
    if not wav_paths:
        raise RuntimeError(f"No wav files found in {audio_dir} (recursive)")

    id_to_path = {p.stem: p for p in wav_paths}
    valid_ids = [sid for sid in licenses.keys() if sid in id_to_path]
    if len(valid_ids) < sample_n:
        raise RuntimeError(f"Not enough valid ids. Have {len(valid_ids)}, need {sample_n}")

    # 載入 ESC-50 動物類別音檔
    animal_bg_root_p = Path(animal_bg_root).resolve()
    if not animal_bg_root_p.exists():
        raise FileNotFoundError(f"ESC-50 audio folder not found: {animal_bg_root_p}")

    bg_paths = load_esc50_animal_paths(animal_bg_root_p)
    if not bg_paths:
        raise RuntimeError(
            f"No ESC-50 animal wav files found in {animal_bg_root_p}. "
            f"請確認路徑正確且檔名格式為 {{fold}}-{{clip_id}}-{{take}}-{{category}}.wav"
        )

    print(f"ESC-50 animal bg files loaded: {len(bg_paths)}")

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
        clip = audio[:total_len]
        clip = safe_peak_normalize(clip, 0.999)
        clip_len = len(clip)

        for r in range(repeats_min, repeats_max + 1):
            # 1) build one-shot canvas
            canvas = np.zeros(total_len, dtype=np.float32)
            starts = place_non_overlapping(total_len, clip_len, r, rng)
            for st in starts:
                canvas[st:st + clip_len] += clip

            # 2) add ESC-50 animal background
            bg_path = rng.choice(bg_paths)
            bg_audio, bg_sr = load_wav_mono(bg_path)
            bg_audio = safe_peak_normalize(bg_audio, 0.999)

            if bg_sr != sr:
                bg_audio = resample_linear(bg_audio, bg_sr, sr)

            bg_audio = crop_or_loop_to_length(bg_audio, total_len, rng)

            snr_val = rng.uniform(snr_db_min, snr_db_max)
            canvas, alpha = mix_background_snr_with_peak_cap(
                foreground=canvas,
                background=bg_audio,
                snr_db=snr_val,
                peak_ratio_cap=bg_peak_ratio_cap,
            )

            # 3) final safety normalize
            canvas = safe_peak_normalize(canvas, 0.999)

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
                "onsets_sec": ",".join([f"{st/sr:.6f}" for st in starts]),
                "bg_wav": str(bg_path),
                "bg_snr_db": f"{snr_val:.3f}",
                "bg_alpha": f"{alpha:.8f}",
                "bg_peak_ratio_cap": f"{bg_peak_ratio_cap:.3f}",
            })

    import csv
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "output_file", "repeat_times", "source_id", "source_wav",
                "original_name", "username", "license", "onsets_sec",
                "bg_wav", "bg_snr_db", "bg_alpha", "bg_peak_ratio_cap"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Generated {len(rows)} files into: {out_root}")
    print(f"Labels CSV: {labels_csv}")


if __name__ == "__main__":
    DATASET_ROOT = "/mnt/data/robertchen/data/one_shot"
    OUT_ROOT = "/mnt/data/robertchen/data/audiocount/hard"
    LICENSES = "/mnt/data/robertchen/data/one_shot/licenses.txt"

    ANIMAL_ROOT = "/mnt/data/robertchen/data/ESC-50/audio"

    main(
        dataset_root=DATASET_ROOT,
        out_root=OUT_ROOT,
        licenses_path=LICENSES,
        sample_n=300,
        repeats_min=1,
        repeats_max=8,
        target_seconds=10.0,
        seed=42,

        animal_bg_root=ANIMAL_ROOT,
        snr_db_min=10.0,
        snr_db_max=15.0,
        bg_peak_ratio_cap=0.5,
    )