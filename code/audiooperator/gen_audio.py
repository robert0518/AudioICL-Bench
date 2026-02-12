import csv
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

try:
    from scipy.signal import resample_poly
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


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

    return x.astype(np.float32, copy=False), sr


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


def peak_normalize(audio: np.ndarray, target_peak: float = 0.999) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak <= 1e-12:
        return audio.astype(np.float32, copy=False)
    if peak > target_peak:
        audio = audio / peak * target_peak
    return audio.astype(np.float32, copy=False)


def trim_silence(audio: np.ndarray, sr: int, top_db: float = 35.0, pad_ms: float = 10.0) -> np.ndarray:
    if len(audio) == 0:
        return audio

    mx = float(np.max(np.abs(audio)) + 1e-12)
    thr = mx * (10 ** (-top_db / 20.0))
    idx = np.where(np.abs(audio) >= thr)[0]
    if idx.size == 0:
        return audio

    pad = int(round(sr * pad_ms / 1000.0))
    st = max(0, int(idx[0]) - pad)
    ed = min(len(audio), int(idx[-1]) + 1 + pad)
    return audio[st:ed].astype(np.float32, copy=False)


def fade_in_out(audio: np.ndarray, sr: int, fade_ms: float = 3.0) -> np.ndarray:
    n = int(round(sr * fade_ms / 1000.0))
    if n <= 0 or len(audio) < 2 * n:
        return audio
    w = np.linspace(0.0, 1.0, n, dtype=np.float32)
    audio = audio.copy()
    audio[:n] *= w
    audio[-n:] *= w[::-1]
    return audio


def resample_audio(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32, copy=False)
    if not HAVE_SCIPY:
        raise RuntimeError("Need scipy for resampling (pip install scipy).")
    g = np.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    y = resample_poly(audio, up, down).astype(np.float32, copy=False)
    return y


def concat_with_gap(parts: list[np.ndarray], sr: int, gap_ms: float) -> np.ndarray:
    gap = np.zeros(int(round(sr * gap_ms / 1000.0)), dtype=np.float32)
    out = []
    for i, p in enumerate(parts):
        out.append(p)
        if i != len(parts) - 1:
            out.append(gap)
    return np.concatenate(out).astype(np.float32, copy=False)


DIGIT_RE = re.compile(r"^([0-9])_([A-Za-z]+)_(\d+)\.wav$", re.IGNORECASE)


def index_fsdd_jackson_only(fsdd_recordings_dir: Path) -> dict[int, list[Path]]:
    """
    Only keep files with speaker == 'jackson'.
    FSDD filenames: {digit}_{speaker}_{index}.wav
    """
    by_digit: dict[int, list[Path]] = {d: [] for d in range(1, 10)}
    wavs = sorted(fsdd_recordings_dir.glob("*.wav"))
    if not wavs:
        raise RuntimeError(f"No wav files found in {fsdd_recordings_dir}")

    for p in wavs:
        m = DIGIT_RE.match(p.name)
        if not m:
            continue
        d = int(m.group(1))
        speaker = m.group(2).lower()
        if speaker != "jackson":
            continue
        if 1 <= d <= 9:
            by_digit[d].append(p)

    missing = [d for d in range(1, 10) if len(by_digit[d]) == 0]
    if missing:
        raise RuntimeError(f"Missing jackson recordings for digits: {missing}")

    return by_digit


def list_sfx_wavs(one_shot_dir: Path) -> list[Path]:
    # ✅ recursive: supports one_shot_percussive_sounds/1..5/<id>.wav
    wavs = sorted(one_shot_dir.rglob("*.wav"))
    if not wavs:
        raise RuntimeError(f"No wav files found (recursive) under: {one_shot_dir}")
    return wavs


def stable_op_for_sfx(sfx_path: Path, ops: list[str], seed: int) -> str:
    key = f"{seed}::{sfx_path.as_posix()}"
    h = 0
    for ch in key:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return ops[h % len(ops)]


def compute_label(a: int, b: int, op: str) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    raise ValueError(f"Unknown op: {op}")


def make_triplet_audio(
    digit_a_path: Path,
    sfx_path: Path,
    digit_b_path: Path,
    target_sr: int,
    trim: bool,
    gap_ms: float,
    top_db: float,
    pad_ms: float,
    fade_ms: float,
) -> np.ndarray:
    da, sr_a = load_wav_mono(digit_a_path)
    sx, sr_s = load_wav_mono(sfx_path)
    db, sr_b = load_wav_mono(digit_b_path)

    da = resample_audio(da, sr_a, target_sr)
    sx = resample_audio(sx, sr_s, target_sr)
    db = resample_audio(db, sr_b, target_sr)

    if trim:
        da = trim_silence(da, target_sr, top_db=top_db, pad_ms=pad_ms)
        sx = trim_silence(sx, target_sr, top_db=top_db, pad_ms=pad_ms)
        db = trim_silence(db, target_sr, top_db=top_db, pad_ms=pad_ms)

    y = concat_with_gap([da, sx, db], target_sr, gap_ms=gap_ms)
    y = fade_in_out(y, target_sr, fade_ms=fade_ms)
    y = peak_normalize(y, target_peak=0.999)
    return y


def main(
    one_shot_dir: str,
    fsdd_recordings_dir: str,
    out_dir: str,
    n_sfx: int = 50,
    per_sfx: int = 8,
    target_sr: int = 16000,
    seed: int = 42,
    trim: bool = True,
    gap_ms: float = 30.0,
    top_db: float = 35.0,
    pad_ms: float = 10.0,
    fade_ms: float = 3.0,
):
    rng = random.Random(seed)

    one_shot_dir = Path(one_shot_dir).resolve()
    fsdd_recordings_dir = Path(fsdd_recordings_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ops = ["+", "-", "*"]
    op_name = {"+": "add", "-": "sub", "*": "mul"}

    # ✅ digits from jackson only
    by_digit = index_fsdd_jackson_only(fsdd_recordings_dir)

    # ✅ sfx recursive (1..5 subfolders ok)
    sfx_wavs = list_sfx_wavs(one_shot_dir)

    if n_sfx > len(sfx_wavs):
        chosen_sfx = sfx_wavs
    else:
        chosen_sfx = rng.sample(sfx_wavs, n_sfx)

    labels_csv = out_dir / "labels.csv"

    rows = []
    for sfx_path in chosen_sfx:
        op = stable_op_for_sfx(sfx_path, ops, seed=seed)
        sfx_id = sanitize_stem(sfx_path.stem)[:40] or "sfx"

        for k in range(per_sfx):
            # ✅ ensure non-negative label
            if op == "-":
                b = rng.randint(1, 9)
                a = rng.randint(b, 9)  # a >= b
            else:
                a = rng.randint(1, 9)
                b = rng.randint(1, 9)

            a_wav = rng.choice(by_digit[a])
            b_wav = rng.choice(by_digit[b])

            label = compute_label(a, b, op)

            fname = f"sfx-{sfx_id}__{a}{op_name[op]}{b}__ans-{label}__k-{k:02d}.wav"
            out_path = out_dir / fname

            y = make_triplet_audio(
                digit_a_path=a_wav,
                sfx_path=sfx_path,
                digit_b_path=b_wav,
                target_sr=target_sr,
                trim=trim,
                gap_ms=gap_ms,
                top_db=top_db,
                pad_ms=pad_ms,
                fade_ms=fade_ms,
            )

            write_wav(out_path, y, target_sr)

            rows.append({
                "output_file": out_path.name,
                "a": a,
                "b": b,
                "op": op,
                "label": label,
                "digit_a_wav": str(a_wav),
                "digit_b_wav": str(b_wav),
                "sfx_wav": str(sfx_path),
                "sfx_id": sfx_id,
                "k": k,
            })

    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "output_file",
                "a", "b", "op", "label",
                "digit_a_wav", "digit_b_wav",
                "sfx_wav", "sfx_id",
                "k",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Generated {len(rows)} files into: {out_dir}")
    print(f"Labels CSV: {labels_csv}")


if __name__ == "__main__":
    ONE_SHOT_DIR = "/home5/b10303106/ICL-benchmark/data/one_shot/one_shot_percussive_sounds"
    FSDD_DIR = "/home5/b10303106/ICL-benchmark/data/free-spoken-digit-dataset/recordings"
    OUT_DIR = "/home5/b10303106/ICL-benchmark/data/audiooperator"

    main(
        one_shot_dir=ONE_SHOT_DIR,
        fsdd_recordings_dir=FSDD_DIR,
        out_dir=OUT_DIR,
        n_sfx=300,
        per_sfx=8,
        target_sr=16000,
        seed=42,
        trim=True,
        gap_ms=25.0,
        top_db=35.0,
        pad_ms=10.0,
        fade_ms=3.0,
    )
