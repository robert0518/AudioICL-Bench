import csv
import random
from pathlib import Path
import numpy as np

try:
    import soundfile as sf
    HAVE_SF = True
except ImportError:
    HAVE_SF = False


# ---------- I/O ----------
def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load wav -> mono float32 [-1,1], sr."""
    if not HAVE_SF:
        raise RuntimeError("Please install soundfile: pip install soundfile")

    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32, copy=False)
    return audio, sr


def write_wav(path: Path, audio: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sf.write(str(path), audio, sr)


# ---------- DSP ----------
def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear resample for dataset gen (good enough for SFX)."""
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    if len(x) == 0:
        return x.astype(np.float32, copy=False)

    dur = (len(x) - 1) / float(sr_in)
    n_out = int(round(dur * sr_out)) + 1
    t_in = np.linspace(0.0, dur, num=len(x), endpoint=True, dtype=np.float32)
    t_out = np.linspace(0.0, dur, num=n_out, endpoint=True, dtype=np.float32)
    y = np.interp(t_out, t_in, x).astype(np.float32)
    return y


def peak_normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    if m <= 1e-9:
        return x.astype(np.float32, copy=False)
    return (x / m * peak).astype(np.float32)


def make_noise(total_len: int, rng: random.Random, noise_rms: float) -> np.ndarray:
    """Generate white noise with target RMS."""
    if total_len <= 0:
        return np.zeros(0, dtype=np.float32)
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    x = np_rng.normal(loc=0.0, scale=1.0, size=total_len).astype(np.float32)
    rms = float(np.sqrt(np.mean(x**2))) if total_len else 1.0
    if rms > 1e-9:
        x = x / rms * float(noise_rms)
    return x


# ---------- Single silence block ----------
def sample_single_silence_block(
    total_len: int,
    sr: int,
    rng: random.Random,
    min_silence_sec: float,
    max_silence_sec: float,
) -> tuple[int, int]:
    """
    Sample exactly ONE contiguous silence block (start,end) in samples.
    The block can be at front/middle/back depending on random placement.
    """
    if total_len <= 0:
        return (0, 0)

    L_sec = total_len / sr
    lo = max(0.0, min(min_silence_sec, L_sec))
    hi = max(lo, min(max_silence_sec, L_sec))
    sil_sec = rng.uniform(lo, hi)
    sil_len = int(round(sil_sec * sr))
    sil_len = max(1, min(sil_len, total_len))

    start = rng.randint(0, total_len - sil_len)
    end = start + sil_len
    return start, end


def build_sound_mask(total_len: int, sil: tuple[int, int]) -> np.ndarray:
    """True where sound is allowed, False where silence."""
    mask = np.ones(total_len, dtype=bool)
    s, e = sil
    if e > s:
        mask[s:e] = False
    return mask


def pick_start_in_sound(mask: np.ndarray, seg_len: int, rng: random.Random) -> int | None:
    """
    Pick a start index so that [st, st+seg_len) is fully inside sound region (mask True).
    Returns None if impossible.
    """
    total_len = len(mask)
    if seg_len <= 0 or seg_len > total_len:
        return None

    valid = np.convolve(mask.astype(np.int32), np.ones(seg_len, dtype=np.int32), mode="valid") == seg_len
    idxs = np.flatnonzero(valid)
    if idxs.size == 0:
        return None
    return int(idxs[rng.randrange(int(idxs.size))])


# ---------- main generation ----------
def main(
    oneshot_root: str,
    out_dir: str,
    target_sr: int = 16000,
    lengths_sec: range = range(1, 21),
    variants_per_length: int = 20,
    seed: int = 42,

    # 底噪：只會出現在「非空白」區
    noise_rms: float = 0.02,

    # one-shot 插入數量（每個檔案隨機）
    oneshot_count_min: int = 2,
    oneshot_count_max: int = 10,

    # one-shot 音量倍率（每次插入隨機）
    oneshot_gain_min: float = 0.8,
    oneshot_gain_max: float = 2.0,

    # ✅ 只要一段空白：空白長度範圍（秒）
    # 建議：讓空白「一小段」：例如 0.2s ~ min(1.5s, 30%*L)
    silence_min_sec: float = 0.2,
    silence_max_sec_cap: float = 1.5,   # 上限（避免太長）
    silence_max_ratio: float = 0.25,    # 也用比例限制（避免短檔被吃光）

    # 最終防爆音
    final_peak: float = 0.98,
):
    rng = random.Random(seed)
    oneshot_root = Path(oneshot_root).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    oneshot_paths = sorted(oneshot_root.rglob("*.wav"))
    if not oneshot_paths:
        raise RuntimeError(f"No wav files found under: {oneshot_root}")

    # preload
    oneshots: list[tuple[Path, np.ndarray]] = []
    for p in oneshot_paths:
        x, sr = load_wav_mono(p)
        x = resample_linear(x, sr, target_sr)
        x = peak_normalize(x, peak=0.9)
        if len(x) < int(0.01 * target_sr):  # <10ms skip
            continue
        oneshots.append((p, x))

    if not oneshots:
        raise RuntimeError("All one-shot wavs are too short or failed to load.")

    print(f"Loaded oneshots: {len(oneshots)}")

    labels_path = out_dir / "labels.csv"
    rows = []

    for L in lengths_sec:
        total_len = int(target_sr * L)
        L_sec = float(L)

        # 依據長度動態決定「空白最大秒數」
        silence_max_sec = min(float(silence_max_sec_cap), float(silence_max_ratio) * L_sec)
        silence_max_sec = max(silence_min_sec, silence_max_sec)  # 確保 >= min

        for idx in range(1, variants_per_length + 1):
            # 1) 抽「唯一一段」空白
            sil_s, sil_e = sample_single_silence_block(
                total_len=total_len,
                sr=target_sr,
                rng=rng,
                min_silence_sec=silence_min_sec,
                max_silence_sec=silence_max_sec,
            )
            sound_mask = build_sound_mask(total_len, (sil_s, sil_e))

            silence_samples = sil_e - sil_s
            sound_samples = total_len - silence_samples
            sound_sec = sound_samples / target_sr

            # 2) 生成 canvas：只有「非空白」區有白噪音底，空白段全 0
            canvas = np.zeros(total_len, dtype=np.float32)
            if sound_samples > 0 and noise_rms > 0:
                noise = make_noise(total_len, rng, noise_rms=noise_rms)
                canvas[sound_mask] = noise[sound_mask]

            # 3) 疊加 one-shots：只能放在非空白區
            n_oneshots = rng.randint(oneshot_count_min, oneshot_count_max)
            for _ in range(n_oneshots):
                _, clip = oneshots[rng.randrange(len(oneshots))]

                # 隨機截 seg
                if len(clip) > total_len:
                    start_in_clip = rng.randint(0, len(clip) - total_len)
                    seg = clip[start_in_clip:start_in_clip + total_len]
                else:
                    max_start = max(0, len(clip) - 1)
                    start_in_clip = rng.randint(0, max_start)
                    seg = clip[start_in_clip:]

                    # seg 至少 30ms
                    min_seg = int(0.03 * target_sr)
                    if len(seg) > min_seg:
                        seg_len = rng.randint(min_seg, len(seg))
                        seg = seg[:seg_len]

                seg_len = len(seg)
                if seg_len <= 0:
                    continue

                st = pick_start_in_sound(sound_mask, seg_len, rng)
                if st is None:
                    continue

                gain = rng.uniform(oneshot_gain_min, oneshot_gain_max)
                canvas[st:st + seg_len] += (seg * gain).astype(np.float32)

            # 4) 防爆音
            peak = float(np.max(np.abs(canvas))) if total_len else 0.0
            if peak > final_peak and peak > 1e-9:
                canvas = canvas / peak * final_peak

            out_path = out_dir / f"{L}_{idx}.wav"
            write_wav(out_path, canvas, target_sr)

            # labels
            rows.append({
                "file": out_path.name,
                "length_sec": f"{L_sec:.3f}",
                "sound_sec": f"{sound_sec:.3f}",              # ✅ 你的 label
                "silence_sec": f"{silence_samples/target_sr:.3f}",
                "silence_span_sec": f"{sil_s/target_sr:.3f}-{sil_e/target_sr:.3f}",
                "n_oneshots": str(n_oneshots),
            })

        print(f"Generated length={L}s, variants={variants_per_length}")

    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "length_sec", "sound_sec", "silence_sec", "silence_span_sec", "n_oneshots"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Output dir: {out_dir}")
    print(f"Labels CSV: {labels_path}")


if __name__ == "__main__":
    ONESHOT_ROOT = "/mnt/data/robertchen/data/one_shot/one_shot_percussive_sounds"
    OUT_DIR = "/mnt/data/robertchen/data/audiolength"

    main(
        oneshot_root=ONESHOT_ROOT,
        out_dir=OUT_DIR,
        target_sr=16000,
        lengths_sec=range(1, 21),
        variants_per_length=20,
        seed=42,

        noise_rms=0.02,
        oneshot_count_min=2,
        oneshot_count_max=10,
        oneshot_gain_min=0.8,
        oneshot_gain_max=2.0,
        final_peak=0.98,

        # ✅ 一段空白（小段）
        silence_min_sec=0.2,
        silence_max_sec_cap=1.5,
        silence_max_ratio=0.25,
    )
