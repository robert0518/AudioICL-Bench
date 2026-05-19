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


# ---------- Single silence block (integer seconds) ----------
def sample_single_silence_block_int_seconds(
    total_len: int,
    sr: int,
    rng: random.Random,
    silence_sec_min: int,
    silence_sec_max: int,
) -> tuple[int, int, int]:
    """
    Sample exactly ONE contiguous silence block with INTEGER seconds length.
    Returns (start_sample, end_sample, silence_sec_int).
    """
    if total_len <= 0:
        return (0, 0, 0)

    total_sec = total_len // sr  # should be L if total_len == sr*L
    # clamp range to [0, total_sec]
    lo = max(0, min(int(silence_sec_min), int(total_sec)))
    hi = max(lo, min(int(silence_sec_max), int(total_sec)))

    silence_sec = rng.randint(lo, hi) if hi >= lo else lo
    silence_len = int(silence_sec * sr)

    # if silence is 0 sec, no silence block
    if silence_len <= 0:
        return (0, 0, 0)

    start = rng.randint(0, total_len - silence_len)
    end = start + silence_len
    return start, end, int(silence_sec)


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

    # ✅ 空白長度用「整數秒」抽樣（不使用趴數）
    silence_sec_min: int = 0,   # 允許 0 秒空白（如果你一定要有空白就改成 1）
    silence_sec_max: int = 2,   # 每段空白最長幾秒（會被 L 自動 clamp）

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

        # clamp silence max to <= L
        sil_max = min(int(silence_sec_max), int(L))
        sil_min = min(int(silence_sec_min), sil_max)

        for idx in range(1, variants_per_length + 1):
            # 1) 抽「唯一一段」空白（整數秒）
            sil_s, sil_e, sil_sec = sample_single_silence_block_int_seconds(
                total_len=total_len,
                sr=target_sr,
                rng=rng,
                silence_sec_min=sil_min,
                silence_sec_max=sil_max,
            )
            sound_mask = build_sound_mask(total_len, (sil_s, sil_e))

            sound_sec = int(L) - int(sil_sec)  # ✅ 整數秒 label
            # 防呆：避免全空白導致沒有可放 one-shot 的地方
            # 若你允許 sound_sec=0，下面流程仍可跑（只會輸出全 0 或只有噪音=0）
            # 若你「一定要有聲音」，可把 silence_sec_max 設為 L-1 或 silence_sec_min>=0 且 max<=L-1

            # 2) 生成 canvas：只有「非空白」區有白噪音底，空白段全 0
            canvas = np.zeros(total_len, dtype=np.float32)
            if sound_sec > 0 and noise_rms > 0:
                noise = make_noise(total_len, rng, noise_rms=noise_rms)
                canvas[sound_mask] = noise[sound_mask]

            # 3) 疊加 one-shots：只能放在非空白區
            n_oneshots = rng.randint(oneshot_count_min, oneshot_count_max) if sound_sec > 0 else 0
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

            # ✅ 檔名：f"有聲秒數_空白秒數_第幾個"
            out_name = f"{sound_sec}_{sil_sec}_{idx}.wav"
            out_path = out_dir / out_name
            write_wav(out_path, canvas, target_sr)

            # labels
            rows.append({
                "file": out_name,
                "total_sec": str(int(L)),
                "sound_sec": str(int(sound_sec)),
                "silence_sec": str(int(sil_sec)),
                "silence_span_sec": f"{sil_s/target_sr:.3f}-{sil_e/target_sr:.3f}" if sil_sec > 0 else "",
                "n_oneshots": str(n_oneshots),
            })

        print(f"Generated length={L}s, variants={variants_per_length}")

    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "total_sec", "sound_sec", "silence_sec", "silence_span_sec", "n_oneshots"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Output dir: {out_dir}")
    print(f"Labels CSV: {labels_path}")


if __name__ == "__main__":
    ONESHOT_ROOT = "/mnt/data/robertchen/data/one_shot/one_shot_percussive_sounds"
    OUT_DIR = "/mnt/data/robertchen/data/audiolength/hard"

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

        # ✅ 整數秒空白（你想要的）
        silence_sec_min=0,  # 若「一定要有空白」改成 1
        silence_sec_max=10,  # 例如 0~2 秒空白；會自動不超過該檔總長 L
    )
