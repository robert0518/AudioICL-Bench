import json
import random
import re
from pathlib import Path
from collections import defaultdict


# ---------- AudioCount parsing ----------
COUNT_RE = re.compile(r"^(?P<stem>.+)_(?P<count>[1-6])$")


def parse_stem_and_count(wav_path: Path):
    """
    Expect filename like: <stem>_<count>.wav where count in 1..6
    Return (stem, count_int) or None if not match.
    """
    m = COUNT_RE.match(wav_path.stem)
    if not m:
        return None
    return m.group("stem"), int(m.group("count"))


def collect_audiocount_tasks(
    synth_dir: Path,
    demos_k: int,
    rng: random.Random,
    use_relative_paths: bool,
):
    wavs = sorted(synth_dir.glob("*.wav"))
    if not wavs:
        raise RuntimeError(f"[AudioCount] No wav files found in: {synth_dir}")

    by_stem = defaultdict(dict)  # stem -> {count -> path}
    for p in wavs:
        parsed = parse_stem_and_count(p)
        if not parsed:
            continue
        stem, count = parsed
        by_stem[stem][count] = p

    eligible = []
    for stem, cmap in by_stem.items():
        if len(cmap) >= (demos_k + 1):
            eligible.append(stem)

    if not eligible:
        raise RuntimeError("[AudioCount] No eligible stems found. Need filenames like <stem>_<1..6>.wav")

    def fmt_path(p: Path) -> str:
        return str(p.relative_to(synth_dir)) if use_relative_paths else str(p)

    tasks = []
    for stem in sorted(eligible):
        cmap = by_stem[stem]
        available_counts = sorted(cmap.keys())

        chosen_counts = rng.sample(available_counts, demos_k + 1)
        demos_counts = chosen_counts[:demos_k]
        query_count = chosen_counts[-1]

        demos = [{"audio_path": fmt_path(cmap[c]), "label": str(c)} for c in demos_counts]
        query = {"audio_path": fmt_path(cmap[query_count]), "label": str(query_count)}

        tasks.append({"task": "AudioCount", "demos": demos, "query": query})

    return tasks


# ---------- AudioOperator parsing ----------
OP_RE = re.compile(
    r"^sfx\-(?P<sfxid>.+?)__"
    r"(?P<a>[1-9])(?P<opword>add|sub|mul)(?P<b>[1-9])__"
    r"ans\-(?P<label>\d+)__k\-(?P<k>\d{2})$"
)


def parse_audiooperator_filename(wav_path: Path):
    m = OP_RE.match(wav_path.stem)
    if not m:
        return None
    sfxid = m.group("sfxid")
    label = int(m.group("label"))
    k = int(m.group("k"))
    return sfxid, label, k


def collect_audiooperator_tasks(
    synth_dir: Path,
    demos_k: int,
    rng: random.Random,
    use_relative_paths: bool,
):
    wavs = sorted(synth_dir.glob("*.wav"))
    if not wavs:
        raise RuntimeError(f"[AudioOperator] No wav files found in: {synth_dir}")

    by_sfxid = defaultdict(dict)  # sfxid -> {k -> (path, label)}
    skipped = 0
    for p in wavs:
        parsed = parse_audiooperator_filename(p)
        if not parsed:
            skipped += 1
            continue
        sfxid, label, k = parsed
        by_sfxid[sfxid][k] = (p, label)

    eligible = []
    for sfxid, km in by_sfxid.items():
        if len(km) >= (demos_k + 1):
            eligible.append(sfxid)

    if not eligible:
        raise RuntimeError(
            "[AudioOperator] No eligible sfx groups found. "
            "Need filenames like sfx-<id>__<a><add|sub|mul><b>__ans-<label>__k-00.wav"
        )

    def fmt_path(p: Path) -> str:
        return str(p.relative_to(synth_dir)) if use_relative_paths else str(p)

    tasks = []
    for sfxid in sorted(eligible):
        km = by_sfxid[sfxid]
        available_ks = sorted(km.keys())

        chosen_ks = rng.sample(available_ks, demos_k + 1)
        demos_ks = chosen_ks[:demos_k]
        query_k = chosen_ks[-1]

        demos = [{"audio_path": fmt_path(km[k][0]), "label": str(km[k][1])} for k in demos_ks]
        query = {"audio_path": fmt_path(km[query_k][0]), "label": str(km[query_k][1])}

        tasks.append({"task": "AudioOperator", "demos": demos, "query": query})

    if skipped:
        print(f"[AudioOperator] Skipped {skipped} wav files (filename pattern not matched).")

    return tasks


# ---------- Unified writer ----------
def main(
    audiocount_dir: str,
    audiooperator_dir: str,
    out_jsonl: str,
    demos_k: int = 2,
    seed: int = 42,
    id_prefix: str = "AudioICL",
    start_index: int = 1,
    use_relative_paths: bool = True,
    order: tuple[str, str] = ("AudioCount", "AudioOperator"),
):
    rng = random.Random(seed)

    audiocount_dir = Path(audiocount_dir).resolve()
    audiooperator_dir = Path(audiooperator_dir).resolve()
    out_path = Path(out_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect tasks
    tasks_by_name = {}

    if "AudioCount" in order:
        tasks_by_name["AudioCount"] = collect_audiocount_tasks(
            synth_dir=audiocount_dir,
            demos_k=demos_k,
            rng=rng,
            use_relative_paths=use_relative_paths,
        )

    if "AudioOperator" in order:
        tasks_by_name["AudioOperator"] = collect_audiooperator_tasks(
            synth_dir=audiooperator_dir,
            demos_k=demos_k,
            rng=rng,
            use_relative_paths=use_relative_paths,
        )

    # Flatten in requested order (deterministic)
    all_tasks = []
    for name in order:
        if name in tasks_by_name:
            all_tasks.extend(tasks_by_name[name])

    # Write JSONL with globally unique IDs
    idx = start_index
    with out_path.open("w", encoding="utf-8") as f:
        for t in all_tasks:
            record = {
                "id": f"{id_prefix}{idx:05d}",
                "task": t["task"],
                "demos": t["demos"],
                "query": t["query"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            idx += 1

    print(f"Done. Wrote {idx - start_index} tasks to: {out_path}")
    print(f"Counts: AudioCount={len(tasks_by_name.get('AudioCount', []))}, "
          f"AudioOperator={len(tasks_by_name.get('AudioOperator', []))}")


if __name__ == "__main__":
    AUDIOCOUNT_DIR = "/home5/b10303106/ICL-benchmark/data/audiocount"
    AUDIOOPERATOR_DIR = "/home5/b10303106/ICL-benchmark/data/audiooperator"
    OUT_JSONL = "/home5/b10303106/ICL-benchmark/data/meta/meta.jsonl"

    main(
        audiocount_dir=AUDIOCOUNT_DIR,
        audiooperator_dir=AUDIOOPERATOR_DIR,
        out_jsonl=OUT_JSONL,
        demos_k=2,
        seed=42,
        id_prefix="AudioICL",
        start_index=1,
        use_relative_paths=True,
        order=("AudioCount", "AudioOperator"),  # 控制輸出順序 -> 決定 id 分配
    )
