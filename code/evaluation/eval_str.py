#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Any, Optional

# ===== Task rules (your spec) =====
INT_TASKS = {
    "AudioCount", "AudioCountHard",
    "AudioOperator", "AudioOperatorHard",
    "AudioLength", "AudioLengthHard"
}
AD_TASK = "AnomalyDetect"   # Normal/Anomalous
AR_TASK = "AudioRemap"      # fixed-length sequence, e.g. EGE
MORSE_TASK = "MorseCode"    # strict string match

# ----- parsing helpers -----
_INT_RE = re.compile(r"[-+]?\d+")
_WS_PUNCT_RE = re.compile(r"[\s\.\,\;\:\!\?\(\)\[\]\{\}\"\']+")
_AR_LETTERS_RE = re.compile(r"[A-Za-z]")

def normalize_strict(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.strip().lower()
    s = _WS_PUNCT_RE.sub("", s)
    return s

def extract_last_int(s: Any) -> Optional[int]:
    s = "" if s is None else str(s)
    matches = _INT_RE.findall(s)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except Exception:
        return None

def parse_ad_label(s: Any) -> Optional[str]:
    s2 = normalize_strict(s)
    mapping = {
        "normal": "normal",
        "anomalous": "anomalous",
        "anomaly": "anomalous",
        "abnormal": "anomalous",
    }
    if s2 in mapping:
        return mapping[s2]
    # common model variants
    if s2 in ("true", "1"):
        return "anomalous"
    if s2 in ("false", "0"):
        return "normal"
    # conservative contains
    if "anomal" in s2 or "abnorm" in s2:
        return "anomalous"
    if "normal" in s2:
        return "normal"
    return None

def parse_ar_seq(s: Any, expected_len: int) -> Optional[str]:
    if s is None:
        return None
    letters = _AR_LETTERS_RE.findall(str(s))
    if not letters:
        return None
    seq = "".join(letters).upper()
    if len(seq) != expected_len:
        return None
    return seq

def is_correct(task: str, gt: Any, pred: Any, ar_len: int) -> bool:
    if pred is None:
        return False

    if task in INT_TASKS:
        gt_i = gt if isinstance(gt, int) else extract_last_int(gt)
        pr_i = extract_last_int(pred)
        return (gt_i is not None) and (pr_i is not None) and (gt_i == pr_i)

    if task == AD_TASK:
        gt_b = parse_ad_label(gt)
        pr_b = parse_ad_label(pred)
        return (gt_b is not None) and (pr_b is not None) and (gt_b == pr_b)

    if task == AR_TASK:
        gt_s = parse_ar_seq(gt, ar_len)
        pr_s = parse_ar_seq(pred, ar_len)
        return (gt_s is not None) and (pr_s is not None) and (gt_s == pr_s)

    if task == MORSE_TASK:
        return normalize_strict(gt) == normalize_strict(pred)

    # fallback: strict normalized match
    return normalize_strict(gt) == normalize_strict(pred)

# ----- filename parser: model & shots -----
# Matches:
#   {model}_results_k3.jsonl
#   {model}_results_meta_k3.jsonl
FNAME_RE = re.compile(r"^(?P<model>.+?)_results(?:_meta)?_k(?P<k>\d+)\.jsonl$")

def parse_model_shots(filename: str):
    m = FNAME_RE.match(filename)
    if not m:
        return None, None
    return m.group("model"), int(m.group("k"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default="/mnt/data/robertchen/AudioICL/result/ablation_audiocount",
                    help="Directory containing *_results*_k*.jsonl files")
    ap.add_argument("--out", dest="out",
                    default="/mnt/data/robertchen/AudioICL/code/evaluation/result/abla/audiocount.csv",
                    help="Output CSV path (model,shots,task,accuracy)")
    ap.add_argument("--pred-key", default="response", help="Key for prediction text")
    ap.add_argument("--gt-key", default="gt", help="Key for ground truth")
    ap.add_argument("--task-key", default="task", help="Key for task name")
    ap.add_argument("--id-key", default="id", help="Key for episode id (default: id)")
    ap.add_argument("--ar-len", type=int, default=3, help="AR sequence length (default 3)")

    # NEW: per-example outputs (per model+shots)
    ap.add_argument("--per-example-dir", default="/mnt/data/robertchen/AudioICL/code/evaluation/string/abla/audiocount",
                    help="Optional: output per-example CSVs per model+shots under this directory.")
    ap.add_argument("--prompt-style", default="specific",
                    help="Prompt style label to record in output, e.g. Specific/General/None.")
    ap.add_argument("--only-model", default="",
                    help="If set, only evaluate files whose parsed model matches this string.")
    ap.add_argument("--only-shots", default="",
                    help="If set, only evaluate specific shots, e.g. ''.")
    ap.add_argument("--keep-text", action="store_true",
                    help="If set, include gt/pred in per-example outputs (can be large).")

    args = ap.parse_args()

    indir = Path(args.inp)
    if not indir.is_dir():
        raise ValueError(f"--in must be a directory, got: {indir}")

    only_shots = None
    if args.only_shots.strip():
        only_shots = set(int(x) for x in args.only_shots.split(",") if x.strip())

    files = sorted(indir.rglob("*.jsonl"))
    files = [p for p in files if FNAME_RE.match(p.name)]
    if not files:
        raise FileNotFoundError("No matching *_results*_k*.jsonl files found.")

    # counts[(model, shots, task)] = [correct, total]
    counts = defaultdict(lambda: [0, 0])

    # per_ex[(prompt_style, model, shots)] = list of rows [id, task, correct (, gt, pred)]
    per_ex = defaultdict(list)

    for fp in files:
        model, shots = parse_model_shots(fp.name)
        if model is None:
            continue

        # optional filters
        if args.only_model and model != args.only_model:
            continue
        if only_shots is not None and shots not in only_shots:
            continue

        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)

                task = str(r.get(args.task_key, ""))
                gt = r.get(args.gt_key)
                pred = r.get(args.pred_key)
                ex_id = r.get(args.id_key)

                key = (model, shots, task)
                counts[key][1] += 1
                corr = is_correct(task, gt, pred, args.ar_len)
                if corr:
                    counts[key][0] += 1

                # per-example record
                if args.per_example_dir:
                    pstyle = args.prompt_style if args.prompt_style else "UNKNOWN"
                    row = [ex_id, task, int(corr)]
                    if args.keep_text:
                        row += [gt if gt is not None else "", pred if pred is not None else ""]
                    per_ex[(pstyle, model, shots)].append(row)

    # write summary CSV
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "shots", "task", "accuracy", "n"])
        for (model, shots, task) in sorted(counts.keys(), key=lambda x: (x[0], x[1], x[2])):
            c, n = counts[(model, shots, task)]
            acc = (c / n) if n else 0.0
            w.writerow([model, shots, task, f"{acc:.6f}", n])

    # print(f"Wrote summary: {outp}  ({len(counts)} rows)")

    # write per-example CSVs per (prompt_style, model, shots)
    if args.per_example_dir:
        base = Path(args.per_example_dir)
        for (pstyle, model, shots), rows in per_ex.items():
            out_dir = base / pstyle / model
            out_dir.mkdir(parents=True, exist_ok=True)
            pe_path = out_dir / f"k{shots}.csv"
            with pe_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                header = ["id", "task", "correct"]
                if args.keep_text:
                    header += ["gt", "pred"]
                w.writerow(header)
                w.writerows(rows)
            print(f"Wrote per-example: {pe_path} ({len(rows)} rows)")

if __name__ == "__main__":
    main()