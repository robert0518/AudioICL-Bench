import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import json
import argparse
import gc
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from transformers import Qwen2_5OmniProcessor


# ============================================================
# Task Instructions
# ============================================================

TASK_INSTRUCTIONS = {
    "None": "",

    "GENERAL": """
The input is a sequence of audio clips. In the in-context examples, each demo clip is paired with a numeric label.

Your task is to infer the hidden rule/pattern that maps audio to the numeric label using only the provided examples.
Then apply the inferred rule to the query clip and output the predicted label as a single integer.

Output only the resulting integer.
""".strip(),

    "AudioCount": """
The input is a sequence of audio clips. In the in-context examples, each demo clip is paired with a numeric label.

Your task is to infer the hidden rule that determines how the audio content corresponds to the numeric label using only the provided examples. The rule is consistent within each example set but may vary across tasks.

After identifying the rule, apply it to the query clip and output the predicted label as a single integer. Output only the resulting integer.
""".strip(),

    "AudioCountHard": """
The input is a sequence of audio clips. In the in-context examples, each demo clip is paired with a numeric label.

Your task is to infer the hidden rule that determines how the audio content corresponds to the numeric label using only the provided examples. The rule is consistent within each example set but may vary across tasks.

After identifying the rule, apply it to the query clip and output the predicted label as a single integer. Output only the resulting integer.
""".strip(),

    "AudioLength": """
The input is a sequence of audio clips. In the in-context examples, each demo clip is paired with a numeric label.

Your task is to infer the hidden rule that determines how the audio corresponds to the numeric label using only the provided examples. The rule is consistent within each example set but may vary across tasks.

After identifying the rule, apply it to the query clip and output the predicted label as a single integer. Output only the resulting integer.
""".strip(),

    "AudioLengthHard": """
The input is a sequence of audio clips. In the in-context examples, each demo clip is paired with a numeric label.

Your task is to infer the hidden rule that determines how the audio corresponds to the numeric label using only the provided examples. The rule is consistent within each example set but may vary across tasks.

After identifying the rule, apply it to the query clip and output the predicted label as a single integer. Output only the resulting integer.
""".strip(),

    "AudioOperator": """
The input is a sequence of audio clips. Each clip contains numbers expressed through sound events, separated by a particular filler sound. The filler sound consistently represents a specific mathematical operation within the example set.

From the in-context examples, infer which arithmetic rule the filler sound represents and how the numbers should be combined. Then apply the same rule to the query clip and compute the final result.

Output only the resulting integer.
""".strip(),

    "AudioSum": """
The input is a sequence of audio clips. In the in-context examples, each demo clip is paired with a numeric label.

In each example set, there are exactly two distinct one-shot sound types. Each sound type corresponds to a fixed integer value (its label) within that example set. The query clip is a mixture containing multiple non-overlapping occurrences of those two sound types.

Your task is to infer which sound corresponds to which integer value from the demos, then compute the total sum for the query by adding the value of each occurrence in the mixture. Output only the resulting integer.
""".strip(),

    "AudioOperatorHard":"""
The input is a sequence of audio clips. Each clip contains numbers expressed through sound events, separated by particular filler sounds. The different filler sounds consistently represents specific mathematical operation respectively within the example set.

From the in-context examples, infer which arithmetic rule the filler sounds represents and how the numbers should be combined. Then apply the same rule to the query clip and compute the final result.

Output only the resulting integer.
""".strip(), 

    "MorseCode": """
You are given Morse code audio clips.
You will first receive a complete reference set of Morse-code letter audio examples paired with their correct uppercase letters (A–Z).
Use these reference examples to decode the query audio.
Then, use the additional in-context examples and decode the query audio.
Only output the word, without any punctuation, explanation. 
""".strip(),

    "AudioRemap": """
The input is a sequence of audio clips.

In each demo, you hear A, B, then MIX.
A and B are reference sounds with letter labels.

MIX consists of exactly THREE consecutive segments.
Each segment is either A or B.

You MUST listen to MIX and match each segment to A or B by audio similarity.
Determine the 3-step A/B order from the demos.

For the query, again hear A, B, then MIX.
Output ONLY the 3-letter uppercase MIX label without explanation. 
""".strip(),

    "AnomalyDetect": """
You will hear audio clips from the SAME object type (e.g., bearing, fan, gearbox, etc.).

In each in-context demo, you will hear two clips Normal clip and Anomalous clip. 

For the query, you will hear ONE clip from the same object type.
Decide whether the query clip is Normal or Anomalous.

Output ONLY the label, without any explanation. 
""".strip(),
}


# ============================================================
# Utils
# ============================================================

def cleanup_vllm():
    try:
        destroy_model_parallel()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if y.dtype != np.float32:
        y = y.astype(np.float32)
    return y


def pad_audio(y: np.ndarray, max_len: int) -> np.ndarray:
    if len(y) == max_len:
        return y
    if len(y) > max_len:
        return y[:max_len]
    out = np.zeros((max_len,), dtype=y.dtype)
    out[:len(y)] = y
    return out


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def get_done_ids(out_file: Path) -> set:
    done = set()
    if not out_file.exists():
        return done
    with out_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    return done


def parse_k_from_filename(p: Path) -> int:
    m = re.search(r"(?:^|[_\-])k(\d+)(?:[_\-]|$)", p.stem)
    if not m:
        m = re.search(r"k(\d+)", p.stem)
    if not m:
        raise ValueError(f"Cannot parse k from filename: {p.name} (expect *_k5.jsonl)")
    return int(m.group(1))


def collect_jsonl_files(meta_path: Path):
    if meta_path.is_file():
        return [meta_path]
    if meta_path.is_dir():
        return sorted(meta_path.glob("*.jsonl"))
    raise FileNotFoundError(f"meta_path not found: {meta_path}")


def normalize_answer_letters_only(text: str) -> str:
    if text is None:
        return ""
    t = text.strip().upper()
    t = re.sub(r"[^A-Z]", "", t)
    return t


def load_morse_az_bank(az_dir: Path, target_sr: int = 16000) -> Dict[str, np.ndarray]:
    bank: Dict[str, np.ndarray] = {}
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        p = az_dir / f"{ch}.wav"
        if not p.exists():
            raise FileNotFoundError(f"Missing Morse bank file: {p}")
        bank[ch] = load_audio(str(p), target_sr=target_sr)
    return bank


# ============================================================
# Prompt Builder (generic: supports multi-clip items)
# ============================================================

def build_prompt(
    task_name: str,
    demo_items: List[Dict],
    query_items: List[Dict],
    prepend_morse_az: bool = False,
):
    """
    Each item:
      {"tag": str, "label": Optional[str]}
    If label is None => leave blank for model to generate.
    """
    instruction = TASK_INSTRUCTIONS.get(task_name, TASK_INSTRUCTIONS["None"])
    A = "<|audio_bos|><|AUDIO|><|audio_eos|>"

    lines = ""

    # MorseCode optional A–Z bank
    if prepend_morse_az:
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            lines += f"[Letter {ch}] {A} Label: {ch}\n"

    for it in demo_items:
        tag = it["tag"]
        lb = it.get("label", "")
        lines += f"[{tag}] {A} Label: {lb}\n"

    for it in query_items:
        tag = it["tag"]
        lb = it.get("label", None)
        if lb is None:
            lines += f"[{tag}] {A} Label:"
        else:
            lines += f"[{tag}] {A} Label: {lb}\n"

    user_text = (instruction + "\n\n" if instruction else "") + lines

    messages = [
        {
            "role": "system",
            "content": (
                "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                "capable of perceiving auditory and visual inputs, as well as generating text and speech."
            ),
        },
        {"role": "user", "content": user_text},
    ]
    return messages


# ============================================================
# Generic flatten for any 'clips' meta
# ============================================================

def _has_clips_format(rec: Dict) -> bool:
    demos = rec.get("demos", [])
    query = rec.get("query", {})
    return (
        isinstance(demos, list)
        and len(demos) > 0
        and isinstance(demos[0], dict)
        and "clips" in demos[0]
        and isinstance(query, dict)
        and "clips" in query
    )


def _flatten_for_prompt(rec: Dict, n_demos: int) -> Tuple[List[Dict], List[Dict], List[str]]:
    """
    Returns:
      demo_items: [{"tag","label","audio_path"}]  (label always string for demos)
      query_items: [{"tag","label(optional)","audio_path"}]
      flat_audio_paths: [path...] in exact placeholder order
    Supports:
      - normal: demo dict has audio_path/label, query has audio_path
      - clips: demo dict has clips list, query has clips list
    """
    task = rec.get("task", "None")

    # -------- multi-clip path (generic) --------
    if _has_clips_format(rec):
        demos = rec.get("demos", [])[:n_demos]
        demo_items: List[Dict] = []
        demo_paths: List[str] = []

        for di, d in enumerate(demos, 1):
            clips = d.get("clips", [])
            for c in clips:
                role = c.get("role", "CLIP")
                p = c["audio_path"]
                lb = str(c.get("label", ""))  # demos should always have label
                demo_items.append({"tag": f"Demo {di} {role}", "label": lb, "audio_path": p})
                demo_paths.append(p)

        q = rec.get("query", {})
        qclips = q.get("clips", [])

        query_items: List[Dict] = []
        query_paths: List[str] = []

        # Heuristic: for query, the last clip is usually the one to predict (AudioRemap MIX, AnomalyDetect Query, etc.)
        # So: last query clip label => None (blank).
        # If you ever make query contain multiple predictable clips, adjust here.
        for qi, c in enumerate(qclips):
            role = c.get("role", "CLIP")
            p = c["audio_path"]

            is_last = (qi == len(qclips) - 1)
            if is_last:
                lb = None
            else:
                # keep provided query labels if any (e.g., AudioRemap query A/B labels)
                lb = str(c["label"]) if "label" in c else None

            query_items.append({"tag": f"Query {role}", "label": lb, "audio_path": p})
            query_paths.append(p)

        return demo_items, query_items, demo_paths + query_paths

    # -------- normal single-clip path --------
    demo_items = []
    demo_paths = []
    for i, d in enumerate(rec.get("demos", [])[:n_demos], 1):
        demo_items.append({"tag": f"Demo Audio {i}", "label": str(d["label"]), "audio_path": d["audio_path"]})
        demo_paths.append(d["audio_path"])

    q = rec.get("query", {})
    query_items = [{"tag": "Query Audio", "label": None, "audio_path": q["audio_path"]}]
    return demo_items, query_items, demo_paths + [q["audio_path"]]


# ============================================================
# Prepare Batch
# ============================================================

def prepare_batch(
    records,
    processor,
    base_dir=None,
    n_demos=2,
    target_sr=16000,
    strict_demos=True,
    morse_az_bank: Optional[Dict[str, np.ndarray]] = None,
):
    """
    Normal tasks: audio = demos + query
    Multi-clip tasks: audio = flattened demo clips + flattened query clips
    MorseCode (if bank provided): audio = A..Z bank + (demos + query)
    """
    inputs = []

    for rec in records:
        try:
            if strict_demos and len(rec.get("demos", [])) < n_demos:
                continue

            task = rec.get("task", "None")
            prepend_morse_az = (task == "MorseCode" and morse_az_bank is not None)

            demo_items, query_items, flat_paths = _flatten_for_prompt(rec, n_demos=n_demos)

            def resolve(p: str) -> str:
                return str(Path(base_dir) / p) if base_dir else p

            flat_paths = [resolve(p) for p in flat_paths]

            wavs = [load_audio(p, target_sr=target_sr) for p in flat_paths]

            az_wavs = []
            if prepend_morse_az:
                az_wavs = [morse_az_bank[ch] for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]

            max_len = max(
                ([len(y) for y in wavs] if wavs else [0])
                + ([len(y) for y in az_wavs] if az_wavs else [0])
            )

            wavs = [pad_audio(y, max_len) for y in wavs]
            if az_wavs:
                az_wavs = [pad_audio(y, max_len) for y in az_wavs]

            messages = build_prompt(
                task_name=task,
                demo_items=[{"tag": it["tag"], "label": it.get("label", "")} for it in demo_items],
                query_items=[{"tag": it["tag"], "label": it.get("label", None)} for it in query_items],
                prepend_morse_az=prepend_morse_az,
            )

            prompt = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            audio_list = []
            if prepend_morse_az:
                audio_list.extend([(y, target_sr) for y in az_wavs])
            audio_list.extend([(y, target_sr) for y in wavs])

            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"audio": audio_list},
                "meta": {
                    "id": rec.get("id", ""),
                    "task": task,
                    "gt": rec.get("query", {}).get("label", None),
                }
            })

        except Exception as e:
            print("skip:", e)

    return inputs


# ============================================================
# Morse A–Z sanity test
# ============================================================

def build_morse_az_sanity_inputs(processor, morse_az_bank: Dict[str, np.ndarray], target_sr: int = 16000):
    messages = build_prompt(
        task_name="MorseCode",
        demo_items=[],
        query_items=[{"tag": "Query Audio", "label": None}],
        prepend_morse_az=True
    )
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    az_raw = [morse_az_bank[ch] for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    max_len = max(len(x) for x in az_raw)
    az_padded = [pad_audio(x, max_len) for x in az_raw]

    inputs = []
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        query = pad_audio(morse_az_bank[ch], max_len)
        audio_list = [(y, target_sr) for y in az_padded] + [(query, target_sr)]
        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"audio": audio_list},
            "meta": {"id": f"AZ_SANITY_{ch}", "task": "MorseCode", "gt": ch}
        })
    return inputs


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", default="/mnt/data/robertchen/data/meta_1")
    parser.add_argument("--base_dir", default="/mnt/data/robertchen")
    parser.add_argument("--out_dir", default="/mnt/data/robertchen/AudioICL/result/none")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--strict_demos", action="store_true",
                        help="If set, skip records with fewer than n_demos demos. (Default: False)")

    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_mem_util", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=16)

    # Morse A-Z bank (optional)
    parser.add_argument("--morse_az_dir", type=str, default=None,
                        help="If set, load A.wav..Z.wav and prepend them for MorseCode prompts (bank).")
    parser.add_argument("--morse_az_sanity", action="store_true",
                        help="If set, run only the A-Z sanity check and exit.")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model, trust_remote_code=True)

    morse_az_bank = None
    if args.morse_az_dir is not None:
        morse_az_bank = load_morse_az_bank(Path(args.morse_az_dir), target_sr=args.target_sr)
        print(f"Loaded Morse A-Z bank from: {args.morse_az_dir}")

    model_name = args.model.split("/")[-1]

    # ===== sanity mode =====
    if args.morse_az_sanity:
        if morse_az_bank is None:
            raise RuntimeError("--morse_az_sanity requires --morse_az_dir")

        out_file = out_dir / f"{model_name}_morse_az_sanity.jsonl"
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            limit_mm_per_prompt={"audio": 27},  # 26 bank + 1 query
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util
        )

        params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop=["<|im_end|>"])
        inputs = build_morse_az_sanity_inputs(processor, morse_az_bank, target_sr=args.target_sr)

        done = get_done_ids(out_file)
        inputs = [x for x in inputs if x["meta"]["id"] not in done]

        if not inputs:
            print("All sanity cases already done.")
        else:
            outputs = llm.generate(inputs, params)
            correct = 0
            total = 0
            with open(out_file, "a", encoding="utf-8") as f:
                for inp, out in zip(inputs, outputs):
                    resp = out.outputs[0].text.strip()
                    pred = normalize_answer_letters_only(resp)
                    gt = str(inp["meta"]["gt"]).upper()
                    ok = (pred == gt)
                    correct += int(ok)
                    total += 1
                    f.write(json.dumps({
                        "id": inp["meta"]["id"],
                        "task": inp["meta"]["task"],
                        "response": resp,
                        "pred": pred,
                        "gt": gt,
                        "correct": ok,
                    }, ensure_ascii=False) + "\n")

            print(f"[Morse AZ SANITY] total={total} acc={correct/max(1,total):.4f} -> {out_file}")

        del llm
        cleanup_vllm()
        return

    # ===== normal mode =====
    meta_path = Path(args.meta_path)
    jsonl_files = collect_jsonl_files(meta_path)
    if not jsonl_files:
        raise RuntimeError(f"No .jsonl files found under: {meta_path}")

    ks = [parse_k_from_filename(fp) for fp in jsonl_files]
    max_k = max(ks)
    print(f"Found {len(jsonl_files)} meta files. k values: {sorted(ks)}. max_k={max_k}")

    # conservative audio limit:
    base_limit = 3 * max_k + 3
    audio_limit = (26 + base_limit) if morse_az_bank is not None else base_limit

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        limit_mm_per_prompt={"audio": audio_limit},
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util
    )

    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop=["<|im_end|>"])
    bs = args.batch_size

    for fp in jsonl_files:
        k = parse_k_from_filename(fp)
        out_file = out_dir / f"{model_name}_results_k{k}.jsonl"

        done = get_done_ids(out_file)
        all_records = list(read_jsonl(fp))
        records = [r for r in all_records if r.get("id") not in done]

        print(f"\n=== Running: {fp.name} | k={k} -> n_demos={k}")
        print(f"Output: {out_file.name}")
        print(f"records total: {len(all_records)} | already done: {len(done)} | remaining: {len(records)}")

        for i in tqdm(range(0, len(records), bs), desc=f"batches k={k}"):
            batch = records[i:i + bs]

            inputs = prepare_batch(
                batch,
                processor,
                base_dir=args.base_dir,
                n_demos=k,
                target_sr=args.target_sr,
                strict_demos=args.strict_demos,
                morse_az_bank=morse_az_bank,
            )
            if not inputs:
                continue

            outputs = llm.generate(inputs, params)

            with open(out_file, "a", encoding="utf-8") as f:
                for inp, out in zip(inputs, outputs):
                    f.write(json.dumps({
                        "id": inp["meta"]["id"],
                        "task": inp["meta"]["task"],
                        "response": out.outputs[0].text.strip(),
                        "gt": inp["meta"]["gt"]
                    }, ensure_ascii=False) + "\n")

    del llm
    cleanup_vllm()


if __name__ == "__main__":
    main()