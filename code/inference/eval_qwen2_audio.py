import os
# [CRITICAL] MUST set before importing vllm (fix CUDA init failures)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import json
import argparse
import gc
import re
from pathlib import Path
from typing import Dict, List, Iterable, Optional, Tuple

import numpy as np
import librosa
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from transformers import AutoTokenizer


# ============================================================
# Task Instructions (ICL reasoning prompts) - AudioICL
# ============================================================

TASK_INSTRUCTIONS = {
    "None": "",

    "General": """
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
    """Force release vLLM GPU memory."""
    try:
        destroy_model_parallel()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GPU memory released.")


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """vLLM audio input: numpy float32 1D"""
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32)


def pad_audio(y: np.ndarray, length: int) -> np.ndarray:
    if len(y) >= length:
        return y[:length]
    return np.pad(y, (0, length - len(y)), mode="constant")


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_done_ids(path: Path) -> set:
    """Always-on resume: read existing output jsonl and collect done ids."""
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    return done


def parse_k_from_filename(p: Path) -> int:
    """
    Parse k from filename, e.g. meta_k10.jsonl -> 10
    Accepts any stem containing k<number>.
    """
    m = re.search(r"(?:^|[_\-])k(\d+)(?:[_\-]|$)", p.stem)
    if not m:
        m = re.search(r"k(\d+)", p.stem)
    if not m:
        raise ValueError(f"Cannot parse k from filename: {p.name} (expect something like *_k5.jsonl)")
    return int(m.group(1))


def collect_jsonl_files(meta_path: Path) -> List[Path]:
    """If meta_path is file -> [file]; if dir -> all *.jsonl sorted"""
    if meta_path.is_file():
        return [meta_path]
    if meta_path.is_dir():
        return sorted(meta_path.glob("*.jsonl"))
    raise FileNotFoundError(f"meta_path not found: {meta_path}")


def normalize_answer_letters_only(text: str) -> str:
    """Uppercase and keep only A-Z letters (for Morse sanity/acc calc)."""
    if text is None:
        return ""
    t = text.strip().upper()
    t = re.sub(r"[^A-Z]", "", t)
    return t


def load_morse_az_bank(az_dir: Path, target_sr: int = 16000) -> Dict[str, np.ndarray]:
    """
    Load A.wav ... Z.wav once.
    """
    bank: Dict[str, np.ndarray] = {}
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        p = az_dir / f"{ch}.wav"
        if not p.exists():
            raise FileNotFoundError(f"Missing Morse bank file: {p}")
        bank[ch] = load_audio(str(p), target_sr=target_sr)
    return bank


# ============================================================
# Prompt Builder & Flatten Logic (Multi-clip support)
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
      demo_items: [{"tag", "label", "audio_path"}]
      query_items: [{"tag", "label(optional)", "audio_path"}]
      flat_audio_paths: [path...] in exact placeholder order
    """
    if _has_clips_format(rec):
        demos = rec.get("demos", [])[:n_demos]
        demo_items: List[Dict] = []
        demo_paths: List[str] = []

        for di, d in enumerate(demos, 1):
            clips = d.get("clips", [])
            for c in clips:
                role = c.get("role", "CLIP")
                p = c["audio_path"]
                lb = str(c.get("label", ""))
                demo_items.append({"tag": f"Demo {di} {role}", "label": lb, "audio_path": p})
                demo_paths.append(p)

        q = rec.get("query", {})
        qclips = q.get("clips", [])

        query_items: List[Dict] = []
        query_paths: List[str] = []

        for qi, c in enumerate(qclips):
            role = c.get("role", "CLIP")
            p = c["audio_path"]
            is_last = (qi == len(qclips) - 1)
            lb = None if is_last else str(c["label"]) if "label" in c else None

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


def build_prompt(
    task_name: str,
    demo_items: List[Dict],
    query_items: List[Dict],
    prepend_morse_az: bool = False,
    instruction_override: Optional[str] = None
) -> List[dict]:
    instruction = instruction_override or TASK_INSTRUCTIONS.get(task_name, TASK_INSTRUCTIONS["General"])
    A = "<|audio_bos|><|AUDIO|><|audio_eos|>"

    lines = ""

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
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text},
    ]
    return messages


# ============================================================
# Prepare Batch for AudioICL (Multi-clip logic applied)
# ============================================================

def prepare_audioicl_inputs(
    records: List[dict],
    tokenizer,
    base_dir: Optional[str] = None,
    n_demos: int = 2,
    target_sr: int = 16000,
    strict_demos: bool = True,
    morse_az_bank: Optional[Dict[str, np.ndarray]] = None,
):
    inputs = []
    loaded_items = []
    max_len = 0

    # Phase 1: Flatten, load audios, and compute per-batch max_len
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

            cur_max = max(
                ([len(x) for x in wavs] if wavs else [0])
                + ([len(x) for x in az_wavs] if az_wavs else [0])
            )
            max_len = max(max_len, cur_max)

            loaded_items.append((rec, task, demo_items, query_items, wavs, az_wavs, prepend_morse_az))
        except Exception as e:
            print(f"skip rec due to error: {e}")
            continue

    if not loaded_items:
        return []

    # Phase 2: build prompts + vLLM inputs
    for rec, task, demo_items, query_items, wavs, az_wavs, prepend_morse_az in loaded_items:
        # Pad all audios within this batch to max_len
        wavs = [pad_audio(y, max_len) for y in wavs]
        if az_wavs:
            az_wavs = [pad_audio(y, max_len) for y in az_wavs]
            
        instruction_override = rec.get("instruction")

        messages = build_prompt(
            task_name=task,
            demo_items=[{"tag": it["tag"], "label": it.get("label", "")} for it in demo_items],
            query_items=[{"tag": it["tag"], "label": it.get("label", None)} for it in query_items],
            prepend_morse_az=prepend_morse_az,
            instruction_override=instruction_override
        )

        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"Template generation failed for id={rec.get('id')}: {e}")
            continue

        audio_list = []

        if prepend_morse_az:
            audio_list.extend([(y, target_sr) for y in az_wavs])
        
        audio_list.extend([(y, target_sr) for y in wavs])

        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"audio": audio_list},
            "meta": {
                "id": rec["id"],
                "task": task,
                "gt": rec.get("query", {}).get("label", None),
            }
        })

    return inputs


# ============================================================
# MorseCode A–Z sanity test (leakage check)
# ============================================================

def build_morse_az_sanity_inputs(tokenizer, morse_az_bank: Dict[str, np.ndarray], target_sr: int = 16000):
    messages = build_prompt(
        task_name="MorseCode",
        demo_items=[],
        query_items=[{"tag": "Query Audio", "label": None}],
        prepend_morse_az=True
    )
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Pad A–Z once to max length among bank
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
            "meta": {
                "id": f"AZ_SANITY_{ch}",
                "task": "MorseCode",
                "gt": ch,
            }
        })
    return inputs


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", default="/mnt/data/robertchen/data/meta_1")
    parser.add_argument("--base_dir", default="/mnt/data/robertchen")
    parser.add_argument("--out_dir", default="/mnt/data/robertchen/AudioICL/result/none")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")

    parser.add_argument("--strict_demos", action="store_true",
                        help="If set, skip records with fewer than n_demos demos.")
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--gpu_mem_util", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_tokens", type=int, default=16)

    # MorseCode special
    parser.add_argument("--morse_az_dir", default="/mnt/data/robertchen/data/MorseCode/alphabet",
                        help="Directory containing A.wav..Z.wav for MorseCode reference bank.")
    parser.add_argument("--morse_sanity_az", action="store_true",
                        help="Run MorseCode leakage sanity test (A–Z ref + A–Z query) and exit.")
    return parser.parse_args()


def main():
    args = parse_args()

    meta_path = Path(args.meta_path)
    jsonl_files = collect_jsonl_files(meta_path)
    if not jsonl_files:
        raise RuntimeError(f"No .jsonl files found under: {meta_path}")

    # Parse all k to decide max mm per prompt once
    ks = []
    for fp in jsonl_files:
        try:
            ks.append(parse_k_from_filename(fp))
        except Exception:
            pass
    max_k = max(ks) if ks else 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- tokenizer ----
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ---- Morse A–Z bank (optional) ----
    morse_az_bank = None
    if args.morse_az_dir is not None:
        morse_az_bank = load_morse_az_bank(Path(args.morse_az_dir), target_sr=args.target_sr)
        print(f"Loaded Morse A–Z bank from: {args.morse_az_dir}")

    # ---- vLLM init (only once) ----
    # conservative audio limit calculation specifically for multi-clip sets
    base_limit = 3 * max_k + 3
    audio_limit = (26 + base_limit) if morse_az_bank is not None else base_limit

    print(f"Initializing vLLM: {args.model} (limit audio per prompt = {audio_limit})")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        limit_mm_per_prompt={"audio": audio_limit},
    )

    # ---- sampling ----
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)
    if hasattr(tokenizer, "additional_special_tokens_ids"):
        stop_token_ids.extend(tokenizer.additional_special_tokens_ids)

    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
    )

    model_short = args.model.split("/")[-1]

    # ---- Morse A–Z sanity test ----
    if args.morse_sanity_az:
        if morse_az_bank is None:
            raise RuntimeError("--morse_sanity_az requires --morse_az_dir")

        out_file = out_dir / f"{model_short}_Morse_AZ_SANITY.jsonl"
        done = get_done_ids(out_file)

        inputs = build_morse_az_sanity_inputs(tokenizer, morse_az_bank, target_sr=args.target_sr)
        inputs = [x for x in inputs if x["meta"]["id"] not in done]

        if not inputs:
            print("All sanity cases already done.")
        else:
            outputs = llm.generate(inputs, params, use_tqdm=False)
            correct = 0
            total = 0

            with out_file.open("a", encoding="utf-8") as f:
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

            acc = correct / max(1, total)
            print(f"[Morse AZ SANITY] total={total} acc={acc:.4f} -> {out_file}")

        # cleanup and exit
        del llm
        del tokenizer
        cleanup_vllm()
        return

    # ---- run per jsonl file ----
    for fp in jsonl_files:
        try:
            k = parse_k_from_filename(fp)
        except Exception:
            k = 0
        n_demos = k

        out_file = out_dir / f"{model_short}_results_k{k}.jsonl"

        # ALWAYS-ON RESUME
        done = get_done_ids(out_file)

        all_records = list(read_jsonl(fp))
        records = [r for r in all_records if r.get("id") not in done]

        print(f"\n=== Running: {fp.name}  (k={k} -> n_demos={n_demos})")
        print(f"Output: {out_file.name}")
        print(f"records total: {len(all_records)} | already done: {len(done)} | remaining: {len(records)}")

        bs = args.batch_size
        for i in tqdm(range(0, len(records), bs), desc=f"batches {fp.stem}"):
            batch = records[i:i + bs]

            inputs = prepare_audioicl_inputs(
                batch,
                tokenizer=tokenizer,
                base_dir=args.base_dir,
                n_demos=n_demos,
                target_sr=args.target_sr,
                strict_demos=args.strict_demos,
                morse_az_bank=morse_az_bank,
            )
            if not inputs:
                continue

            outputs = llm.generate(inputs, params, use_tqdm=False)

            with out_file.open("a", encoding="utf-8") as f:
                for inp, out in zip(inputs, outputs):
                    f.write(json.dumps({
                        "id": inp["meta"]["id"],
                        "task": inp["meta"]["task"],
                        "response": out.outputs[0].text.strip(),
                        "gt": inp["meta"]["gt"],
                    }, ensure_ascii=False) + "\n")

    # ---- cleanup ----
    del llm
    del tokenizer
    cleanup_vllm()
    print("All done.")


if __name__ == "__main__":
    main()