# inference_audioicl_qwen25_omni_thinker.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


# -------------------------
# Config (YOU SHOULD EDIT)
# -------------------------
MODEL_ID = "Qwen/Qwen2.5-Omni-7B"  # or other omni checkpoint
META_PATH = "meta.json"
DATA_PATH = "audioicl.jsonl"
OUT_PATH = "predictions.jsonl"

# IMPORTANT:
# audioicl.jsonl 內的 audio_path 通常是「相對於 dataset root」的相對路徑
# 你需要為每個 task 指定它對應的 root，才能組出絕對路徑
TASK_ROOTS = {
    "speaker_recognition": "/home5/b10303106/VCTK/wav48_silence_trimmed",
    "accent": "/path/to/VCTK-Corpus-0.92/wav48_silence_trimmed",
    "emotion": "/home5/b10303106/RAVDESS/archive",  # 你剛剛的 RAVDESS root
}

# generation params
MAX_NEW_TOKENS = 16
DO_SAMPLE = False


# -------------------------
# IO helpers
# -------------------------
def load_json(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p.resolve()}")
    return json.loads(p.read_text(encoding="utf-8"))

def iter_jsonl(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------
# Template -> multimodal message builder
# -------------------------
AUDIO_MARKERS = ("<audio_demo_1>", "<audio_demo_2>", "<audio_query>")

def split_template_by_markers(template: str) -> List[Tuple[str, Optional[str]]]:
    """
    Split a template into segments where each segment is (text, marker_to_insert_next_audio_or_None).
    Example:
      "foo <audio_demo_1> bar <audio_query> baz"
    becomes:
      [("foo ", "<audio_demo_1>"), (" bar ", "<audio_query>"), (" baz", None)]
    """
    segments: List[Tuple[str, Optional[str]]] = []
    s = template
    while True:
        # find nearest marker
        positions = [(m, s.find(m)) for m in AUDIO_MARKERS]
        positions = [(m, pos) for m, pos in positions if pos != -1]
        if not positions:
            segments.append((s, None))
            break

        m, pos = min(positions, key=lambda x: x[1])
        before = s[:pos]
        segments.append((before, m))
        s = s[pos + len(m):]
    return segments

def build_conversation_from_template(
    template: str,
    label1: str,
    label2: str,
    demo1_path_abs: str,
    demo2_path_abs: str,
    query_path_abs: str,
) -> List[Dict[str, Any]]:
    """
    Make a single user message with interleaved text/audio that matches the template order.
    """
    formatted = template.format(label1=label1, label2=label2)

    segs = split_template_by_markers(formatted)

    marker_to_audio = {
        "<audio_demo_1>": demo1_path_abs,
        "<audio_demo_2>": demo2_path_abs,
        "<audio_query>": query_path_abs,
    }

    content: List[Dict[str, Any]] = []
    for text, marker in segs:
        if text:
            content.append({"type": "text", "text": text})
        if marker is not None:
            content.append({"type": "audio", "audio": marker_to_audio[marker]})

    conversation = [
        {
            "role": "user",
            "content": content
        }
    ]
    return conversation


# -------------------------
# Inference
# -------------------------
@torch.inference_mode()
def infer_one(
    model,
    processor,
    prompt_template: str,
    task: str,
    demo1_rel: str,
    demo2_rel: str,
    query_rel: str,
    label1: str,
    label2: str,
) -> str:
    root = TASK_ROOTS.get(task)
    if root is None:
        raise KeyError(f"Missing TASK_ROOTS config for task={task}")

    demo1_abs = str(Path(root) / demo1_rel)
    demo2_abs = str(Path(root) / demo2_rel)
    query_abs = str(Path(root) / query_rel)

    conversation = build_conversation_from_template(
        prompt_template,
        label1=label1,
        label2=label2,
        demo1_path_abs=demo1_abs,
        demo2_path_abs=demo2_abs,
        query_path_abs=query_abs,
    )
    print(conversation)
    # Build text prompt and extract multimodal inputs
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    ).to(model.device).to(model.dtype)

    gen = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        use_audio_in_video=False,
    )

    # Decode only the newly generated tokens (avoid echoing the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = gen[:, prompt_len:]
    out = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return out


def main():
    meta = load_json(META_PATH)
    task_to_template = {k: v["prompt_template"] for k, v in meta["tasks"].items()}

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)

    preds: List[Dict[str, Any]] = []

    for ex in iter_jsonl(DATA_PATH):
        task = ex["task"]
        template = task_to_template[task]

        demo1 = ex["demos"][0]["audio_path"]
        demo2 = ex["demos"][1]["audio_path"]
        query = ex["query"]["audio_path"]

        label1 = ex["demos"][0]["label"]
        label2 = ex["demos"][1]["label"]

        pred = infer_one(
            model=model,
            processor=processor,
            prompt_template=template,
            task=task,
            demo1_rel=demo1,
            demo2_rel=demo2,
            query_rel=query,
            label1=label1,
            label2=label2,
        )

        preds.append({
            "id": ex["id"],
            "task": task,
            "pred": pred,
            "gold": ex["query"]["label"],
            "options": ex["options"],
        })

        # (optional) print a few
        if len(preds) <= 3:
            print(f"[{ex['id']}] task={task} pred={pred} gold={ex['query']['label']} options={ex['options']}")

    write_jsonl(preds, OUT_PATH)
    print(f"Done. Wrote {len(preds)} predictions to {OUT_PATH}")


if __name__ == "__main__":
    main()
