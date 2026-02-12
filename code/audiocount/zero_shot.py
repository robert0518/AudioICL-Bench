# inference_audioicl_audiocount_qwen25_omni_thinker_zeroshot.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from tqdm import tqdm  # ✅ progress bar


# -------------------------
# Config (YOU SHOULD EDIT)
# -------------------------
MODEL_ID = "Qwen/Qwen2.5-Omni-7B"

DATA_PATH = "/home5/b10303106/ICL-benchmark/data/meta/meta.jsonl"
OUT_PATH = "predictions.jsonl"

TASK_ROOTS = {
    "AudioCount": "/home5/b10303106/ICL-benchmark/data/audiocount",
}

ZERO_SHOT_PROMPT = (
    "You will hear an audio clip. "
    "How many times do you hear the same percussive sound repeated in the clip? "
    "Answer with an integer from 1 to 8. "
    "Only output the number."
)

MAX_NEW_TOKENS = 8
DO_SAMPLE = False


# -------------------------
# IO helpers
# -------------------------
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
# Conversation builder
# -------------------------
def build_zero_shot_conversation(prompt: str, query_audio_abs: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt + "\n<audio_query>"},
                {"type": "audio", "audio": query_audio_abs},
            ],
        }
    ]


# -------------------------
# Inference
# -------------------------
@torch.inference_mode()
def infer_one_zero_shot(model, processor, task: str, query_rel: str) -> str:
    root = TASK_ROOTS.get(task)
    if root is None:
        raise KeyError(f"Missing TASK_ROOTS config for task={task}")

    query_abs = str(Path(root) / query_rel)

    conversation = build_zero_shot_conversation(
        prompt=ZERO_SHOT_PROMPT,
        query_audio_abs=query_abs,
    )

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

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = gen[:, prompt_len:]
    out = processor.batch_decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return out


def extract_first_int_1_to_8(text: str) -> str:
    """Find first integer 1..8 in model output. Return '' if none."""
    import re
    m = re.search(r"\b([1-8])\b", text)
    return m.group(1) if m else ""


def main():
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)

    # ✅ 先把資料讀成 list，tqdm 才能顯示總長度
    examples = list(iter_jsonl(DATA_PATH))

    preds: List[Dict[str, Any]] = []

    for ex in tqdm(examples, desc="Infer AudioCount", unit="ex"):
        task = ex.get("task", "AudioCount")
        query_rel = ex["query"]["audio_path"]
        gold = ex["query"]["label"]

        raw_pred = infer_one_zero_shot(
            model=model,
            processor=processor,
            task=task,
            query_rel=query_rel,
        )
        pred_num = extract_first_int_1_to_8(raw_pred)

        preds.append({
            "id": ex.get("id", ""),
            "task": task,
            "query_audio_path": query_rel,
            "pred_raw": raw_pred,
            "pred": pred_num,
            "gold": str(gold),
        })

    write_jsonl(preds, OUT_PATH)
    print(f"Done. Wrote {len(preds)} predictions to {OUT_PATH}")


if __name__ == "__main__":
    main()
