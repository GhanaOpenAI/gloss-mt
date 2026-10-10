"""Build the Twi translation tool-calling + reasoning SFT dataset.

Source: ghananlpcommunity/twi-english-reasoning-translation, English->Twi rows
(`pristine` and `ghana_chat_en2tw`, whose Twi side is the gold reference). Each row already
has a <think> reasoning trace that quotes the English phrases it translates.

For each row:
  1. the quoted English noun phrases in the reasoning           -> tool call
  2. those phrases looked up in the bundled offline dictionary  -> tool result
     (phrases not in the dictionary come back with an empty list)
  3. the original reasoning + gold Twi translation              -> final assistant turn

So the trajectory is:
  system -> user -> assistant(tool_calls) -> tool(results) -> assistant(reasoning + translation)
and the dictionary results come before, and feed, the linguistic reasoning.

Usage:
  python dataset_builder/build_dataset.py --out data
"""

import argparse
import hashlib
import json
import os
import random
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_builder.phrases import load_nlp, phrases_for_examples  # noqa: E402
from twi_translate.lookup import TwiLookup  # noqa: E402
from twi_translate.schema import LOOKUP_TOOL, SYSTEM_PROMPT, TOOL_NAME, USER_TEMPLATE, format_tool_result  # noqa: E402

SOURCE_REPO = "ghananlpcommunity/twi-english-reasoning-translation"
EN2TW_SOURCES = ("pristine", "ghana_chat_en2tw")


def load_rows() -> List[Dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SOURCE_REPO, "data/train-00000-of-00001.parquet", repo_type="dataset")
    rows = []
    for r in pq.read_table(path).to_pylist():
        if r["source"] not in EN2TW_SOURCES or len(r["messages"]) != 2:
            continue
        user, answer = r["messages"][0]["content"], r["messages"][1]["content"]
        if "<think>" not in answer or "</think>" not in answer or "\n\n" not in user:
            continue
        reasoning, translation = answer.split("<think>", 1)[1].split("</think>", 1)
        text = user.split("\n\n", 1)[1]
        if not reasoning.strip() or not translation.strip() or not text.strip():
            continue
        rows.append({"id": r["id"], "source": r["source"], "markdown": r["markdown"], "text": text,
                     "reasoning": reasoning.strip(), "translation": translation.strip()})
    return rows


def split_of(row_id: str, n_val: int, n_test: int, total: int) -> str:
    """Stable split from a hash of the row id."""
    h = int(hashlib.sha1(row_id.encode()).hexdigest(), 16) % total
    return "test" if h < n_test else "validation" if h < n_test + n_val else "train"


def build_example(row: Dict, phrases: List[str], lookup: TwiLookup, idx: int) -> Dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=row["text"])},
    ]
    results = lookup.lookup_many(phrases)
    if results:
        call_id = f"call_{idx:06d}"
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": TOOL_NAME, "arguments": {"phrases": [r["phrase"] for r in results]}},
            }],
        })
        messages.append({"role": "tool", "tool_call_id": call_id, "name": TOOL_NAME,
                         "content": format_tool_result(results)})
    messages.append({"role": "assistant", "reasoning_content": row["reasoning"], "content": row["translation"]})
    tool_result = messages[3]["content"] if results else ""
    return {
        "id": row["id"],
        # flat, human-readable view of the trajectory (training uses `messages`)
        "english": row["text"],
        "tool_phrases": [r["phrase"] for r in results],
        "tool_result": tool_result,
        "reasoning": row["reasoning"],
        "translation": row["translation"],
        "messages": messages,
        "tools": [LOOKUP_TOOL],
        "source": row["source"],
        "markdown": row["markdown"],
        "num_phrases": len(results),
        "num_found": sum(1 for r in results if r["translations"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--validation", type=int, default=300)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--max-phrases", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--push-to-hub", default=None, help="optional HF dataset repo id to upload to")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = load_rows()
    print(f"{len(rows):,} English->Twi rows with reasoning", flush=True)
    lookup = TwiLookup()
    nlp = load_nlp()
    phrase_lists = phrases_for_examples(nlp, ((r["reasoning"], r["text"]) for r in rows))

    splits: Dict[str, List[Dict]] = {"train": [], "validation": [], "test": []}
    for idx, (row, phrases) in enumerate(zip(rows, phrase_lists)):
        ex = build_example(row, phrases[: args.max_phrases], lookup, idx)
        splits[split_of(row["id"], args.validation, args.test, len(rows))].append(ex)

    stats = {}
    for name, exs in splits.items():
        random.Random(f"{args.seed}-{name}").shuffle(exs)
        with open(os.path.join(args.out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for ex in exs:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        n_ph = sum(e["num_phrases"] for e in exs)
        stats[name] = {
            "examples": len(exs),
            "with_tool_call": sum(1 for e in exs if e["num_phrases"]),
            "phrases": n_ph,
            "phrases_found": sum(e["num_found"] for e in exs),
            "lookup_hit_rate": round(sum(e["num_found"] for e in exs) / max(n_ph, 1), 4),
            "all_phrases_missing": sum(1 for e in exs if e["num_phrases"] and not e["num_found"]),
        }
        print(f"[{name}] {stats[name]}", flush=True)
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    if args.push_to_hub:
        from datasets import load_dataset

        files = {s: os.path.join(args.out, f"{s}.jsonl") for s in splits}
        load_dataset("json", data_files=files).push_to_hub(args.push_to_hub, private=True)
        print(f"Pushed to https://huggingface.co/datasets/{args.push_to_hub}")


if __name__ == "__main__":
    main()
