"""Extract missing noun phrases from the training dataset using spaCy,
translate them into Asante Twi via Gemini, and augment the offline dictionary.

This creates the 'Expanded Dictionary' condition for comparing test-time
dictionary scaling in the paper.
"""

import argparse
import concurrent.futures
import gzip
import json
import os
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Set

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_builder.phrases import clean_nominal, load_nlp
from twi_translate.lookup import DEFAULT_PATH, TwiLookup

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


def get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        env_path = "/home/owusus/Documents/GitHub/nsanku-mt-benchmark/.env"
        if os.path.exists(env_path):
            for line in open(env_path):
                if line.startswith("GEMINI_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        raise ValueError("GEMINI_API_KEY not found in environment or .env")
    return key


def extract_missing_from_data(data_dir: str, nlp, lookup: TwiLookup) -> Counter:
    """Scan train, validation, and test splits with spaCy to find all missing noun phrases."""
    print("Extracting noun phrases with spaCy across dataset splits...", flush=True)
    counts = Counter()
    splits = ["train", "validation", "test"]

    for split in splits:
        path = os.path.join(data_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        print(f"  Scanning {path}...", flush=True)
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                ex = json.loads(line)
                
                # 1. Quoted tool phrases from the example
                tool_phrases = ex.get("tool_phrases", [])
                for p in tool_phrases:
                    p_clean = p.strip().lower()
                    if len(p_clean) >= 3 and not lookup.lookup(p_clean)["translations"]:
                        counts[p_clean] += 1

                # 2. Extract noun chunks using spaCy from the source English text
                eng_text = ex.get("english", "")
                if eng_text:
                    doc = nlp(eng_text)
                    for chunk in doc.noun_chunks:
                        c_clean = clean_nominal(chunk)
                        if c_clean and not lookup.lookup(c_clean)["translations"]:
                            counts[c_clean] += 1

                if idx > 0 and idx % 3000 == 0:
                    print(f"    processed {idx:,} examples, {len(counts):,} unique missing phrases so far...", flush=True)

    print(f"Extraction complete: {len(counts):,} unique missing noun phrases found.", flush=True)
    return counts


def translate_batch(batch: List[str], api_key: str, model: str = "gemini-2.5-flash") -> Dict[str, str]:
    """Call Gemini to translate a batch of English noun phrases into Asante Twi."""
    prompt = (
        "Translate each English noun phrase into natural, native Asante Twi (Akan). "
        "Return ONLY a JSON object mapping the exact lowercase English phrase to its Asante Twi translation. "
        "Keep proper names and loanwords accurate in Akan orthography.\n\n"
        f"Phrases:\n{json.dumps(batch, ensure_ascii=False)}"
    )

    url = GEMINI_URL.format(model=model, key=api_key)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0}
    }

    for attempt in range(5):
        try:
            r = requests.post(url, json=body, timeout=60)
            if r.status_code == 200:
                raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(raw)
            elif r.status_code == 429:
                time.sleep(2 ** attempt * 3)
            else:
                time.sleep(2)
        except Exception as e:
            time.sleep(2 ** attempt * 2)

    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data", help="Directory containing train/val/test jsonl")
    ap.add_argument("--dict", default=DEFAULT_PATH, help="Existing dictionary path")
    ap.add_argument("--out", default="twi_translate/data/twi_dictionary_expanded.tsv.gz")
    ap.add_argument("--checkpoint", default="data/gemini_translations_cache.json")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--top-n", type=int, default=0, help="Limit to top N most frequent phrases (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    api_key = get_api_key()
    lookup = TwiLookup(path=args.dict)
    nlp = load_nlp()

    # Step 1: Extract missing noun phrases
    counts = extract_missing_from_data(args.data, nlp, lookup)
    phrases = [p for p, _ in counts.most_common(args.top_n if args.top_n > 0 else None)]
    print(f"Targeting {len(phrases):,} missing noun phrases for Gemini translation.", flush=True)

    # Step 2: Load checkpoint cache if exists
    translations = {}
    if os.path.exists(args.checkpoint):
        try:
            with open(args.checkpoint, "r", encoding="utf-8") as f:
                translations = json.load(f)
            print(f"Loaded {len(translations):,} existing translations from checkpoint.", flush=True)
        except Exception:
            translations = {}

    to_translate = [p for p in phrases if p not in translations]
    print(f"{len(to_translate):,} phrases remaining to translate.", flush=True)

    # Step 3: Batch translate with Gemini
    batches = [to_translate[i:i + args.batch_size] for i in range(0, len(to_translate), args.batch_size)]
    print(f"Processing {len(batches)} batches across {args.workers} workers...", flush=True)

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(translate_batch, b, api_key, args.model): b for b in batches}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            for k, v in res.items():
                if isinstance(v, str) and v.strip():
                    translations[k.strip().lower()] = v.strip()
            completed += 1
            if completed % 10 == 0 or completed == len(batches):
                print(f"  {completed}/{len(batches)} batches complete ({len(translations):,} translations total)...", flush=True)
                with open(args.checkpoint, "w", encoding="utf-8") as f:
                    json.dump(translations, f, ensure_ascii=False, indent=1)

    # Step 4: Write augmented dictionary
    print(f"Writing expanded dictionary to {args.out}...", flush=True)
    table = dict(lookup.table)
    new_entries_count = 0
    for eng, twi in translations.items():
        opts = table.setdefault(eng, [])
        if twi not in opts:
            opts.append(twi)
            new_entries_count += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write("# English phrase -> Twi candidates (tab-separated). CC-BY-NC-4.0.\n")
        f.write(f"# Original SHOLA table + Gemini augmented ({new_entries_count:,} new terms).\n")
        for key in sorted(table):
            f.write(key.replace("\t", " ") + "\t" + "\t".join(o.replace("\t", " ") for o in table[key]) + "\n")

    print(f"Success! Expanded dictionary has {len(table):,} total entries (+{new_entries_count:,} new terms).", flush=True)


if __name__ == "__main__":
    main()
