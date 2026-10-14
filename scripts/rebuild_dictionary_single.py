"""Rebuild the English->Twi dictionary with exactly one translation per phrase.

~412k entries of twi_dictionary.tsv.gz carry 2-3 candidates (the superseded
gemini-3.6-flash fallback of AfriSpeech/shola-machine-translations); single-translation entries
are kept as they are. Every multi-candidate phrase is translated afresh by Gemini (English only,
the old candidates are not shown) and gets that one translation. The canonical dictionary is
not modified: the result is written to a new file (default twi_dictionary_single.tsv.gz).

Translations are appended to a JSONL cache as they arrive, so the run can be stopped and
resumed; re-running only sends phrases that are not in the cache yet.

  python scripts/rebuild_dictionary_single.py --limit 300          # pilot (prints usage)
  python scripts/rebuild_dictionary_single.py                      # full run, then writes --out
"""

import argparse
import concurrent.futures
import gzip
import json
import os
import re
import sys
import threading
import time
from typing import Dict, List, Tuple

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twi_translate.lookup import DEFAULT_PATH  # noqa: E402

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
SINGLE_PATH = os.path.join(os.path.dirname(DEFAULT_PATH), "twi_dictionary_single.tsv.gz")

PROMPT = (
    "You are an expert English to Asante Twi (Akan) translator building a bilingual dictionary.\n"
    "For each English phrase below, give the single most natural, commonly used Asante Twi "
    "translation, as a native speaker would say it in everyday writing. Use standard Akan "
    "orthography (ɛ, ɔ). Keep proper names, brand names and acronyms as they are (adapt "
    "spelling only where Twi does). Give exactly one translation per phrase, with no "
    "alternatives, slashes or explanations. Return only the Twi word or phrase itself: never add "
    "anything in brackets or parentheses after it (no glosses, English, notes or variants).\n"
    "Return ONLY a JSON object mapping each English phrase, exactly as given, to its Twi "
    "translation.\n\nPhrases:\n"
)


def read_table(path: str) -> Tuple[List[str], Dict[str, List[str]]]:
    header, table = [], {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                header.append(line.rstrip("\n"))
                continue
            key, *options = line.rstrip("\n").split("\t")
            if key and options:
                table[key] = options
    return header, table


_BRACKETS = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")
_ALTERNATIVES = re.compile(r"\s*[/|]\s*")


def clean_twi(text: str) -> Tuple[str, bool]:
    """Strip bracketed asides and keep only the first of "a / b" or "a | b" alternatives.
    Returns (cleaned text, whether anything was changed); "" when nothing usable is left."""
    cleaned = _BRACKETS.sub("", text)
    cleaned = _ALTERNATIVES.split(cleaned)[0]
    cleaned = " ".join(cleaned.split()).strip(" ,;:.")
    return cleaned, cleaned != text.strip()


def load_cache(path: str) -> Dict[str, str]:
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    cache[item["phrase"]] = item["twi"]
                except (json.JSONDecodeError, KeyError):
                    continue
    return cache


def translate_batch(batch: List[str], api_key: str, model: str) -> Tuple[Dict[str, str], Dict[str, int]]:
    """One Gemini call; returns ({phrase: twi} for the phrases it answered, usage counts)."""
    body = {
        "contents": [{"parts": [{"text": PROMPT + json.dumps(batch, ensure_ascii=False)}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0},
    }
    for attempt in range(6):
        try:
            r = requests.post(GEMINI_URL.format(model=model), params={"key": api_key}, json=body, timeout=180)
            if r.status_code == 200:
                data = r.json()
                raw = json.loads(data["candidates"][0]["content"]["parts"][0]["text"])
                wanted, out, cleaned = set(batch), {}, 0
                for k, v in raw.items():
                    if not isinstance(v, str) or k.strip() not in wanted:
                        continue
                    twi, changed = clean_twi(v)
                    if twi:  # empty after cleaning -> left unanswered, retried on a re-run
                        out[k.strip()] = twi
                        cleaned += changed
                u = data.get("usageMetadata", {})
                return out, {"prompt": u.get("promptTokenCount", 0), "output": u.get("candidatesTokenCount", 0),
                             "thoughts": u.get("thoughtsTokenCount", 0), "cleaned": cleaned}
            time.sleep(min(60, 2 ** attempt * (3 if r.status_code == 429 else 1)))
        except Exception:  # noqa: BLE001  (network errors, malformed JSON) -> retry
            time.sleep(min(60, 2 ** attempt))
    return {}, {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dict", default=DEFAULT_PATH, help="input dictionary (not modified)")
    ap.add_argument("--out", default=SINGLE_PATH)
    ap.add_argument("--cache", default="build/dictionary_single_cache.jsonl")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="translate only N phrases (pilot); no --out written")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set")

    header, table = read_table(args.dict)
    multi = sorted(k for k, v in table.items() if len(v) > 1)
    cache = load_cache(args.cache)
    todo = [k for k in multi if k not in cache]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(table):,} entries | {len(multi):,} with >1 candidate | {len(cache):,} cached | "
          f"{len(todo):,} to translate with {args.model}", flush=True)

    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
    lock = threading.Lock()
    usage = {"prompt": 0, "output": 0, "thoughts": 0, "cleaned": 0, "calls": 0}
    done, t0 = 0, time.time()
    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    with open(args.cache, "a", encoding="utf-8") as cf, \
            concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res, u in ex.map(lambda b: translate_batch(b, api_key, args.model), batches):
            with lock:
                for k, v in res.items():
                    cf.write(json.dumps({"phrase": k, "twi": v}, ensure_ascii=False) + "\n")
                    cache[k] = v
                cf.flush()
                for k in ("prompt", "output", "thoughts", "cleaned"):
                    usage[k] += u.get(k, 0)
                usage["calls"] += 1
                done += len(res)
                if usage["calls"] % 50 == 0 or usage["calls"] == len(batches):
                    rate = done / max(1e-9, time.time() - t0)
                    print(f"  {usage['calls']}/{len(batches)} calls | {done:,} translated | {rate:.0f}/s | "
                          f"tokens in {usage['prompt']:,} out {usage['output']:,} thinking {usage['thoughts']:,} | "
                          f"cleaned {usage['cleaned']:,}",
                          flush=True)

    missing = [k for k in multi if k not in cache]
    print(f"Done: {done:,} new translations | usage {usage} | {len(missing):,} multi-candidate phrases "
          f"still untranslated", flush=True)
    if args.limit:
        return

    if missing:
        print("Re-run to retry the untranslated phrases; their first old candidate is used until then.",
              flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write("# English phrase -> Twi translation (tab-separated), exactly one per phrase. CC-BY-NC-4.0.\n")
        f.write(f"# Single-translation entries of {os.path.basename(args.dict)} kept; the "
                f"{len(multi):,} multi-candidate entries re-translated with {args.model} "
                f"({len(missing):,} fell back to their first old candidate).\n")
        for key in sorted(table):
            twi = table[key][0] if len(table[key]) == 1 else clean_twi(cache.get(key, table[key][0]))[0] or table[key][0]
            f.write(key.replace("\t", " ") + "\t" + twi.replace("\t", " ").replace("\n", " ") + "\n")
    print(f"Wrote {args.out} ({len(table):,} entries, one translation each)", flush=True)


if __name__ == "__main__":
    main()
