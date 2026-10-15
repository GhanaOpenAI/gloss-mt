"""Build (and optionally push) the chunked, glossary-annotated English-Twi parallel dataset.

Each document of data/{train,validation,test}.jsonl is split into its line-aligned English/Twi
lines and consecutive lines are packed into chunks of at most --chunk-tokens umT5 tokens. For
each chunk, spaCy extracts the noun phrases (glossary_for, as at inference), and:

  - english_pretranslated: the Twi spans of the gold chunk that Gemini maps the phrases
    to (it sees only the Twi chunk and the English phrases), inserted as "<twi>" (training
    input). A span is kept only if it is in the gold chunk verbatim, on the same line as its
    phrase;
  - twi_bracketed: the gold chunk with those same spans in "<...>" (training target).

Alignments are cached in --align-cache (JSONL, resumable); re-running only sends new chunks.

  python scripts/build_chunked_dataset.py --limit 40                 # pilot: print samples only
  python scripts/build_chunked_dataset.py --out build/glossary-chunks \\
      --push ghanaopenai/gloss-mt-parallel-en-twi
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twi_translate import TwiLookup  # noqa: E402
from twi_translate.seq2seq import (  # noqa: E402
    GLOSSARY_DICT, glossary_for, join_span, pack_lines, split_lines)

SOURCE_DATASET = "ghanaopenai/twi-translation-tool-calling"

CARD = """---
language:
- en
- ak
license: cc-by-nc-4.0
task_categories:
- translation
tags:
- twi
- akan
- ghana
- machine-translation
- glossary
- terminology
- parallel-corpus
pretty_name: English-Twi Glossary-Annotated Parallel Chunks
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

# English-Twi Glossary-Annotated Parallel Chunks

**Author:** [Ghana Open AI](https://huggingface.co/ghanaopenai)
**Supported by:** [Ghana NLP](https://ghananlp.org)

English → Twi (Akan) parallel text in short, aligned chunks, each annotated with dictionary
lookups for its key noun phrases, the Twi words the reference translation uses for them, and
versions of the English with those phrases already **pretranslated in place**, in angle
brackets. It is built for training translation models that get dictionary
terms with the source text, such as encoder-decoder models (umT5, NLLB) or
translation-specialised LLMs (TranslateGemma), which have no tool calling.

It is a re-cut of [`{source}`](https://huggingface.co/datasets/{source}): the same documents,
splits and gold Twi references, without the tool-call trajectories and reasoning traces.

## How it was built

1. Each document is split into lines; English and Twi are line-aligned (documents whose line
   counts differ are skipped: {skipped} in total).
2. Consecutive lines are packed into chunks of at most **{chunk_tokens} umT5 tokens** of
   English, cutting only at line boundaries, so every English chunk matches its Twi chunk.
   Markdown line structure (headings, lists, tables) is kept inside a chunk.
3. The chunk's noun phrases are extracted with spaCy (`en_core_web_sm`, nkrane-style
   stopword filtering) and looked up in an offline English→Twi dictionary of {dict_size:,}
   phrases with **exactly one translation each**. It starts from the Gemini translations in
   [AfriSpeech/shola-machine-translations](https://huggingface.co/datasets/AfriSpeech/shola-machine-translations):
   entries that had a single translation (`gemini-3.6-flash-one`) are kept, and the ~412k
   entries that had 2-3 candidates (the superseded `gemini-3.6-flash` fallback) were
   re-translated with `gemini-3.8-flash` into one translation each
   (`scripts/rebuild_dictionary_single.py`). Phrases not in the dictionary keep their English
   as the translation (`found` is false), telling the model to carry them over as they are.
4. **Gold alignment.** `{align_model}` is given the gold Twi chunk and the English noun phrases
   (not the English chunk) and maps each phrase to the exact span of the Twi chunk that
   translates it (or nothing). A span is kept only if it occurs verbatim in the gold Twi chunk
   **on the same line** as its English phrase and differs from the English.
5. **Pretranslation.** A phrase is replaced in the chunk by its Twi in angle brackets,
   `<twi>` (case-insensitive, whole words, longest phrase first). In the gold pair a phrase is
   bracketed only as many times as its Twi span is bracketed in the gold chunk, so both sides
   carry the same terms. Phrases without a translation stay English.
   - `english_pretranslated` uses the gold-aligned spans, and `twi_bracketed` is the gold
     Twi with the same spans in `<brackets>`: every inserted term is what the reference really
     says and sits in the same place on both sides, so a model trained on this pair learns to
     **keep** the bracketed terms and translate the English around them. Built from the
     reference (brackets are removed from the model's output).

The build script is `scripts/build_chunked_dataset.py` in
[GhanaOpenAI/gloss-mt](https://github.com/GhanaOpenAI/gloss-mt), which uses the same
`twi_translate.seq2seq` helpers as training and inference.

## Columns

| column | content |
|---|---|
| `english` | English chunk |
| `english_pretranslated` | English chunk with its phrases replaced by `<the Twi span the gold chunk uses>` (model input) |
| `twi_bracketed` | gold Twi chunk with the aligned spans in `<brackets>` (model target) |
| `twi` | gold Twi chunk (human reference, no brackets) |

Every `<term>` in `english_pretranslated` has exactly one matching `<term>` in
`twi_bracketed` (checked at build time).

## Statistics

| split | documents | chunks | chunks with ≥1 glossary hit | glossary phrases (found / total) | gold terms (aligned / noun phrases) |
|---|---|---|---|---|---|
{stats}

## Example input format

Models read `english_pretranslated` and produce `twi_bracketed` (with `translate English to Twi: ` in front for umT5; NLLB uses its
eng_Latn/twi_Latn codes instead):

```
{example}
```

The model learns to carry the bracketed terms over and to translate the rest; the brackets
are stripped from its output (`clean_output`). Validation and test inputs use dictionary
terms in brackets, and `twi` is the plain reference.

## License

CC BY-NC 4.0, same as the source dataset.
"""

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
ALIGN_PROMPT = (
    "Below is a text in Asante Twi (Akan) and a list of English noun phrases taken from its English original.\n"
    "1. For each English noun phrase, identify the exact contiguous span of words in the Twi text that translates it (or null if not found or kept in English).\n"
    "2. Produce a bracketed version of the Twi text ('bracketed_twi') where the translated Twi spans corresponding to the English noun phrases are enclosed in angle brackets <...>. Do not change, add, or drop any words or punctuation from the Twi text; only insert < and > around the translated terms.\n"
    "- Never nest brackets: brackets must never be inside another bracket (prefer the longer span if overlapping).\n\n"
    "Return ONLY a JSON object with two keys:\n"
    "- \"mapping\": an object mapping each English noun phrase to its exact Twi span (or null)\n"
    "- \"bracketed_twi\": the Twi text with non-nested angle brackets around those translated spans\n\n"
    "Twi text:\n{tw}\n\nEnglish noun phrases:\n{phrases}"
)


def contains(text: str, phrase: str) -> Optional[str]:
    """The phrase as written in text (case-insensitive, whole words, any spacing), else None."""
    words = phrase.split()
    if not words:
        return None
    m = re.search(r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)", text, re.IGNORECASE)
    return m.group(0) if m else None


def align(tw: str, phrases: List[str], api_key: str, model: str) -> Optional[Dict]:
    """One Gemini call with the Twi chunk and the English noun phrases:
    returns {"mapping": {...}, "bracketed_twi": "..."} or None if repeated failure."""
    body = {"contents": [{"parts": [{"text": ALIGN_PROMPT.format(
                tw=tw, phrases=json.dumps(phrases, ensure_ascii=False))}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.0,
                "thinkingConfig": {"thinkingBudget": 0},
            }}
    for attempt in range(6):
        try:
            r = requests.post(GEMINI_URL.format(model=model), params={"key": api_key}, json=body, timeout=120)
            if r.status_code == 200:
                raw = json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
                if isinstance(raw, dict) and "mapping" in raw and "bracketed_twi" in raw:
                    return raw
            time.sleep(min(60, 2 ** attempt * (3 if r.status_code == 429 else 1)))
        except Exception:  # noqa: BLE001
            time.sleep(min(60, 2 ** attempt))
    return None


def chunk_key(tw: str, phrases: List[str]) -> str:
    return hashlib.sha1(json.dumps([tw, phrases], ensure_ascii=False).encode("utf-8")).hexdigest()


def align_all(jobs: List[Dict], cache_path: str, api_key: str, model: str, workers: int) -> Dict[str, Dict]:
    """Alignments for every job ({key, tw, phrases, ...}), from the cache or Gemini."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    cache[item["key"]] = item.get("result", item.get("mapping"))
                except (json.JSONDecodeError, KeyError):
                    continue
    # Re-align if entry is missing or in legacy format (only mapping, no bracketed_twi)
    todo = list({j["key"]: j for j in jobs if j["phrases"] and (
        j["key"] not in cache or not isinstance(cache[j["key"]], dict) or "bracketed_twi" not in cache[j["key"]]
    )}.values())
    print(f"alignment: {len(jobs):,} chunks, {len(jobs) - len(todo):,} valid cached, {len(todo):,} to align with {model}", flush=True)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    done, failed, t0 = 0, 0, time.time()
    with open(cache_path, "a", encoding="utf-8") as cf, \
            concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(align, j["tw"], j["phrases"], api_key, model): j for j in todo}
        for fut in concurrent.futures.as_completed(futures):
            j, res = futures[fut], fut.result()
            if res is None:
                failed += 1
                continue
            cache[j["key"]] = res
            cf.write(json.dumps({"key": j["key"], "result": res}, ensure_ascii=False) + "\n")
            cf.flush()
            done += 1
            if done % 500 == 0:
                print(f"  aligned {done:,}/{len(todo):,} ({done / (time.time() - t0):.1f}/s)", flush=True)
    if failed:
        print(f"  {failed:,} chunks failed; re-run to retry them (their gold terms are empty for now)", flush=True)
    return cache


WATTS = re.compile(r"\b(watts|Watts|WATTS)(?=[A-Za-z])")


def repair_watts(text: str) -> str:
    """The source English has a normalisation glitch: a word-initial "w" became "watts"
    ("wattshich" for "which", "wattsas" for "was"). Put the "w" back; a bare "Watts"/"watts" stays."""
    def fix(m):
        rest = text[m.end():m.end() + 8]
        if m.group(1) == "watts" and rest.lower().startswith("ich"):
            return "wh"
        if m.group(1) == "Watts" and rest.startswith("ich"):
            return "Wh"
        if m.group(1) == "WATTS" or (m.group(1) == "watts" and rest[:1].isupper() and rest.split()[0].isupper()):
            return "W"
        return "W" if m.group(1) == "Watts" else "w"
    return WATTS.sub(fix, text)


def chunk_split(rows, tok, lookup, chunk_tokens, repair=False):
    out, skipped = [], 0
    for r in rows:
        if repair:
            r = {**r, "english": repair_watts(r["english"])}
        en_lines, en_seps = split_lines(r["english"])
        tw_lines, tw_seps = split_lines(r["translation"])
        if len(en_lines) != len(tw_lines):
            skipped += 1
            continue
        lengths = [len(tok(line, add_special_tokens=False).input_ids) for line in en_lines]
        spans = pack_lines(lengths, chunk_tokens)
        for i, (s, e) in enumerate(spans):
            en, tw = join_span(en_lines, en_seps, s, e), join_span(tw_lines, tw_seps, s, e)
            gloss = glossary_for(en, lookup)
            phrases = list(dict.fromkeys(contains(en, g["phrase"]) or g["phrase"] for g in gloss))
            out.append({"row": r, "index": i, "num": len(spans), "en": en, "tw": tw, "gloss": gloss,
                        "phrases": phrases, "key": chunk_key(tw, phrases)})
    return out, skipped


COLUMNS = ["english", "english_pretranslated", "twi", "twi_bracketed"]
BRACKET = re.compile(r"<([^<>\n]+)>")


def pair_brackets(en: str, mapping: Dict[str, Optional[str]], bracketed_twi: str) -> Tuple[str, str, List[Dict]]:
    """Bracket the same terms on both sides. Gemini maps each English phrase to one Twi span and
    brackets every occurrence it uses in the Twi chunk, so for each (phrase, span) we bracket
    min(English occurrences, Twi occurrences) of it, first to last on both sides, and unbracket
    the Twi occurrences that have no English partner. Returns (english, twi, terms used)."""
    twi_count = Counter(BRACKET.findall(bracketed_twi))
    pairs = {}
    for phrase, span in (mapping or {}).items():
        if isinstance(span, str) and phrase.strip() and span.strip() in twi_count:
            pairs.setdefault(" ".join(phrase.split()), span.strip())
    kept, used = Counter(), []
    for phrase in sorted(pairs, key=len, reverse=True):
        span = pairs[phrase]
        rx = re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, phrase.split())) + r"(?!\w)", re.IGNORECASE)
        occ = [m for m in rx.finditer(en) if not re.search(r"<[^<>]*$", en[:m.start()])]
        for m in reversed(occ[:max(0, twi_count[span] - kept[span])]):
            used.append({"phrase": m.group(0), "translation": span})
            en = en[:m.start()] + f"<{span}>" + en[m.end():]
            kept[span] += 1
    seen = Counter()

    def keep_or_strip(m):
        seen[m.group(1)] += 1
        return m.group(0) if seen[m.group(1)] <= kept[m.group(1)] else m.group(1)

    return en, BRACKET.sub(keep_or_strip, bracketed_twi), used[::-1]


def to_row(c, align_result: Optional[Dict]):
    tw = c["tw"]
    en = c["en"]
    mapping = (align_result or {}).get("mapping") if isinstance(align_result, dict) else {}
    bracketed_twi = (align_result or {}).get("bracketed_twi") if isinstance(align_result, dict) else ""

    # bracketed_twi must equal the original Twi once the brackets are removed
    clean_bw = re.sub(r"\s+", " ", (bracketed_twi or "").replace("<", "").replace(">", "")).strip()
    clean_tw = re.sub(r"\s+", " ", tw).strip()
    en_gold, target_bracketed, gold_terms_used = en, tw, []
    if bracketed_twi and clean_bw == clean_tw:
        en_gold, target_bracketed, gold_terms_used = pair_brackets(en, mapping, bracketed_twi)
        if Counter(BRACKET.findall(en_gold)) != Counter(BRACKET.findall(target_bracketed)):
            en_gold, target_bracketed, gold_terms_used = en, tw, []

    return {
        "id": c["row"]["id"], "chunk_index": c["index"], "num_chunks": c["num"],
        "english": en,
        "english_pretranslated": en_gold,
        "twi": tw, "twi_bracketed": target_bracketed,
        "glossary": [{"phrase": g["phrase"], "translation": g["translations"][0], "found": g["found"]}
                     for g in c["gloss"]],
        "gold_terms": gold_terms_used,
        "source": c["row"]["source"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="build/glossary-chunks")
    ap.add_argument("--tokenizer", default="google/umt5-xl")
    ap.add_argument("--dict", default=GLOSSARY_DICT, help="glossary dictionary (one translation per phrase)")
    ap.add_argument("--chunk-tokens", type=int, default=192)
    ap.add_argument("--align-model", default="gemini-3.8-flash")
    ap.add_argument("--align-cache", default="build/gold_alignment_cache.jsonl")
    ap.add_argument("--repair-watts", action="store_true", help="repair the source's \"watts\" glitch (new chunk keys)")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="pilot: align the first N train chunks, print, no output")
    ap.add_argument("--push", default=None, help="HF dataset repo id to push to")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set")
    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    lookup = TwiLookup(args.dict)
    chunks, skipped, n_rows = {}, 0, {}
    for split in (("train",) if args.limit else ("train", "validation", "test")):
        with open(os.path.join(args.data, f"{split}.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if args.limit:
            rows = rows[:args.limit * 2]
        chunks[split], sk = chunk_split(rows, tok, lookup, args.chunk_tokens, args.repair_watts)
        if args.limit:
            chunks[split] = chunks[split][:args.limit]
        skipped += sk
        n_rows[split] = len(rows) - sk
        print(f"{split}: {n_rows[split]:,} docs -> {len(chunks[split]):,} chunks", flush=True)

    cache = align_all([c for cs in chunks.values() for c in cs], args.align_cache, api_key, args.align_model,
                      args.workers)
    splits, stats, full = {}, [], {}
    for split, cs in chunks.items():
        rows = [to_row(c, cache.get(c["key"])) for c in cs]
        full[split] = rows
        splits[split] = Dataset.from_list(rows).select_columns(COLUMNS)
        found = sum(g["found"] for r in rows for g in r["glossary"])
        total = sum(len(r["glossary"]) for r in rows)
        hit = sum(any(g["found"] for g in r["glossary"]) for r in rows)
        gold = sum(len(r["gold_terms"]) for r in rows)
        stats.append(f"| {split} | {n_rows[split]:,} | {len(rows):,} | {hit:,} | {found:,} / {total:,} | "
                     f"{gold:,} / {total:,} |")
        print(f"{split}: gold terms {gold:,} of {total:,} noun phrases ({gold / max(1, total):.1%}), "
              f"{sum(bool(r['gold_terms']) for r in rows) / len(rows):.1%} of chunks with >=1", flush=True)

    if args.limit:
        for r in full["train"][:5]:
            print("\nENGLISH:", r["english"], "\nSOURCE :", r["english_pretranslated"],
                  "\nTARGET :", r["twi_bracketed"], "\nGOLD   :", r["twi"], "\nTERMS  :", [(g["phrase"], g["translation"]) for g in r["gold_terms"]])
        return

    ds = DatasetDict(splits)
    os.makedirs(args.out, exist_ok=True)
    ds.save_to_disk(os.path.join(args.out, "dataset"))
    ex = next(r for r in full["train"] if 3 <= len(r["gold_terms"]) and len(r["english"]) < 300)
    example = "\n".join([f"english:                    {ex['english']}",
                         f"english_pretranslated: {ex['english_pretranslated']}",
                         f"twi_bracketed:              {ex['twi_bracketed']}",
                         f"twi:                        {ex['twi']}"])
    card = CARD.format(source=SOURCE_DATASET, skipped=skipped, chunk_tokens=args.chunk_tokens,
                       dict_size=len(lookup), stats="\n".join(stats), align_model=args.align_model,
                       example=example)
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print(f"Saved to {args.out}", flush=True)

    if args.push:
        from huggingface_hub import HfApi
        ds.push_to_hub(args.push, private=False)
        HfApi().upload_file(path_or_fileobj=os.path.join(args.out, "README.md"), path_in_repo="README.md",
                            repo_id=args.push, repo_type="dataset",
                            commit_message="Add dataset card")
        print(f"Pushed https://huggingface.co/datasets/{args.push}", flush=True)


if __name__ == "__main__":
    main()
