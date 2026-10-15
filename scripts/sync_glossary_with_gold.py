"""Put every aligned (English phrase -> Twi span) pair of the dataset into the glossary dictionary.

Rebuilds the train/validation/test chunks exactly as build_chunked_dataset.py does (cached Gemini
alignment, the same occurrence pairing, no new API calls) and collects the bracketed pairs
(phrase as written in the English chunk, lowercased -> Twi span). The dictionary keeps one
translation per phrase: a phrase already there is overwritten with its aligned span, a new
phrase is added with it. A phrase aligned to different spans in different chunks is ambiguous
and left out (an existing dictionary entry for it is not changed).

  python scripts/sync_glossary_with_gold.py            # report only
  python scripts/sync_glossary_with_gold.py --write    # rewrite --dict (old file kept as .bak)
"""

import argparse
import gzip
import json
import os
import shutil
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_chunked_dataset as b  # noqa: E402
from twi_translate import TwiLookup  # noqa: E402
from twi_translate.lookup import normalize_phrase  # noqa: E402
from twi_translate.seq2seq import GLOSSARY_DICT  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--dict", default=GLOSSARY_DICT)
    ap.add_argument("--cache", default="build/gold_alignment_cache.jsonl")
    ap.add_argument("--tokenizer", default="google/umt5-xl")
    ap.add_argument("--repair-watts", action="store_true", help="apply the source-glitch repair (changes chunk keys)")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok, lookup = AutoTokenizer.from_pretrained(args.tokenizer), TwiLookup(args.dict)
    cache = {}
    with open(args.cache, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cache[item["key"]] = item.get("result")

    pairs = defaultdict(Counter)
    for split in ("train", "validation", "test"):
        with open(os.path.join(args.data, f"{split}.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        chunks, _ = b.chunk_split(rows, tok, lookup, 192, args.repair_watts)
        n_terms = 0
        for c in chunks:
            for t in b.to_row(c, cache.get(c["key"]))["gold_terms"]:
                pairs[normalize_phrase(t["phrase"])][t["translation"]] += 1
                n_terms += 1
        print(f"{split}: {len(chunks):,} chunks ({sum(c['key'] not in cache for c in chunks):,} uncached), "
              f"{n_terms:,} bracketed terms", flush=True)

    aligned = {p: next(iter(s)) for p, s in pairs.items() if len(s) == 1}
    ambiguous = len(pairs) - len(aligned)
    table = {k: v[0] for k, v in lookup.table.items()}
    added = sum(p not in table for p in aligned)
    same = sum(table.get(p) == t for p, t in aligned.items())
    overwritten = len(aligned) - added - same
    print(f"{len(pairs):,} aligned phrases | ambiguous (more than one span, left out): {ambiguous:,} | "
          f"used: {len(aligned):,} -> already with that translation: {same:,}, overwritten: {overwritten:,}, "
          f"added: {added:,}", flush=True)
    print("sample ambiguous  :", [(p, dict(s)) for p, s in pairs.items() if len(s) > 1][:5])
    print("sample overwritten:", [(p, table[p], t) for p, t in aligned.items() if p in table and table[p] != t][:8])
    print("sample added      :", [(p, t) for p, t in aligned.items() if p not in table][:8], flush=True)
    if not args.write:
        return

    table.update(aligned)
    shutil.copy(args.dict, args.dict + ".bak")
    with gzip.open(args.dict, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write("# English phrase -> Twi translation (tab-separated), exactly one per phrase. CC-BY-NC-4.0.\n")
        f.write(f"# Every phrase aligned in the gloss-mt dataset uses its aligned Twi span "
                f"({overwritten:,} overwritten, {added:,} added; {ambiguous:,} ambiguous phrases left out).\n")
        for key in sorted(table):
            f.write(key.replace("\t", " ") + "\t" + table[key].replace("\t", " ").replace("\n", " ") + "\n")
    check = TwiLookup(args.dict)
    wrong = sum(check.table.get(p) != [t] for p, t in aligned.items())
    print(f"Wrote {args.dict}: {len(table):,} entries; aligned phrases not matching their span: {wrong}", flush=True)


if __name__ == "__main__":
    main()
