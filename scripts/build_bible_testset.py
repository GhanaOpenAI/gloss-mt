"""Human-translated English -> Asante Twi test set from the Bible (AfriSpeech/africa-corpus,
built with https://github.com/AfriSpeech/africa-corpus-builder), for tests that need real human
references (training/eval_controllability.py --jsonl).

Twi: "Asante Twi Nkwa Asɛm" (YouVersion 2094, Asante forms such as sɛdeɛ/deɛ; a modern,
meaning-based translation). English: Good News Translation (YouVersion 68), the meaning-based
English version of the corpus. Each item is a passage of --verses consecutive verses of one
chapter, all present and non-empty on both sides; --n passages are sampled with a fixed seed.
The text comes from YouVersion (see its terms of use): the output stays out of the repository.

  python scripts/build_bible_testset.py --out build/bible/asante_gnt_300.jsonl
"""

import argparse
import csv
import json
import os
import random
import re

REPO = "AfriSpeech/africa-corpus"
TWI = "Akan_aka_v2094.csv"  # Asante Twi Nkwa Asɛm
ENGLISH = "reference_caches/English_en_v68.csv"  # Good News Translation


def verses(filename: str) -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, filename, repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        rows = csv.reader(f)
        next(rows)
        return {r[0]: " ".join(r[-1].split()) for r in rows if r and r[-1].strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="build/bible/asante_gnt_300.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--verses", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    twi, en = verses(TWI), verses(ENGLISH)
    key = re.compile(r"^([1-3]?[A-Z]+)\.(\d+)\.(\d+)$")
    chapters = {}
    for k in twi:
        m = key.match(k)
        if m and k in en:
            chapters.setdefault((m.group(1), int(m.group(2))), []).append(int(m.group(3)))
    starts = []
    for (book, ch), vs in sorted(chapters.items()):
        have = set(vs)
        starts += [(book, ch, v) for v in sorted(vs) if all(v + i in have for i in range(args.verses))]
    # Non-overlapping passages: sample, then drop any that overlaps one already taken.
    rng = random.Random(args.seed)
    rng.shuffle(starts)
    taken, items = set(), []
    for book, ch, v in starts:
        keys = [f"{book}.{ch}.{v + i}" for i in range(args.verses)]
        if taken & set(keys):
            continue
        taken.update(keys)
        items.append({"id": f"{book}.{ch}.{v}-{v + args.verses - 1}",
                      "english": " ".join(en[k] for k in keys), "twi": " ".join(twi[k] for k in keys)})
        if len(items) == args.n:
            break
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in sorted(items, key=lambda x: x["id"]):
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} passages of {args.verses} verses -> {args.out}")


if __name__ == "__main__":
    main()
