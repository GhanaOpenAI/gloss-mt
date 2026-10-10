"""The lookup_twi_translation tool: an offline English -> Twi dictionary.

The dictionary ships with the package (twi_translate/data/twi_dictionary.tsv.gz) and is
read from disk, so lookups never touch the network. It was exported once from the Gemini
translations in AfriSpeech/shola-machine-translations:

  1. gemini-3.6-flash-one/all-tiers.jsonl.gz  - current, one translation per phrase
  2. gemini-3.6-flash/all-tiers.jsonl.gz      - superseded, up to 3 options per phrase,
                                                 used only for phrases missing from (1)

Rebuild it with:  python -m twi_translate.lookup --rebuild
The same class builds the training data and executes the tool at deployment, so the tool
results the model was trained on match what it receives in production.
"""

import gzip
import json
import os
import re
from typing import Dict, List, Optional

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "twi_dictionary.tsv.gz")

REPO_ID = "AfriSpeech/shola-machine-translations"
PRIMARY_FILE = "gemini-3.6-flash-one/all-tiers.jsonl.gz"
FALLBACK_FILE = "gemini-3.6-flash/all-tiers.jsonl.gz"
LANGUAGE = "twi"
MAX_OPTIONS = 3

_WS = re.compile(r"\s+")


def normalize_phrase(phrase: str) -> str:
    """Lowercase, trim and collapse whitespace so lookups are case-insensitive."""
    return _WS.sub(" ", phrase).strip().lower()


class TwiLookup:
    def __init__(self, path: str = DEFAULT_PATH):
        self.table: Dict[str, List[str]] = {}
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                key, *options = line.rstrip("\n").split("\t")
                if key and options:
                    self.table[key] = options

    def lookup(self, phrase: str) -> Dict[str, object]:
        """Return {"phrase": ..., "translations": [...]}; empty list when not found."""
        key = normalize_phrase(phrase)
        return {"phrase": key, "translations": list(self.table.get(key, []))}

    def lookup_many(self, phrases: List[str]) -> List[Dict[str, object]]:
        seen, results = set(), []
        for p in phrases:
            key = normalize_phrase(p)
            if key and key not in seen:
                seen.add(key)
                results.append(self.lookup(key))
        return results

    def __call__(self, phrases: List[str]) -> List[Dict[str, object]]:
        return self.lookup_many(phrases)

    def __len__(self) -> int:
        return len(self.table)


def build_table_from_hub(revision: Optional[str] = None) -> Dict[str, List[str]]:
    """Download the SHOLA Gemini files from the Hugging Face Hub and merge them."""
    from huggingface_hub import hf_hub_download

    table: Dict[str, List[str]] = {}
    for filename, only_new in ((PRIMARY_FILE, False), (FALLBACK_FILE, True)):
        path = hf_hub_download(REPO_ID, filename, repo_type="dataset", revision=revision)
        existing = set(table) if only_new else set()
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item.get("language") != LANGUAGE:
                    continue
                key = normalize_phrase(item["phrase"])
                text = _WS.sub(" ", item["text"]).strip()
                if not key or not text or key in existing:
                    continue
                options = table.setdefault(key, [])
                if text not in options and len(options) < MAX_OPTIONS:
                    options.append(text)
    return table


def write_table(table: Dict[str, List[str]], path: str = DEFAULT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write(f"# English phrase -> Twi candidates (tab-separated). Source: {REPO_ID}, {PRIMARY_FILE}, "
                f"then {FALLBACK_FILE} for phrases missing from it. CC-BY-NC-4.0.\n")
        for key in sorted(table):
            f.write(key.replace("\t", " ") + "\t" + "\t".join(o.replace("\t", " ") for o in table[key]) + "\n")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Rebuild the bundled Twi dictionary from the HF Hub.")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--out", default=DEFAULT_PATH)
    args = ap.parse_args()
    if args.rebuild:
        table = build_table_from_hub()
        write_table(table, args.out)
        print(f"wrote {len(table):,} phrases to {args.out}")
    else:
        print(f"{len(TwiLookup()):,} phrases in {DEFAULT_PATH}")
