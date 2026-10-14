"""Build (and optionally push) the chunked, glossary-annotated English-Twi parallel dataset.

Same chunks and glossaries the seq2seq / TranslateGemma pipeline trains on
(twi_translate.seq2seq): each document of data/{train,validation,test}.jsonl is split into
its line-aligned English/Twi lines, consecutive lines are packed into chunks of at most
--chunk-tokens umT5 tokens, and each chunk gets the dictionary results for its spaCy noun
phrases, plus the chunk with those phrases pretranslated in place: with dictionary terms
(inference input) and with the terms its gold Twi chunk uses (training input, from
training/gold_glossary.py). One row per chunk.

  python scripts/build_chunked_dataset.py --out build/glossary-chunks
  python scripts/build_chunked_dataset.py --out build/glossary-chunks \\
      --push ghanaopenai/twi-translation-glossary-chunks
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))
from twi_translate import TwiLookup  # noqa: E402
from twi_translate.seq2seq import (  # noqa: E402
    GLOSSARY_DICT, glossary_for, join_span, pack_lines, pretranslate, split_lines)
from gold_glossary import gold_glossary, reasoning_pairs  # noqa: E402

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
lookups for its key noun phrases, and a version of the English with those phrases already
**pretranslated in place**. It is built for training translation models that get dictionary
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
4. `english_pretranslated` replaces every occurrence of each found phrase in the chunk with its
   Twi translation (case-insensitive, whole words, longest phrase first; all-caps and
   capitalised matches keep that casing). Phrases not in the dictionary, or not written
   verbatim in the chunk, stay English. The model learns to translate the rest around them.
   This is what a model sees at inference.
5. `english_pretranslated_gold` uses the same noun phrases, but each phrase's Twi is what the
   **gold Twi chunk actually uses**: an `'English' -> 'Twi'` pair quoted in the source
   example's reasoning trace (written with the reference in view), else a dictionary
   candidate found in the gold chunk; other phrases stay English (`gold_terms`). Plain
   dictionary terms appear in the reference only ~31% of the time, so training on
   `english_pretranslated` teaches a model to overrule inserted terms; training on
   `english_pretranslated_gold` teaches it to keep them. Since it is built from the
   reference, use it for **training only**; evaluate on `english_pretranslated`.

The build script is `scripts/build_chunked_dataset.py` in
[GhanaOpenAI/gloss-mt](https://github.com/GhanaOpenAI/gloss-mt), which uses the same
`twi_translate.seq2seq` helpers as training and inference.

## Columns

| column | content |
|---|---|
| `id` | source document id in `{source}` |
| `chunk_index` / `num_chunks` | position of the chunk within its document |
| `english` | English chunk |
| `english_pretranslated` | English chunk with its dictionary phrases replaced by their Twi translations (inference / evaluation input) |
| `english_pretranslated_gold` | English chunk with its phrases replaced only by Twi that the gold chunk uses (training input) |
| `twi` | gold Twi chunk (human reference) |
| `glossary` | list of `{{phrase, translation, found}}`; when `found` is false the phrase is not in the dictionary and `translation` is the English phrase itself |
| `gold_terms` | list of `{{phrase, translation, source}}` used for `english_pretranslated_gold`; `source` is `reasoning` or `dictionary` |
| `source` | origin of the document (`pristine`, `ghana_chat_en2tw`) |

Joining a document's chunks in `chunk_index` order (with the original line breaks) gives back
the full document.

## Statistics

| split | documents | chunks | chunks with ≥1 glossary hit | glossary phrases (found / total) | gold terms (reasoning / dictionary) |
|---|---|---|---|---|---|
{stats}

## Example input format

Models are trained on `english_pretranslated_gold` and translate `english_pretranslated` at
inference (with `translate English to Twi: ` in front for umT5; NLLB uses its
eng_Latn/twi_Latn codes instead):

```
english:                    On the first day in the month of May, it is a special day for women.
english_pretranslated_gold: On the first da in the bosome of May, it is a da titiriw for mmaa.
twi:                        Ɛda a ɛdi kan wɔ May bosome no mu no, ɛyɛ da titiriw ma mmaa.
```

## License

CC BY-NC 4.0, same as the source dataset.
"""


def build_split(rows, tok, lookup, chunk_tokens, multi):
    out, skipped = [], 0
    for r in rows:
        pairs = reasoning_pairs(r.get("reasoning", ""))
        en_lines, en_seps = split_lines(r["english"])
        tw_lines, tw_seps = split_lines(r["translation"])
        if len(en_lines) != len(tw_lines):
            skipped += 1
            continue
        lengths = [len(tok(line, add_special_tokens=False).input_ids) for line in en_lines]
        spans = pack_lines(lengths, chunk_tokens)
        for i, (s, e) in enumerate(spans):
            en = join_span(en_lines, en_seps, s, e)
            tw = join_span(tw_lines, tw_seps, s, e)
            gloss = glossary_for(en, lookup)
            gold = gold_glossary(en, tw, pairs, lookup, multi)
            out.append({
                "id": r["id"], "chunk_index": i, "num_chunks": len(spans),
                "english": en, "english_pretranslated": pretranslate(en, gloss),
                "english_pretranslated_gold": pretranslate(en, gold),
                "twi": tw,
                "glossary": [{"phrase": g["phrase"], "translation": g["translations"][0],
                              "found": g["found"]}
                             for g in gloss],
                "gold_terms": [{"phrase": g["phrase"], "translation": g["translations"][0], "source": g["source"]}
                               for g in gold],
                "source": r["source"],
            })
    return out, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="build/glossary-chunks")
    ap.add_argument("--tokenizer", default="google/umt5-xl")
    ap.add_argument("--dict", default=GLOSSARY_DICT, help="glossary dictionary (one translation per phrase)")
    ap.add_argument("--chunk-tokens", type=int, default=192)
    ap.add_argument("--push", default=None, help="HF dataset repo id to push to")
    args = ap.parse_args()

    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    lookup = TwiLookup(args.dict)
    multi = TwiLookup()  # multi-candidate dictionary, for matching terms against the gold chunk
    splits, stats, skipped = {}, [], 0
    for split in ("train", "validation", "test"):
        with open(os.path.join(args.data, f"{split}.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        chunks, sk = build_split(rows, tok, lookup, args.chunk_tokens, multi)
        skipped += sk
        splits[split] = Dataset.from_list(chunks)
        found = sum(1 for c in chunks for g in c["glossary"] if g["found"])
        total = sum(len(c["glossary"]) for c in chunks)
        hit = sum(1 for c in chunks if any(g["found"] for g in c["glossary"]))
        src = Counter(g["source"] for c in chunks for g in c["gold_terms"])
        stats.append(f"| {split} | {len(rows) - sk:,} | {len(chunks):,} | {hit:,} | {found:,} / {total:,} | "
                     f"{src['reasoning']:,} / {src['dictionary']:,} |")
        print(split, len(rows), "docs ->", len(chunks), "chunks", dict(Counter(c["source"] for c in chunks)), flush=True)

    ds = DatasetDict(splits)
    os.makedirs(args.out, exist_ok=True)
    ds.save_to_disk(os.path.join(args.out, "dataset"))
    card = CARD.format(source=SOURCE_DATASET, skipped=skipped, chunk_tokens=args.chunk_tokens,
                       dict_size=len(lookup), stats="\n".join(stats))
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
