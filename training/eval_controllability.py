"""Controllability test: does the translation follow the dictionary, and follow an edited one?

Translates a fixed sample of validation chunks (English -> Twi; --jsonl: a human-translated set
such as the Asante Twi Bible passages of scripts/build_bible_testset.py) through the GLOSS-MT pipeline
three times with the same model, changing only the dictionary:

  dict   the GLOSS-MT dictionary (what is deployed)
  none   no dictionary (glossary=False): how often the same Twi terms appear on their own
  swap   a counterfactual dictionary: every dictionary phrase found in the chunk gets a different
         Twi term, taken from the other terms of the sample, that occurs neither in the source nor
         in the dict/none outputs; so it can only reach the output through the dictionary

and reports, over the dictionary terms actually inserted into the inputs:

  adherence        share of inserted terms that appear in the output (dict)
  natural_rate     share of the same terms that appear without the dictionary (none)
  control_gain     adherence - natural_rate: what the dictionary adds
  swap_adherence   share of the swapped-in terms that appear in the output (swap)
  swap_leak        share of the original terms that still appear after the swap
  BLEU / chrF2 against the reference for dict and none, and chrF2 of swap against dict
  (how much of the rest of the translation stays the same when only the terms change).

A term "appears" when its words occur as whole words, case-insensitively. Swapped terms are other
real Twi terms, usually wrong for the phrase: the test measures whether the dictionary steers the
output, not translation quality.

  python training/eval_controllability.py --model checkpoints/nllb-3.3b-twi/final --name gloss-mt-nllb-3.3b
  python training/eval_controllability.py --backend translategemma --model google/translategemma-4b-it \\
      --adapter checkpoints/translategemma-4b-twi/adapter --name gloss-mt-translategemma-4b
  python training/eval_controllability.py --model facebook/nllb-200-3.3B --name nllb-200-3.3b-base
"""

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_eval import sample_examples, text_metrics  # noqa: E402
from train_seq2seq import DATASET, load_chunks  # noqa: E402
from twi_translate.lookup import TwiLookup, normalize_phrase  # noqa: E402
from twi_translate.seq2seq import GLOSSARY_DICT, pretranslate  # noqa: E402


def appears(term: str, text: str) -> bool:
    """The term's words, in order, as whole words of the text (case-insensitive)."""
    words = term.lower().split()
    return bool(words) and re.search(r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)",
                                     text.lower()) is not None


class EditedLookup:
    """A TwiLookup whose translations for the given phrases are replaced (phrase -> [term])."""

    def __init__(self, base: TwiLookup, edits: Dict[str, str]):
        self.base, self.edits = base, edits

    def lookup_many(self, phrases: List[str]) -> List[Dict]:
        out = self.base.lookup_many(phrases)
        for r in out:
            if r["phrase"] in self.edits:
                r["translations"] = [self.edits[r["phrase"]]]
        return out


def inserted_terms(text: str, results: List[Dict]) -> Dict[str, str]:
    """phrase -> Twi term for the found phrases that pretranslate actually replaces in the text."""
    out = {}
    for r in results:
        if r.get("translations") and r["phrase"] not in out and pretranslate(text, [r]) != text:
            out[normalize_phrase(r["phrase"])] = r["translations"][0]
    return out


def rate(hits: int, total: int) -> float:
    return round(100.0 * hits / max(1, total), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["seq2seq", "translategemma"], default="seq2seq")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter (translategemma)")
    ap.add_argument("--name", required=True, help="results/controllability/<name>.json")
    ap.add_argument("--data", default=DATASET, help="HF dataset id or local save_to_disk directory")
    ap.add_argument("--jsonl", default=None,
                    help="test on these {english, twi} items instead of the validation split, e.g. the "
                         "human-translated Bible set of scripts/build_bible_testset.py")
    ap.add_argument("--n", type=int, default=300, help="validation chunks (fixed seeded sample)")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.jsonl:
        with open(args.jsonl, encoding="utf-8") as f:
            rows = [{"chunk": d["english"], "ref": d["twi"]} for d in map(json.loads, f) if d]
    else:
        rows = sample_examples(load_chunks(args.data, "validation"), args.n, args.seed)
    srcs, refs = [r["chunk"] for r in rows], [r["ref"] for r in rows]
    lookup = TwiLookup(GLOSSARY_DICT)
    if args.backend == "translategemma":
        from twi_translate.translategemma import TranslateGemmaTranslator
        tr = TranslateGemmaTranslator(args.model, adapter=args.adapter, lookup=lookup, device=args.device,
                                      batch_size=args.batch_size)
    else:
        from twi_translate.seq2seq import Seq2SeqTranslator
        tr = Seq2SeqTranslator(args.model, lookup=lookup, device=args.device, num_beams=args.num_beams,
                               batch_size=args.batch_size)

    def run(glossary: bool, lkp) -> List[Dict]:
        tr.glossary, tr.lookup = glossary, lkp
        t0 = time.time()
        out = tr.translate_many(srcs)
        print(f"  {len(srcs)} chunks in {time.time() - t0:.0f}s", flush=True)
        return out

    print("dict", flush=True)
    dict_out = run(True, lookup)
    terms = [inserted_terms(s, o["lookup_results"]) for s, o in zip(srcs, dict_out)]
    print("none", flush=True)
    none_out = run(False, lookup)
    hyp_dict = [o["translation"] for o in dict_out]
    hyp_none = [o["translation"] for o in none_out]

    # One counterfactual term per phrase, the same wherever the phrase occurs: another term of the
    # sample that is absent from the source and from both outputs of every chunk the phrase is in.
    pool = sorted({t for ts in terms for t in ts.values()})
    rng = random.Random(args.seed)
    edits = {}
    for phrase in sorted({p for ts in terms for p in ts}):
        where = [i for i, ts in enumerate(terms) if phrase in ts]
        orig = terms[where[0]][phrase]
        cands = list(pool)
        rng.shuffle(cands)
        new = next((t for t in cands if normalize_phrase(t) != normalize_phrase(orig)
                    and not appears(orig, t) and not appears(t, orig)
                    and not any(appears(t, srcs[i]) or appears(t, hyp_dict[i]) or appears(t, hyp_none[i])
                                for i in where)), None)
        if new:
            edits[phrase] = new
    print(f"swap ({len(edits)} phrases edited)", flush=True)
    swap_out = run(True, EditedLookup(lookup, edits))
    hyp_swap = [o["translation"] for o in swap_out]

    n_terms = adh = nat = n_swap = swap_adh = leak = 0
    by_len = {}
    examples = []
    for i, ts in enumerate(terms):
        ex = {"src": srcs[i], "ref": refs[i], "dict": hyp_dict[i], "none": hyp_none[i], "swap": hyp_swap[i], "terms": []}
        for phrase, term in ts.items():
            a, b = appears(term, hyp_dict[i]), appears(term, hyp_none[i])
            n_terms, adh, nat = n_terms + 1, adh + a, nat + b
            k = "1 word" if len(term.split()) == 1 else "2+ words"
            by_len.setdefault(k, [0, 0])
            by_len[k][0] += 1
            by_len[k][1] += a
            t = {"phrase": phrase, "term": term, "in_dict": a, "in_none": b}
            if phrase in edits:
                new = edits[phrase]
                s, lk = appears(new, hyp_swap[i]), appears(term, hyp_swap[i])
                n_swap, swap_adh, leak = n_swap + 1, swap_adh + s, leak + lk
                t.update(swapped_to=new, in_swap=s, orig_in_swap=lk)
            ex["terms"].append(t)
        examples.append(ex)

    import sacrebleu

    m_dict, m_none = text_metrics(hyp_dict, refs), text_metrics(hyp_none, refs)
    metrics = {
        "chunks": len(rows),
        "inserted_terms": n_terms,
        "adherence": rate(adh, n_terms),
        "natural_rate": rate(nat, n_terms),
        "control_gain": round(rate(adh, n_terms) - rate(nat, n_terms), 2),
        "adherence_by_term_length": {k: {"terms": v[0], "adherence": rate(v[1], v[0])} for k, v in sorted(by_len.items())},
        "swapped_terms": n_swap,
        "swap_adherence": rate(swap_adh, n_swap),
        "swap_leak": rate(leak, n_swap),
        "dict_bleu": m_dict["bleu"], "dict_chrf2": m_dict["chrf2"],
        "none_bleu": m_none["bleu"], "none_chrf2": m_none["chrf2"],
        "swap_vs_dict_chrf2": round(sacrebleu.corpus_chrf(hyp_swap, [hyp_dict], beta=2).score, 2),
    }
    print(json.dumps(metrics, indent=2), flush=True)

    out = os.path.join("results", "controllability", f"{args.name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "args": vars(args), "examples": examples}, f, ensure_ascii=False, indent=1)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
