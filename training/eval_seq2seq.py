"""Evaluate umT5 / NLLB / TranslateGemma through the chunk -> phrases -> lookup -> translate pipeline.

Reports the same document-level metrics as eval_checkpoints.py (BLEU, chrF2, CER, length
ratio, GeminiScore when GEMINI_API_KEY is set) plus recall of the dataset's tool_phrases by the spaCy extractor.

  python training/eval_seq2seq.py --model checkpoints/umt5-xl-twi/final --data data/test.jsonl
  python training/eval_seq2seq.py --model facebook/nllb-200-3.3B --no-glossary \\
      --out results/zeroshot_nllb_3.3b_test.json                       # zero-shot baseline
  python training/eval_seq2seq.py --backend translategemma --model google/translategemma-4b-it \\
      --adapter checkpoints/translategemma-4b-twi/adapter
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twi_translate.lookup import normalize_phrase  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["seq2seq", "translategemma"], default="seq2seq")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter (translategemma)")
    ap.add_argument("--no-glossary", action="store_true", help="skip the dictionary (zero-shot baselines)")
    ap.add_argument("--dict", default=None, help="glossary dictionary (default: one translation per phrase)")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--chunk-tokens", type=int, default=192)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None, help="defaults to <model>/eval_<split>.json")
    args = ap.parse_args()

    import sacrebleu

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = rows[:args.limit] if args.limit > 0 else rows

    from twi_translate import TwiLookup
    from twi_translate.seq2seq import GLOSSARY_DICT
    lookup = None if args.no_glossary else TwiLookup(args.dict or GLOSSARY_DICT)
    if args.backend == "translategemma":
        from twi_translate.translategemma import TranslateGemmaTranslator
        tr = TranslateGemmaTranslator(args.model, adapter=args.adapter, lookup=lookup, chunk_tokens=args.chunk_tokens,
                                      batch_size=args.batch_size, glossary=not args.no_glossary)
    else:
        from twi_translate.seq2seq import Seq2SeqTranslator
        tr = Seq2SeqTranslator(args.model, lookup=lookup, chunk_tokens=args.chunk_tokens, num_beams=args.num_beams,
                               batch_size=args.batch_size, glossary=not args.no_glossary)
    t0 = time.time()
    preds = tr.translate_many([r["english"] for r in rows])
    elapsed = time.time() - t0
    hyps, refs = [p["translation"] for p in preds], [r["translation"] for r in rows]

    gold = [{normalize_phrase(p) for p in r["tool_phrases"]} for r in rows]
    pred = [{normalize_phrase(p) for p in p_["phrases"]} for p_ in preds]
    hit = sum(len(g & p) for g, p in zip(gold, pred))

    import editdistance
    metrics = {
        "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
        "chrf2": round(sacrebleu.corpus_chrf(hyps, [refs], beta=2).score, 2),
        "cer": round(100.0 * sum(editdistance.eval(r, h) for r, h in zip(refs, hyps))
                     / max(1, sum(len(r) for r in refs)), 2),
        "length_ratio": round(sum(len(h.split()) for h in hyps) / max(1, sum(len(r.split()) for r in refs)), 3),
        "phrase_recall": round(hit / max(1, sum(map(len, gold))), 3),
        "phrase_precision": round(hit / max(1, sum(map(len, pred))), 3),
        "docs": len(rows),
        "seconds": round(elapsed, 1),
    }
    if os.environ.get("GEMINI_API_KEY"):
        try:
            import gemini_score
            model = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
            res = gemini_score.score(hyps, refs, api_key=os.environ["GEMINI_API_KEY"], model=model)
            metrics.update(gemini_score=round(res.score, 2), gemini_tier=res.tier)
            res = gemini_score.score(hyps, [r["english"] for r in rows],
                                     api_key=os.environ["GEMINI_API_KEY"], model=model)
            metrics["gemini_score_src"] = round(res.score, 2)
        except Exception as e:  # noqa: BLE001
            print(f"GeminiScore skipped: {e}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)

    split = os.path.splitext(os.path.basename(args.data))[0]
    out_dir = args.adapter or args.model
    out = args.out or (os.path.join(out_dir, f"eval_{split}.json") if os.path.isdir(out_dir)
                       else os.path.join("results", f"eval_{out_dir.replace('/', '_')}_{split}.json"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "args": vars(args),
                   "predictions": [{"id": r["id"], "hyp": h, **{k: v for k, v in p.items() if k != "translation"}}
                                   for r, h, p in zip(rows, hyps, preds)]}, f, ensure_ascii=False, indent=1)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
