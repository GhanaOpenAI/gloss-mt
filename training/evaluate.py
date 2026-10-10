"""Evaluate a deployed model through the real runtime (twi_translate.TwiTranslator).

Every test example goes through the full deployment loop: model -> tool call -> lookup
-> tool result -> translation. Reports tool-call rate, phrase recall against the spaCy
reference phrases, and chrF++ / BLEU against the human Twi reference. With --no-tools
it instead asks for a direct translation (baseline for the base model).

  vllm serve checkpoints/twi-tool-sft --served-model-name twi --port 8300 \\
      --enable-auto-tool-choice --tool-call-parser hermes
  python training/evaluate.py --base-url http://localhost:8300/v1 --model twi --n 200
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twi_translate import OpenAIBackend, TwiLookup, TwiTranslator  # noqa: E402


def source_text(ex):
    return ex["messages"][1]["content"].split("\n\n", 1)[1]


def ref_phrases(ex):
    for m in ex["messages"]:
        for c in m.get("tool_calls") or []:
            return set(c["function"]["arguments"]["phrases"])
    return set()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-tools", action="store_true", help="baseline: direct translation, no tool")
    ap.add_argument("--out", default=None, help="write per-example predictions (jsonl)")
    args = ap.parse_args()

    import sacrebleu

    examples = [json.loads(l) for l in open(args.data, encoding="utf-8")][: args.n]
    backend = OpenAIBackend(args.base_url, args.model)
    translator = TwiTranslator(backend, TwiLookup())

    def run(ex):
        text = source_text(ex)
        if args.no_tools:
            msgs = [{"role": "user", "content": f"Translate the following text into Twi. "
                                                f"Reply with only the translation.\n\n{text}"}]
            hyp, _ = backend.chat(msgs, None)
            return {"hyp": hyp, "phrases": set(), "called": False}
        out = translator.translate(text)
        phrases = set()
        for l in out["lookups"]:
            phrases |= set(l["call"]["arguments"].get("phrases") or [])
        return {"hyp": out["translation"], "phrases": phrases, "called": bool(out["lookups"])}

    with ThreadPoolExecutor(args.workers) as pool:
        preds = list(pool.map(run, examples))

    refs = [ex["messages"][-1]["content"] for ex in examples]
    hyps = [p["hyp"] for p in preds]
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    expected = [ref_phrases(ex) for ex in examples]
    with_ref = [(p, e) for p, e in zip(preds, expected) if e]
    report = {
        "model": args.model, "mode": "no-tools" if args.no_tools else "tool-calling", "n": len(examples),
        "chrf++": round(chrf, 2), "bleu": round(bleu, 2),
        "tool_call_rate": round(sum(p["called"] for p in preds) / len(preds), 4),
        "phrase_recall": round(sum(len(p["phrases"] & e) / len(e) for p, e in with_ref) / max(len(with_ref), 1), 4),
        "phrase_precision": round(sum(len(p["phrases"] & e) / max(len(p["phrases"]), 1) for p, e in with_ref)
                                  / max(len(with_ref), 1), 4),
        "empty_outputs": sum(1 for h in hyps if not h.strip()),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for ex, p in zip(examples, preds):
                f.write(json.dumps({"source": source_text(ex), "reference": ex["messages"][-1]["content"],
                                    "hypothesis": p["hyp"], "phrases": sorted(p["phrases"])},
                                   ensure_ascii=False) + "\n")
        with open(os.path.splitext(args.out)[0] + ".report.json", "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
