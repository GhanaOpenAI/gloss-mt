"""Full fine-tune of an OPUS-MT (MarianMT) model, Helsinki-NLP/opus-mt-en-tw by default, on the
chunked English-Twi dataset ghanaopenai/gloss-mt-parallel-en-twi, with the same inputs as
train_seq2seq.py (load_chunks: english_pretranslated -> twi_bracketed; ref = plain twi).

Marian is one language pair per model, so there is no task prefix and no language code. Its
length limit is 512 tokens, and its vocabulary cuts the pretranslated chunks into more tokens than
NLLB's (median 293, 14% of train inputs over 512), so a chunk over --max-src / --max-tgt is cut at
line boundaries (English and Twi chunks are line-aligned) into pieces that fit; a piece is kept only
when both sides carry the same bracketed terms (split_to_fit). Validation is split the same way.
"\\n" is added as a token (prepare_tokenizer) as for NLLB. fp32 weights with bf16 autocast, AdamW.

Usage (H200; --data is the HF dataset or a local save_to_disk copy of it):
  python training/train_opus_mt.py --out checkpoints/opus-mt-en-tw-twi
  python training/train_opus_mt.py ... --max-steps 20   # memory/throughput pilot
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Callable, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_eval import batched_generate, sample_examples, with_gen_metrics  # noqa: E402
from train_seq2seq import DATASET, disk_guard, load_chunks  # noqa: E402
from twi_translate.seq2seq import clean_output, join_span, prepare_tokenizer, split_lines  # noqa: E402

BRACKET = re.compile(r"<([^<>\n]+)>")
COLS = ("input", "tgt", "chunk", "ref")


def split_to_fit(e: Dict, fits: Callable[[str, str], bool]) -> List[Dict]:
    """[e] when fits(input, tgt), else line-aligned pieces of it that fit (consecutive lines packed
    greedily). A piece whose two sides differ in bracketed terms, or a single line that does not
    fit, is dropped; [] when the columns do not split into the same number of lines."""
    if fits(e["input"], e["tgt"]):
        return [e]
    split = {c: split_lines(e[c]) for c in COLS}
    n = len(split["input"][0])
    if any(len(split[c][0]) != n for c in COLS):
        return []

    def piece(s, t):
        return {c: join_span(*split[c], s, t) for c in COLS}

    def ok(s, t):
        p = piece(s, t)
        return fits(p["input"], p["tgt"])

    out, start = [], 0
    while start < n:
        if not ok(start, start + 1):
            start += 1  # one line too long on its own: drop it
            continue
        end = start + 1
        while end < n and ok(start, end + 1):
            end += 1
        p = piece(start, end)
        if Counter(BRACKET.findall(p["input"])) == Counter(BRACKET.findall(p["tgt"])):
            out.append(p)
        start = end
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Helsinki-NLP/opus-mt-en-tw")
    ap.add_argument("--data", default=DATASET, help="HF dataset id or local save_to_disk directory")
    ap.add_argument("--out", default="checkpoints/opus-mt-en-tw-twi")
    ap.add_argument("--max-src", type=int, default=512)
    ap.add_argument("--max-tgt", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=30)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--early-stopping-patience", type=int, default=5)
    ap.add_argument("--save-total-limit", type=int, default=2,
                    help="2 keeps the best and the most recent checkpoint")
    ap.add_argument("--min-free-gb", type=float, default=40, help="stop cleanly below this much free disk")
    ap.add_argument("--gen-eval-chunks", type=int, default=200,
                    help="validation pieces translated at each eval for BLEU/chrF2 and the "
                         "composite that selects the best checkpoint (0 = select on eval_loss only)")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback, MarianMTModel,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained(args.model)
    prepare_tokenizer(tok)

    def fits(src: str, tgt: str) -> bool:
        return (len(tok(src).input_ids) <= args.max_src
                and len(tok(text_target=tgt).input_ids) <= args.max_tgt)

    ds, gen_examples = {}, []
    for split in ("train", "validation"):
        chunks = load_chunks(args.data, split)
        ex = [p for e in chunks for p in split_to_fit(e, fits)]
        enc = tok([e["input"] for e in ex], text_target=[e["tgt"] for e in ex])
        ds[split] = Dataset.from_dict({k: enc[k] for k in ("input_ids", "attention_mask", "labels")})
        print(f"{split}: {len(chunks)} chunks -> {len(ex)} pieces within {args.max_src}/{args.max_tgt} tokens",
              flush=True)
        if split == "validation" and args.gen_eval_chunks > 0:
            gen_examples = sample_examples(ex, args.gen_eval_chunks)

    model = MarianMTModel.from_pretrained(args.model)
    if len(tok) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok))
    model.config.use_cache = False

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_steps=500,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        tf32=True,
        train_sampling_strategy="group_by_length",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="composite" if gen_examples else "eval_loss",
        greater_is_better=bool(gen_examples),
        report_to="none",
        dataloader_num_workers=2,
    )

    def gen_fn(m, examples):
        outs = batched_generate(m, tok, [e["input"] for e in examples], args.batch_size, args.max_tgt - 2)
        return [clean_output(o) for o in outs]

    trainer_cls = with_gen_metrics(Seq2SeqTrainer)
    trainer_cls.gen_examples, trainer_cls.gen_fn = gen_examples, staticmethod(gen_fn)
    trainer = trainer_cls(
        model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["validation"],
        data_collator=DataCollatorForSeq2Seq(tok, model=model, padding=True),
        processing_class=tok,
        callbacks=[disk_guard(args.out, args.min_free_gb)]
        + ([EarlyStoppingCallback(args.early_stopping_patience)] if args.early_stopping_patience > 0 else []),
    )
    has_ckpt = os.path.isdir(args.out) and any(d.startswith("checkpoint-") for d in os.listdir(args.out))
    trainer.train(resume_from_checkpoint=True if has_ckpt else None)
    print(trainer.evaluate(), flush=True)

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    with open(os.path.join(args.out, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved model to {final}", flush=True)


if __name__ == "__main__":
    main()
