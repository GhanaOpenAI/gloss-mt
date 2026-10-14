"""Full or LoRA fine-tune of umT5 or NLLB on the Twi dataset as chunked, glossary-conditioned translation.

Each document is split into its (English/Twi line-aligned) lines, consecutive lines are packed
into chunks of at most --chunk-tokens English tokens, and every chunk yields one example:
  "translate English to Twi: <chunk with its dictionary phrases pretranslated in place>" -> Twi chunk
The phrases come from spaCy noun phrases + TwiLookup (glossary_for) and are substituted by
translate_input, exactly as twi_translate.seq2seq does at inference. NLLB (detected from its tokenizer) gets
eng_Latn -> twi_Latn language codes instead of the "translate English to Twi: " prefix.
Documents whose line counts differ between English and Twi are skipped (5 of 15k).

Full fine-tune: bf16 autocast over fp32 weights, Adafactor. --lora-r N instead trains LoRA
adapters (bf16 base, AdamW) on every attention/FFN projection plus the embedding row of the added
"\\n" token; checkpoints hold the adapter and final/ the merged model.

Usage (H200):
  python training/train_seq2seq.py --model google/umt5-xl --data data \\
      --out checkpoints/umt5-xl-twi
  python training/train_seq2seq.py --model facebook/nllb-200-3.3B --data data \\
      --out checkpoints/nllb-3.3b-twi --lr 1e-4
  python training/train_seq2seq.py --model facebook/nllb-200-3.3B --data data \\
      --out checkpoints/nllb-3.3b-twi-lora --lora-r 32 --lr 2e-4
  python training/train_seq2seq.py ... --max-steps 20   # memory/throughput pilot
"""

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twi_translate import TwiLookup  # noqa: E402
from twi_translate.seq2seq import (  # noqa: E402
    GLOSSARY_DICT, TRANSLATE_PREFIX, clean_output, glossary_for, join_span, load_seq2seq, nllb_langs, pack_lines,
    prepare_tokenizer, split_lines, translate_input)

LORA_TARGETS = r".*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2|q|k|v|o|wi_0|wi_1|wo)$"  # NLLB and umT5

from gen_eval import batched_generate, sample_examples, with_gen_metrics  # noqa: E402


def build_examples(rows: List[Dict], tok, lookup: TwiLookup, chunk_tokens: int,
                   prefix: str = TRANSLATE_PREFIX) -> List[Dict]:
    out = []
    for r in rows:
        en_lines, en_seps = split_lines(r["english"])
        tw_lines, tw_seps = split_lines(r["translation"])
        if len(en_lines) != len(tw_lines):
            continue
        lengths = [len(tok(line, add_special_tokens=False).input_ids) for line in en_lines]
        for s, e in pack_lines(lengths, chunk_tokens):
            en, tw = join_span(en_lines, en_seps, s, e), join_span(tw_lines, tw_seps, s, e)
            results = glossary_for(en, lookup)
            out.append({"src": translate_input(en, results, prefix), "tgt": tw, "chunk": en, "results": results})
    return out


def disk_guard(path: str, min_free_gb: float):
    """Callback that saves and stops training cleanly when free disk space falls below min_free_gb
    (resume later from the last checkpoint once space is freed)."""
    from transformers import TrainerCallback

    class DiskGuard(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if control.should_save or state.global_step % 50:
                return
            free = shutil.disk_usage(path).free / 1e9
            if free < min_free_gb:
                print(f"DiskGuard: {free:.0f} GB free < {min_free_gb:.0f} GB; saving and stopping at step "
                      f"{state.global_step}", flush=True)
                control.should_save = True
                control.should_training_stop = True

    return DiskGuard()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/umt5-xl")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="checkpoints/umt5-xl-twi")
    ap.add_argument("--dict", default=GLOSSARY_DICT, help="glossary dictionary (default: one translation per phrase)")
    ap.add_argument("--chunk-tokens", type=int, default=192)
    ap.add_argument("--max-src", type=int, default=1024)
    ap.add_argument("--max-tgt", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lora-r", type=int, default=0, help="LoRA rank (0 = full fine-tune)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--early-stopping-patience", type=int, default=3)
    ap.add_argument("--save-total-limit", type=int, default=2,
                    help="2 keeps the best and the most recent checkpoint")
    ap.add_argument("--min-free-gb", type=float, default=40, help="stop cleanly below this much free disk")
    ap.add_argument("--gen-eval-chunks", type=int, default=200,
                    help="validation chunks translated at each eval for BLEU/chrF2 and the "
                         "composite that selects the best checkpoint (0 = select on eval_loss only)")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained(args.model)
    added = prepare_tokenizer(tok)
    langs = nllb_langs(tok)  # NLLB: language codes replace the T5-style task prefix
    if langs:
        tok.src_lang, tok.tgt_lang = langs
    prefix = "" if langs else TRANSLATE_PREFIX
    lookup = TwiLookup(args.dict)

    ds, gen_examples = {}, []
    for split in ("train", "validation"):
        with open(os.path.join(args.data, f"{split}.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        ex = build_examples(rows, tok, lookup, args.chunk_tokens, prefix)
        enc = tok([e["src"] for e in ex], text_target=[e["tgt"] for e in ex])
        keep = [i for i in range(len(ex))
                if len(enc["input_ids"][i]) <= args.max_src and len(enc["labels"][i]) <= args.max_tgt]
        ds[split] = Dataset.from_dict({k: [enc[k][i] for i in keep]
                                       for k in ("input_ids", "attention_mask", "labels")})
        print(f"{split}: {len(rows)} docs -> {len(ex)} examples, {len(ex) - len(keep)} over length dropped",
              flush=True)
        if split == "validation":
            gen_examples = sample_examples(ex, args.gen_eval_chunks) if args.gen_eval_chunks > 0 else []

    model = load_seq2seq(args.model, dtype=torch.bfloat16 if args.lora_r else torch.float32)
    if len(tok) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok))
    model.config.use_cache = False
    if args.lora_r:
        from peft import LoraConfig, get_peft_model

        new_ids = [tok.convert_tokens_to_ids("\n")] if added else []  # "\n" is new to the vocab: train its row
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="SEQ_2_SEQ_LM",
            target_modules=LORA_TARGETS, trainable_token_indices=new_ids or None))
        model.print_trainable_parameters()

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        optim="adamw_torch" if args.lora_r else "adafactor",
        lr_scheduler_type="linear",
        warmup_steps=200,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
    forced = {"forced_bos_token_id": tok.convert_tokens_to_ids(langs[1])} if langs else {}

    def gen_fn(m, examples):
        outs = batched_generate(m, tok, [e["src"] for e in examples], args.batch_size * 2, args.max_tgt, **forced)
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
    if args.lora_r:
        trainer.model.save_pretrained(os.path.join(args.out, "adapter"))
        trainer.model.merge_and_unload().save_pretrained(final)
    else:
        trainer.save_model(final)
    tok.save_pretrained(final)
    with open(os.path.join(args.out, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved model to {final}", flush=True)


if __name__ == "__main__":
    main()
